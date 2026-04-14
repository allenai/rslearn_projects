"""Unified negative sampling with three tiers.

This module draws negative (non-fire) grid-cell samples in three equal tiers,
all balanced by region and year:

1. **Random / FWI-uniform** -- equal draws from FWI quantile buckets of the
   candidate pool (no LC or month stratification).
2. **LC + month matched** -- negatives whose land-cover and month distribution
   matches the positives, with hierarchical backoff when strata are sparse.
3. **LC + month + FWI-hard** -- like tier 2 but also matches the FWI
   distribution of positives (optionally biased toward higher FWI).

All three tiers preserve regional balancing: each region receives
``sampling_ratio / 3 * n_positives_in_region`` negatives per tier.

Inputs
------
* Positive samples GeoDataFrame (with ``grid_id``, ``start_date``, ``lc``,
  ``region``, ``geometry``).
* Full spatial grid GeoDataFrame (with ``id``, ``lc``, ``geometry``).
* FWI NetCDF per year (aggregated to the same temporal grid as positives).
* Region boundary shapefile / GeoDatabase.

Output
------
A single GeoDataFrame / GDB with all negatives, tagged with a ``tier``
column (``"random"``, ``"lc_month"``, ``"lc_month_fwi"``).
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import geopandas as gpd
import hydra
import pandas as pd
import xarray as xr
from omegaconf import DictConfig
from shapely import box
from tqdm import tqdm

from data_preproc_script.constants import CONFIG_PATH, EE_CRS
from data_preproc_script.preprocess.temporal_grid_agg import build_temporal_grid
from data_preproc_script.utils import assign_val, create_logger

logger = create_logger("neg_sampling_unified", "logs/neg_sampling_unified")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _month_to_season(month: int) -> str:
    """Map calendar month (1--12) to a coarse season label."""
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _load_fwi_year(
    fwi_base_path: str,
    year: int,
    bounds: tuple[float, float, float, float],
    min_date: datetime,
    max_date: datetime,
) -> gpd.GeoDataFrame:
    """Load aggregated FWI NetCDF for *year* and return polygon GeoDataFrame.

    Parameters
    ----------
    fwi_base_path:
        Directory containing ``{year}/fwi_dc_agg_{year}.nc``.
    year:
        Calendar year.
    bounds:
        ``(minx, miny, maxx, maxy)`` for spatial clipping (speed-up).
    min_date, max_date:
        Temporal range filter.
    """
    fwi_file = Path(fwi_base_path) / str(year) / f"fwi_dc_agg_{year}.nc"
    logger.info(f"Loading FWI from {fwi_file}")
    fwi_ds = xr.open_dataset(fwi_file)
    fwi_df = fwi_ds.to_dataframe().reset_index()
    fwi_df = fwi_df.dropna(subset=["fwinx_mean"])
    fwi_df = fwi_df[
        (fwi_df["valid_time"] >= min_date) & (fwi_df["valid_time"] <= max_date)
    ]

    # Spatial clip using bounds + margin (FWI cell half-width)
    minx, miny, maxx, maxy = bounds
    margin = 0.25
    fwi_df = fwi_df[
        (fwi_df["longitude"] >= minx - margin)
        & (fwi_df["longitude"] <= maxx + margin)
        & (fwi_df["latitude"] >= miny - margin)
        & (fwi_df["latitude"] <= maxy + margin)
    ]

    fwi_gdf = gpd.GeoDataFrame(
        fwi_df,
        geometry=gpd.points_from_xy(fwi_df["longitude"], fwi_df["latitude"]),
        crs=EE_CRS,
    )
    hl = 0.25 / 2
    fwi_gdf["geometry"] = box(
        fwi_gdf["longitude"].values - hl,
        fwi_gdf["latitude"].values - hl,
        fwi_gdf["longitude"].values + hl,
        fwi_gdf["latitude"].values + hl,
    )
    return fwi_gdf


def _assign_regions(
    gdf: gpd.GeoDataFrame,
    region_bounds: gpd.GeoDataFrame,
    region_col: str,
) -> gpd.GeoDataFrame:
    """Assign a ``region`` column via spatial join (largest-overlap wins)."""
    gdf = gdf.copy()
    gdf["_orig_idx"] = gdf.index

    rb = region_bounds[[region_col, "geometry"]].copy()
    rb["geometry_region"] = rb["geometry"]
    joined = gpd.sjoin(gdf, rb, how="inner", predicate="intersects")
    joined["_overlap_area"] = joined.apply(
        lambda r: r["geometry"].intersection(r["geometry_region"]).area, axis=1
    )
    joined = joined.sort_values("_overlap_area", ascending=False).drop_duplicates(
        subset="_orig_idx", keep="first"
    )
    joined = joined.drop(
        columns=["geometry_region", "_overlap_area", "index_right", "_orig_idx"],
        errors="ignore",
    )
    joined = joined.rename(columns={region_col: "region"})
    return joined


# ---------------------------------------------------------------------------
# Per-stratum backoff sampler
# ---------------------------------------------------------------------------
def _sample_stratum_with_backoff(
    pool: pd.DataFrame,
    target: int,
    filter_chain: list[dict[str, object]],
    seed: int,
) -> pd.DataFrame:
    """Draw *target* rows from *pool*, trying progressively coarser filters.

    Parameters
    ----------
    pool:
        Candidate rows.
    target:
        Desired sample count.
    filter_chain:
        Ordered list of ``{column: value}`` dicts, from most specific to
        least.  Each dict defines a filter on *pool*.
    seed:
        Random seed.

    Returns:
    -------
    pd.DataFrame
        Up to *target* rows sampled from *pool*.
    """
    collected: list[pd.DataFrame] = []
    used: set[int] = set()
    remaining = target

    for filt in filter_chain:
        if remaining <= 0:
            break
        mask = pd.Series(True, index=pool.index)
        for col, val in filt.items():
            if col in pool.columns and val is not None and pd.notna(val):
                mask &= pool[col] == val
        mask &= ~pool.index.isin(used)
        available = pool[mask]
        take = min(remaining, len(available))
        if take > 0:
            s = available.sample(n=take, random_state=seed)
            collected.append(s)
            used.update(s.index)
            remaining -= take
            seed += 1

    # Final fallback: random from anything not yet used
    if remaining > 0:
        leftover = pool[~pool.index.isin(used)]
        take = min(remaining, len(leftover))
        if take > 0:
            collected.append(leftover.sample(n=take, random_state=seed))

    if not collected:
        return pool.iloc[:0]
    return pd.concat(collected, ignore_index=True)


# ---------------------------------------------------------------------------
# Tier samplers
# ---------------------------------------------------------------------------
def _sample_tier1_fwi_uniform(
    pool: pd.DataFrame,
    n_target: int,
    num_buckets: int,
    seed: int,
) -> pd.DataFrame:
    """Tier 1: FWI-uniform random -- equal draws from FWI quantile buckets."""
    if len(pool) == 0 or n_target <= 0:
        return pool.iloc[:0]

    pool = pool.copy()
    try:
        pool["_fwi_qbin"] = pd.qcut(
            pool["fwinx_mean"], num_buckets, labels=False, duplicates="drop"
        )
    except ValueError:
        pool["_fwi_qbin"] = 0

    actual_buckets = pool["_fwi_qbin"].nunique()
    per_bucket = math.ceil(n_target / max(actual_buckets, 1))

    sampled = pool.groupby("_fwi_qbin", group_keys=False).apply(
        lambda g: g.sample(n=min(per_bucket, len(g)), random_state=seed)
    )

    # Trim to exact target if we overshot
    if len(sampled) > n_target:
        sampled = sampled.sample(n=n_target, random_state=seed)

    sampled = sampled.drop(columns=["_fwi_qbin"], errors="ignore")
    sampled["tier"] = "random"
    return sampled


def _sample_tier2_lc_month(
    pool: pd.DataFrame,
    positives: pd.DataFrame,
    n_target: int,
    seed: int,
) -> pd.DataFrame:
    """Tier 2: LC + month matched -- match positive ``(lc, month)`` distribution."""
    if len(pool) == 0 or n_target <= 0:
        return pool.iloc[:0]

    has_lc = "lc" in pool.columns and "lc" in positives.columns

    if not has_lc:
        logger.warning("No 'lc' column; tier 2 falls back to month-only matching")
        pos_counts = positives.groupby("month").size().reset_index(name="count")
        total = pos_counts["count"].sum()
        month_parts: list[pd.DataFrame] = []
        for _, row in pos_counts.iterrows():
            stratum_n = max(1, round(row["count"] / total * n_target))
            chain: list[dict[str, object]] = [{"month": row["month"]}, {}]
            month_parts.append(
                _sample_stratum_with_backoff(pool, stratum_n, chain, seed)
            )
            seed += 1
        result = (
            pd.concat(month_parts, ignore_index=True) if month_parts else pool.iloc[:0]
        )
        if len(result) > n_target:
            result = result.sample(n=n_target, random_state=seed)
        result["tier"] = "lc_month"
        return result

    # --- Full LC + month matching ---
    pos_counts = positives.groupby(["lc", "month"]).size().reset_index(name="count")
    total = pos_counts["count"].sum()
    parts: list[pd.DataFrame] = []
    for _, row in pos_counts.iterrows():
        stratum_n = max(1, round(row["count"] / total * n_target))
        lc_month_chain: list[dict[str, object]] = [
            {"lc": row["lc"], "month": row["month"]},  # exact
            {"lc": row["lc"]},  # LC only
            {"month": row["month"]},  # month only
            {},  # random
        ]
        parts.append(
            _sample_stratum_with_backoff(pool, stratum_n, lc_month_chain, seed)
        )
        seed += 1

    result = pd.concat(parts, ignore_index=True) if parts else pool.iloc[:0]
    if len(result) > n_target:
        result = result.sample(n=n_target, random_state=seed)
    result["tier"] = "lc_month"
    return result


def _sample_tier3_fwi_hard(
    pool: pd.DataFrame,
    positives: pd.DataFrame,
    n_target: int,
    fwi_bins: list[float],
    fwi_bias_factor: float,
    seed: int,
) -> pd.DataFrame:
    """Tier 3: LC + month + FWI-hard -- match positive distribution, optional FWI bias."""
    if len(pool) == 0 or n_target <= 0:
        return pool.iloc[:0]

    pool = pool.copy()
    positives = positives.copy()

    # Bin FWI on both sides
    pool["fwi_bin"] = pd.cut(
        pool["fwinx_mean"], bins=fwi_bins, labels=False, include_lowest=True
    )
    positives["fwi_bin"] = pd.cut(
        positives["fwinx_mean"], bins=fwi_bins, labels=False, include_lowest=True
    )

    # Coarse season for backoff
    pool["season"] = pool["month"].apply(_month_to_season)
    positives["season"] = positives["month"].apply(_month_to_season)

    # Drop positives without FWI (could not be spatially matched)
    positives = positives.dropna(subset=["fwi_bin"])

    has_lc = "lc" in pool.columns and "lc" in positives.columns
    strata_cols = (["lc"] if has_lc else []) + ["month", "fwi_bin"]

    pos_counts = positives.groupby(strata_cols).size().reset_index(name="count")

    # Optional upward FWI bias: inflate counts for higher FWI bins
    if fwi_bias_factor > 1.0:
        max_bin = pos_counts["fwi_bin"].max()
        if pd.notna(max_bin) and max_bin > 0:
            pos_counts["count"] = pos_counts.apply(
                lambda r: r["count"]
                * (1 + (fwi_bias_factor - 1) * r["fwi_bin"] / max_bin),
                axis=1,
            )

    total = pos_counts["count"].sum()
    if total == 0:
        sampled = pool.sample(n=min(n_target, len(pool)), random_state=seed)
        sampled = sampled.drop(columns=["fwi_bin", "season"], errors="ignore")
        sampled["tier"] = "lc_month_fwi"
        return sampled

    parts: list[pd.DataFrame] = []
    for _, row in pos_counts.iterrows():
        stratum_n = max(1, round(row["count"] / total * n_target))
        season = _month_to_season(int(row["month"]))

        if has_lc:
            chain: list[dict[str, object]] = [
                {"lc": row["lc"], "month": row["month"], "fwi_bin": row["fwi_bin"]},
                {"lc": row["lc"], "month": row["month"]},
                {"lc": row["lc"], "season": season, "fwi_bin": row["fwi_bin"]},
                {"lc": row["lc"]},
                {},
            ]
        else:
            chain = [
                {"month": row["month"], "fwi_bin": row["fwi_bin"]},
                {"month": row["month"]},
                {"season": season, "fwi_bin": row["fwi_bin"]},
                {},
            ]
        parts.append(_sample_stratum_with_backoff(pool, stratum_n, chain, seed))
        seed += 1

    result = pd.concat(parts, ignore_index=True) if parts else pool.iloc[:0]
    if len(result) > n_target:
        result = result.sample(n=n_target, random_state=seed)
    result = result.drop(columns=["fwi_bin", "season"], errors="ignore")
    result["tier"] = "lc_month_fwi"
    return result


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------
def _postprocess_tier_samples(
    tier_samples: pd.DataFrame,
    region_bounds: gpd.GeoDataFrame,
    region_col: str,
) -> gpd.GeoDataFrame:
    """Rename columns, add bounds/centroids, assign region via spatial join."""
    tier_samples = tier_samples.rename(
        columns={"valid_time": "start_date", "id": "grid_id"}
    )

    keep = ["grid_id", "start_date", "fwinx_mean", "geometry", "tier"]
    if "lc" in tier_samples.columns:
        keep.append("lc")
    tier_samples = tier_samples[[c for c in keep if c in tier_samples.columns]]

    tier_samples = gpd.GeoDataFrame(tier_samples, geometry="geometry", crs=EE_CRS)

    # Add bounds and centroids
    tier_samples = pd.concat([tier_samples, tier_samples.bounds], axis=1)
    tier_samples["center_x"] = tier_samples.geometry.centroid.x
    tier_samples["center_y"] = tier_samples.geometry.centroid.y

    # Assign region via spatial join (handles border cells: largest overlap wins).
    # Samples that don't intersect any region polygon are dropped (inner join).
    n_before_sjoin = len(tier_samples)
    rb = region_bounds.copy()
    rb["geometry_region"] = rb["geometry"]
    tier_samples = gpd.sjoin(
        tier_samples,
        rb[[region_col, "geometry_region", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    tier_samples["_area"] = tier_samples.apply(
        lambda x: x["geometry"].intersection(x["geometry_region"]).area, axis=1
    )
    tier_samples = tier_samples.sort_values(
        by=["grid_id", "start_date", "_area"], ascending=[True, True, False]
    )
    tier_samples = tier_samples.drop_duplicates(
        subset=["grid_id", "start_date"], keep="first"
    )
    n_dropped = n_before_sjoin - len(tier_samples)
    if n_dropped > 0:
        logger.warning(f"  Dropped {n_dropped} samples outside all region boundaries")
    tier_samples = tier_samples.drop(
        columns=["geometry_region", "_area", "index_right"], errors="ignore"
    )
    tier_samples = tier_samples.rename(columns={region_col: "region"})
    return tier_samples


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def sample_negatives(
    pos_grid_path: str,
    grid_path: str,
    fwi_base_path: str,
    region_bounds_path: str,
    region_col: str,
    output_path: str,
    start_year: int,
    end_year: int,
    offset: int = 8,
    sampling_ratio: float = 2.0,
    fwi_num_buckets: int = 10,
    fwi_bins: list[float] | None = None,
    fwi_bias_factor: float = 1.0,
    seed: int = 42,
    select_year: int | None = None,
    oversample_pct: float = 5.0,
) -> None:
    """Draw three-tier negative samples, saving one file per year.

    Each year's output is written to ``{stem}_{year}{suffix}`` derived from
    *output_path*.  Use :func:`merge_yearly_outputs` afterwards to combine
    them into a single deduplicated file.

    Parameters
    ----------
    pos_grid_path:
        Positive samples (temporally-gridded fire samples, ideally with
        ``lc`` already assigned).
    grid_path:
        Full spatial grid with ``id``, ``lc``, ``geometry``.
    fwi_base_path:
        Directory with ``{year}/fwi_dc_agg_{year}.nc``.
    region_bounds_path:
        Region boundary file (shapefile, GDB, GeoPackage, ...).
    region_col:
        Column in region bounds identifying region names.
    output_path:
        Base path for output files.  Per-year files are derived from this
        (e.g. ``negatives.gdb`` -> ``negatives_2018.gdb``).
    start_year, end_year:
        Year range (inclusive).  Also used for deterministic seed
        computation even when *select_year* restricts execution to one
        year.
    offset:
        Temporal grid step in days.
    sampling_ratio:
        Total negative-to-positive ratio (split equally among 3 tiers).
    fwi_num_buckets:
        Number of FWI quantile buckets for tier 1.
    fwi_bins:
        Fixed FWI bin edges for tier 3 (default: 10 bins from 0 to 100).
    fwi_bias_factor:
        >1.0 biases tier 3 toward higher FWI bins.
    seed:
        Random seed.
    select_year:
        If given, process only this year (for parallel execution across
        machines).  Seeds remain deterministic w.r.t. the full range.
    oversample_pct:
        Percentage by which to inflate per-tier targets to compensate for
        cross-year ``grid_id`` deduplication in the merge step.
    """
    # ------------------------------------------------------------------
    # 0. Load reference data
    # ------------------------------------------------------------------
    print(f"Loading positive samples from {pos_grid_path} ...")
    pos_samples = gpd.read_file(pos_grid_path)
    pos_samples["start_date"] = pd.to_datetime(pos_samples["start_date"])

    print(f"Loading spatial grid from {grid_path} ...")
    grid = gpd.read_file(grid_path)

    print(f"Loading region bounds from {region_bounds_path} ...")
    region_bounds = gpd.read_file(region_bounds_path)
    region_bounds = region_bounds.to_crs(epsg=4326)

    if fwi_bins is None:
        fwi_bins = [0, 2, 4, 6, 10, 15, 20, 25, 30, 40, 100]

    # Ensure positives have a region column
    if "region" not in pos_samples.columns:
        print("Assigning regions to positive samples via spatial join ...")
        pos_samples = _assign_regions(pos_samples, region_bounds, region_col)

    # If positives don't have LC, try to get from grid
    if "lc" not in pos_samples.columns and "lc" in grid.columns:
        print("Joining LC from grid onto positive samples ...")
        lc_map = grid[["id", "lc"]].rename(columns={"id": "grid_id"})
        pos_samples = pos_samples.merge(lc_map, on="grid_id", how="left")

    total_bounds = cast(
        tuple[float, float, float, float], tuple(region_bounds.total_bounds)
    )

    prior_neg_ids: set[int] = set()
    out_base = Path(output_path)
    oversample_factor = 1.0 + oversample_pct / 100.0

    # ------------------------------------------------------------------
    # 1. Year loop
    # ------------------------------------------------------------------
    years_to_process = (
        [select_year]
        if select_year is not None
        else list(range(start_year, end_year + 1))
    )
    for year in tqdm(years_to_process, desc="Years"):
        logger.info(f"===== Year {year} =====")
        print(f"\n--- Year {year} ---")
        year_neg: list[pd.DataFrame] = []

        year_start = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31)

        year_pos = pos_samples[
            (pos_samples["start_date"] >= year_start)
            & (pos_samples["start_date"] <= year_end)
        ]
        if len(year_pos) == 0:
            logger.info(f"No positives for year {year}, skipping")
            continue

        # All positives up to end of this year (for cell exclusion)
        up_to_year_pos = pos_samples[pos_samples["start_date"] <= year_end]

        min_date = year_pos["start_date"].min()
        max_date = year_pos["start_date"].max()
        logger.info(f"Positive date range: {min_date} -- {max_date}")

        # Load FWI data for this year
        fwi_gdf = _load_fwi_year(fwi_base_path, year, total_bounds, min_date, max_date)

        # Verify that FWI valid_time values sit on the expected temporal grid.
        # The FWI NetCDF should have been pre-aggregated with the same offset
        # (see fwi_agg in utils.py / fwi.yaml temp_offset).
        expected_dates = set(build_temporal_grid(year, year, offset))
        fwi_dates = set(fwi_gdf["valid_time"].dt.to_pydatetime())
        misaligned = fwi_dates - expected_dates
        if misaligned:
            logger.warning(
                f"FWI contains {len(misaligned)} dates not on the "
                f"{offset}-day temporal grid (year {year}). "
                f"First few: {sorted(misaligned)[:5]}. "
                f"Check that fwi.yaml temp_offset == sampling offset."
            )

        # --------------------------------------------------------------
        # 2. Region loop
        # --------------------------------------------------------------
        regions = year_pos["region"].unique()
        for region_idx, region in enumerate(
            tqdm(regions, desc=f"Regions ({year})", leave=False)
        ):
            logger.info(f"Region: {region}")

            # Vary seed per (year, region) so each combination draws
            # independently.  Tier offsets (0 / 10k / 20k) are large enough
            # that the per-year/region shift (max ~1030) never collides.
            yr_rgn_seed = seed + (year - start_year) * 100 + region_idx

            region_geo = region_bounds[region_bounds[region_col] == region]
            if len(region_geo) == 0:
                logger.warning(f"Region '{region}' not in bounds file, skipping")
                continue

            # Grid cells in this region
            region_grid = grid[
                grid["geometry"].intersects(region_geo.geometry.values[0])
            ]

            # Exclude cells with fires (up to this year) or prior negatives
            fire_ids = set(
                up_to_year_pos.loc[
                    up_to_year_pos["region"] == region, "grid_id"
                ].unique()
            )
            exclude_ids = fire_ids | prior_neg_ids
            pool_grid = region_grid[~region_grid["id"].isin(exclude_ids)].copy()

            if len(pool_grid) == 0:
                logger.warning(f"No candidate cells in {region}, skipping")
                continue

            logger.info(
                f"  Grid cells in region: {len(region_grid)}, "
                f"available after exclusion: {len(pool_grid)}"
            )

            # Enrich pool: spatial join with FWI -> (cell, date, fwi) rows
            key_cols = ["id", "valid_time"]
            if "lc" in pool_grid.columns:
                key_cols.append("lc")
            pool_enriched = assign_val(pool_grid, fwi_gdf, key_cols, "fwinx_mean")

            if len(pool_enriched) == 0:
                logger.warning(f"No FWI-enriched candidates in {region}, skipping")
                continue

            pool_enriched["month"] = pool_enriched["valid_time"].dt.month
            pool_enriched["season"] = pool_enriched["month"].apply(_month_to_season)
            logger.info(
                f"  Enriched pool size (cell x date pairs): " f"{len(pool_enriched)}"
            )

            # Prepare positives for this region (with month + FWI)
            region_pos = year_pos[year_pos["region"] == region].copy()
            region_pos["month"] = region_pos["start_date"].dt.month

            # Assign FWI to positives for tier 3 strata computation
            if "fwinx_mean" not in region_pos.columns:
                pos_key = ["grid_id", "start_date"]
                if "lc" in region_pos.columns:
                    pos_key.append("lc")
                pos_fwi = assign_val(
                    region_pos[pos_key + ["geometry"]].copy(),
                    fwi_gdf,
                    pos_key,
                    "fwinx_mean",
                    temp_cols=["start_date", "valid_time"],
                )
                if len(pos_fwi) > 0:
                    region_pos = region_pos.merge(
                        pos_fwi[
                            ["grid_id", "start_date", "fwinx_mean"]
                        ].drop_duplicates(),
                        on=["grid_id", "start_date"],
                        how="left",
                    )

            n_pos = len(region_pos)
            n_per_tier = math.ceil(sampling_ratio / 3 * n_pos * oversample_factor)
            logger.info(f"  {n_pos} positives -> {n_per_tier} negatives per tier")

            # --- Tier 1: FWI-uniform random ---
            t1 = _sample_tier1_fwi_uniform(
                pool_enriched, n_per_tier, fwi_num_buckets, yr_rgn_seed
            )

            # --- Tier 2: LC + month matched ---
            t2 = _sample_tier2_lc_month(
                pool_enriched, region_pos, n_per_tier, yr_rgn_seed + 10_000
            )

            # --- Tier 3: LC + month + FWI-hard ---
            t3 = _sample_tier3_fwi_hard(
                pool_enriched,
                region_pos,
                n_per_tier,
                fwi_bins,
                fwi_bias_factor,
                yr_rgn_seed + 20_000,
            )

            # Concat tiers, deduplicate across tiers, and post-process.
            # The same (cell, date) can be drawn by multiple tiers; keep the
            # first occurrence (tier priority: random > lc_month > lc_month_fwi).
            tier_samples = pd.concat([t1, t2, t3], ignore_index=True)
            pre_dedup = len(tier_samples)
            tier_samples = tier_samples.drop_duplicates(
                subset=["id", "valid_time"], keep="first"
            )
            if len(tier_samples) < pre_dedup:
                logger.info(
                    f"  Dropped {pre_dedup - len(tier_samples)} cross-tier "
                    f"duplicates"
                )
            tier_samples = _postprocess_tier_samples(
                tier_samples, region_bounds, region_col
            )
            logger.info(f"  Sampled {len(tier_samples)} negatives total")
            year_neg.append(tier_samples)
            prior_neg_ids.update(tier_samples["grid_id"].unique())

        # -- Save per-year file ------------------------------------------
        if not year_neg:
            logger.warning(f"No negative samples for year {year}")
            continue

        year_result = pd.concat(year_neg, ignore_index=True)
        year_result = gpd.GeoDataFrame(year_result, geometry="geometry", crs=EE_CRS)
        year_out = out_base.parent / f"{out_base.stem}_{year}{out_base.suffix}"
        year_out.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(year_result)} negatives for {year} to {year_out}")
        year_result.to_file(str(year_out))
        logger.info(f"Saved {len(year_result)} negatives for {year}")

        tier_counts = year_result["tier"].value_counts()
        print(f"  Tier breakdown ({year}):")
        for tier_name, count in tier_counts.items():
            print(f"    {tier_name}: {count}")


# ---------------------------------------------------------------------------
# Merge utility
# ---------------------------------------------------------------------------
def merge_yearly_outputs(
    output_path: str,
    pos_grid_path: str,
    start_year: int,
    end_year: int,
) -> gpd.GeoDataFrame:
    """Merge per-year negative-sample files into a single deduplicated file.

    Reads ``{stem}_{year}{suffix}`` for each year in the range, deduplicates
    on ``grid_id`` (keeping the earliest year's samples), re-assigns
    contiguous IDs, and saves to *output_path*.

    Parameters
    ----------
    output_path:
        Base output path (same as passed to :func:`sample_negatives`).
    pos_grid_path:
        Positive samples file (used to read ``max(id)`` for ID assignment).
    start_year, end_year:
        Year range (inclusive).
    """
    out_base = Path(output_path)
    frames: list[pd.DataFrame] = []

    print("Scanning per-year files ...")
    for year in range(start_year, end_year + 1):
        year_file = out_base.parent / f"{out_base.stem}_{year}{out_base.suffix}"
        if not year_file.exists():
            print(f"  {year}: not found ({year_file}), skipping")
            continue
        gdf = gpd.read_file(str(year_file))
        gdf["_year"] = year
        frames.append(gdf)
        print(f"  {year}: {len(gdf)} samples")

    if not frames:
        logger.warning("No per-year files found -- nothing to merge")
        return gpd.GeoDataFrame()

    result = pd.concat(frames, ignore_index=True)
    pre_dedup = len(result)

    # For cells appearing in multiple years, keep only the earliest year
    result = result.sort_values("_year")
    earliest = result.groupby("grid_id")["_year"].transform("min")
    dup_mask = result["_year"] != earliest
    n_deduped = int(dup_mask.sum())
    if n_deduped:
        print(
            f"  Removed {n_deduped} cross-year grid_id duplicates "
            f"({pre_dedup} -> {pre_dedup - n_deduped})"
        )
    result = result[~dup_mask]
    result = result.drop(columns=["_year"])

    # Assign unique IDs (continuing from max positive ID)
    pos_samples = gpd.read_file(pos_grid_path)
    max_pos_id = int(pos_samples["id"].max())
    result["id"] = range(max_pos_id + 1, max_pos_id + 1 + len(result))

    result = gpd.GeoDataFrame(result, geometry="geometry", crs=EE_CRS)

    out_base.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(result)} merged negatives to {output_path}")
    result.to_file(output_path)
    logger.info(f"Saved {len(result)} merged negatives to {output_path}")

    tier_counts = result["tier"].value_counts()
    print("\nTier breakdown (merged):")
    for tier_name, count in tier_counts.items():
        print(f"  {tier_name}: {count}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_exec_args() -> argparse.Namespace:
    """Parse execution-control CLI args before Hydra processes sys.argv.

    These flags are separate from the Hydra/YAML configuration so they can
    be freely combined with any config without modifying YAML files.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--select-year",
        type=int,
        default=None,
        help="Process only this year (for parallel execution on clusters).",
    )
    parser.add_argument(
        "--oversample-pct",
        type=float,
        default=5.0,
        help="Pct to oversample to compensate for cross-year dedup (default 5).",
    )
    parser.add_argument(
        "--merge-mode",
        action="store_true",
        default=False,
        help="Merge per-year outputs instead of sampling.",
    )
    exec_args, remaining = parser.parse_known_args()
    sys.argv = sys.argv[:1] + remaining
    return exec_args


_EXEC_ARGS: argparse.Namespace | None = None


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_PATH),
    config_name="sampling_unified",
)
def main(cfg: DictConfig) -> None:
    """Run unified three-tier negative sampling (Hydra entry point)."""
    if _EXEC_ARGS is not None and _EXEC_ARGS.merge_mode:
        print("Running in merge mode ...")
        merge_yearly_outputs(
            output_path=cfg.output_path,
            pos_grid_path=cfg.pos_grid_path,
            start_year=cfg.start_year,
            end_year=cfg.end_year,
        )
    else:
        select_year = _EXEC_ARGS.select_year if _EXEC_ARGS else None
        oversample_pct = _EXEC_ARGS.oversample_pct if _EXEC_ARGS else 5.0
        sample_negatives(
            pos_grid_path=cfg.pos_grid_path,
            grid_path=cfg.grid_path,
            fwi_base_path=cfg.fwi_base_path,
            region_bounds_path=cfg.region_bounds_path,
            region_col=cfg.region_col,
            output_path=cfg.output_path,
            start_year=cfg.start_year,
            end_year=cfg.end_year,
            offset=cfg.offset,
            sampling_ratio=cfg.sampling_ratio,
            fwi_num_buckets=cfg.fwi_num_buckets,
            fwi_bins=list(cfg.fwi_bins),
            fwi_bias_factor=cfg.fwi_bias_factor,
            seed=cfg.seed,
            select_year=select_year,
            oversample_pct=oversample_pct,
        )


if __name__ == "__main__":
    _EXEC_ARGS = _parse_exec_args()
    main()
