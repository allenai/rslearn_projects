"""Sampling distribution analysis.

Compare positive and negative sample distributions across FWI and land
cover.  Zone-agnostic -- works for any geographic region given the
appropriate config paths.

Analyses
--------
* **FWI** -- overlay histograms of FWI values for positives vs negatives
  (per year and aggregated).  Also outputs FWI quantile bin edges from
  the positive distribution, useful for setting ``fwi_bins`` in the
  negative-sampling config.
* **Land cover** -- bar charts of LC class counts for positives vs
  negatives (per year and aggregated).
* **Land cover grid** -- assign the modal LC class to every cell in a
  spatial grid or sample GDB (prerequisite for LC-stratified sampling).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import cast

import geopandas as gpd
import hydra
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import seaborn as sns
import xarray as xr
from omegaconf import DictConfig
from scipy.stats import mode
from shapely.geometry import Polygon
from tqdm import tqdm

from data_preproc_script.constants import CONFIG_PATH, EE_CRS
from data_preproc_script.utils import assign_val, create_logger

sns.set_theme(style="whitegrid")

logger = create_logger("analyze_sampling", "logs/analyze_sampling")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _create_fwi_square(row: gpd.GeoSeries, side_length: float = 0.25) -> Polygon:
    """Build a square polygon centred on a point (FWI grid cell)."""
    lon, lat = row.geometry.x, row.geometry.y
    hl = side_length / 2
    return Polygon(
        [
            (lon - hl, lat - hl),
            (lon + hl, lat - hl),
            (lon + hl, lat + hl),
            (lon - hl, lat + hl),
        ]
    )


def _load_fwi_year(
    fwi_base_path: str | os.PathLike,
    year: int,
    bounds: tuple[float, float, float, float],
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> gpd.GeoDataFrame:
    """Load pre-aggregated FWI NetCDF for *year*, clip, and polygonise."""
    fwi_file = Path(fwi_base_path) / str(year) / f"fwi_dc_agg_{year}.nc"
    logger.info(f"Loading FWI from {fwi_file}")

    fwi_df = (
        xr.open_dataset(fwi_file)
        .to_dataframe()
        .reset_index()
        .dropna(subset=["fwinx_mean"])
    )
    fwi_df = fwi_df[
        (fwi_df["valid_time"] >= min_date) & (fwi_df["valid_time"] <= max_date)
    ]

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
    fwi_gdf["geometry"] = fwi_gdf.apply(_create_fwi_square, axis=1)
    return fwi_gdf


def _save_plot(fig: plt.Figure, path: os.PathLike) -> None:
    """Save a matplotlib figure and close it."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved plot to {path}")
    plt.close(fig)


def _plot_fwi_histogram(
    pos_fwi: pd.Series,
    neg_fwi: pd.Series,
    output_file: os.PathLike,
    n_bins: int = 100,
) -> None:
    """Overlay histograms of positive and negative FWI distributions."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(pos_fwi, ax=ax, color="steelblue", label="Positive", bins=n_bins)
    sns.histplot(neg_fwi, ax=ax, color="indianred", label="Negative", bins=n_bins)
    ax.set_xlabel("FWI", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    _save_plot(fig, output_file)


def _plot_lc_bars(
    pos_lc: pd.Series,
    neg_lc: pd.Series,
    output_file: os.PathLike,
) -> None:
    """Side-by-side bar chart of positive vs negative land-cover counts."""
    pos_counts = pos_lc.value_counts().reset_index()
    pos_counts.columns = ["lc", "count"]
    pos_counts["source"] = "Positive"

    neg_counts = neg_lc.value_counts().reset_index()
    neg_counts.columns = ["lc", "count"]
    neg_counts["source"] = "Negative"

    combined = pd.concat([pos_counts, neg_counts])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=combined, x="lc", y="count", hue="source", ax=ax)
    ax.set_xlabel("Land Cover Class", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    _save_plot(fig, output_file)


def _extract_modal_lc(
    gdf: gpd.GeoDataFrame,
    lc_path: str | os.PathLike,
    id_col: str = "grid_id",
    desc: str = "Extracting LC",
) -> pd.DataFrame:
    """Return a ``(id_col, lc)`` DataFrame with the modal LC per unique cell.

    The GeoDataFrame is temporarily reprojected to the raster CRS so that
    geometry bounds match the raster coordinate space.
    """
    unique_cells = gdf[[id_col, "geometry"]].drop_duplicates(subset=[id_col]).copy()

    with rasterio.open(lc_path) as src:
        if unique_cells.crs and unique_cells.crs != src.crs:
            unique_cells = unique_cells.to_crs(src.crs)

        lc_values: list[int] = []
        for polygon in tqdm(unique_cells.geometry, desc=desc, total=len(unique_cells)):
            window = src.window(*polygon.bounds)
            data = src.read(1, window=window).flatten()
            if src.nodata is not None:
                data = data[data != src.nodata]
            if len(data) > 0:
                lc_values.append(int(mode(data, keepdims=True).mode[0]))
            else:
                lc_values.append(-1)

    unique_cells["lc"] = lc_values
    return unique_cells[[id_col, "lc"]]


# ---------------------------------------------------------------------------
# FWI analysis
# ---------------------------------------------------------------------------
def fwi_analysis(
    pos_path: str | os.PathLike,
    fwi_base_path: str | os.PathLike,
    output_path: str | os.PathLike,
    neg_path: str | os.PathLike | None = None,
    num_bucket: int = 10,
) -> None:
    """Analyse FWI distributions for positive (and optionally negative) samples.

    Always computes and writes the positive-FWI quantile bin edges to
    ``pos_fwi_quantiles.txt`` (useful for setting ``fwi_bins`` in the
    negative-sampling config).  When *neg_path* is provided, also produces
    per-year and aggregated comparison histograms.
    """
    logger.info("Starting FWI analysis")
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    pos_grid = gpd.read_file(pos_path)
    pos_grid["start_date"] = pd.to_datetime(pos_grid["start_date"])

    has_neg = neg_path is not None and str(neg_path).strip() != ""
    neg_grid: gpd.GeoDataFrame | None = None
    if has_neg:
        neg_grid = gpd.read_file(neg_path)
        neg_grid["start_date"] = pd.to_datetime(neg_grid["start_date"])
        assert "fwinx_mean" in neg_grid.columns, "Negative samples must have fwinx_mean"
    else:
        logger.info("No neg_path provided — skipping comparative plots")

    # Derive spatial bounds from the positive samples (zone-agnostic).
    total_bounds = cast(tuple[float, float, float, float], tuple(pos_grid.total_bounds))

    years = sorted(pos_grid["start_date"].dt.year.unique())
    logger.info(f"Found {len(years)} years to analyse: {years}")

    all_pos_with_fwi: list[pd.DataFrame] = []

    for year in years:
        logger.info(f"Processing year {year}")
        pos_year = pos_grid[pos_grid["start_date"].dt.year == year]

        min_date = pos_year["start_date"].min()
        max_date = pos_year["start_date"].max()

        # Assign FWI to positives via area-weighted spatial join with the
        # aggregated FWI grid (same approach as negative_sampling.py).
        if "fwinx_mean" not in pos_year.columns:
            fwi_gdf = _load_fwi_year(
                fwi_base_path, year, total_bounds, min_date, max_date
            )
            unique_pos = pos_year[["grid_id", "geometry"]].drop_duplicates()
            pos_fwi = assign_val(
                unique_pos, fwi_gdf, ["grid_id", "valid_time"], "fwinx_mean"
            )
            pos_fwi = pos_fwi.rename(columns={"valid_time": "start_date"})
            pos_year = pos_year.merge(
                pos_fwi[["grid_id", "start_date", "fwinx_mean"]].drop_duplicates(),
                on=["grid_id", "start_date"],
                how="inner",
            )

        all_pos_with_fwi.append(pos_year)

        if has_neg:
            assert neg_grid is not None
            neg_year = neg_grid[neg_grid["start_date"].dt.year == year]
            _plot_fwi_histogram(
                pos_year["fwinx_mean"],
                neg_year["fwinx_mean"],
                out / f"fwi_distribution_{year}.pdf",
            )

    # Aggregated plot across all years
    all_pos = pd.concat(all_pos_with_fwi)

    if has_neg:
        assert neg_grid is not None
        _plot_fwi_histogram(
            all_pos["fwinx_mean"],
            neg_grid["fwinx_mean"],
            out / "fwi_distribution_all_years.pdf",
        )

    # Compute and save quantile bin edges from the positive distribution.
    _, bin_edges = pd.qcut(
        all_pos["fwinx_mean"], num_bucket, retbins=True, labels=False
    )
    quantile_file = out / "pos_fwi_quantiles.txt"
    with open(quantile_file, "w") as f:
        for i, edge in enumerate(bin_edges):
            f.write(f"Quantile {i}: {edge}\n")
        f.write("\nSuggested fwi_bins list for sampling config:\n")
        edges_list = [round(float(e), 2) for e in bin_edges]
        f.write(f"fwi_bins: {edges_list}\n")
    logger.info(f"Saved FWI quantiles to {quantile_file}")

    logger.info("FWI analysis completed")


# ---------------------------------------------------------------------------
# Land cover analysis
# ---------------------------------------------------------------------------
def land_cover_analysis(
    pos_path: str | os.PathLike,
    lc_path: str | os.PathLike,
    output_path: str | os.PathLike,
    neg_path: str | os.PathLike | None = None,
) -> None:
    """Analyse land-cover distributions for positive (and optionally negative) samples.

    When *neg_path* is provided, produces per-year and aggregated
    comparative bar charts.  Otherwise only extracts and logs the
    positive LC distribution.
    """
    logger.info("Starting land cover analysis")
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    pos_grid = gpd.read_file(pos_path)
    pos_grid["start_date"] = pd.to_datetime(pos_grid["start_date"])

    has_neg = neg_path is not None and str(neg_path).strip() != ""
    neg_grid: gpd.GeoDataFrame | None = None
    if has_neg:
        neg_grid = gpd.read_file(neg_path)
        neg_grid["start_date"] = pd.to_datetime(neg_grid["start_date"])
    else:
        logger.info("No neg_path provided — skipping comparative plots")

    # Extract modal LC per unique grid cell (handles CRS reprojection
    # internally using the geometry bounds, not stale column values).
    logger.info(f"Extracting LC for {pos_grid['grid_id'].nunique()} positive cells")
    pos_lc_map = _extract_modal_lc(pos_grid, lc_path, desc="Positive LC")
    pos_grid = pos_grid.merge(pos_lc_map, on="grid_id", how="inner")

    if has_neg:
        assert neg_grid is not None
        if "lc" in neg_grid.columns:
            neg_grid = neg_grid.drop(columns=["lc"])
        logger.info(f"Extracting LC for {neg_grid['grid_id'].nunique()} negative cells")
        neg_lc_map = _extract_modal_lc(neg_grid, lc_path, desc="Negative LC")
        neg_grid = neg_grid.merge(neg_lc_map, on="grid_id", how="inner")

    # Per-year plots
    years = sorted(pos_grid["start_date"].dt.year.unique())
    logger.info(f"Found {len(years)} years to analyse: {years}")
    for year in years:
        logger.info(f"Processing year {year}")
        if has_neg:
            assert neg_grid is not None
            _plot_lc_bars(
                pos_grid.loc[pos_grid["start_date"].dt.year == year, "lc"],
                neg_grid.loc[neg_grid["start_date"].dt.year == year, "lc"],
                out / f"lc_distribution_{year}.pdf",
            )

    # Aggregated plot
    if has_neg:
        assert neg_grid is not None
        _plot_lc_bars(
            pos_grid["lc"], neg_grid["lc"], out / "lc_distribution_all_years.pdf"
        )
    logger.info("Land cover analysis completed")


# ---------------------------------------------------------------------------
# Land cover grid assignment
# ---------------------------------------------------------------------------
def land_cover_grid(
    grid_path: str | os.PathLike,
    lc_path: str | os.PathLike,
    output_path: str | os.PathLike,
) -> None:
    """Assign the modal land-cover class to every cell in a spatial grid.

    Run this on:

    1. The full spatial grid  -->  ``{grid_name}_lc.gdb``
    2. The positive samples GDB  -->  ``{samples_name}_lc.gdb``

    The outputs provide the ``lc`` column needed for LC-stratified
    negative sampling.
    """
    logger.info("Starting land cover grid computation")
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    grid = gpd.read_file(grid_path)
    original_crs = grid.crs or "epsg:4326"
    logger.info(f"Processing {len(grid)} grid cells")

    with rasterio.open(lc_path) as src:
        raster_crs = src.crs
        logger.info(f"Raster CRS: {raster_crs}, Grid CRS: {grid.crs}")
        if grid.crs != raster_crs:
            grid = grid.to_crs(raster_crs)

        lc_values: list[int] = []
        skipped = 0
        for polygon in tqdm(grid.geometry, desc="Grid LC", total=len(grid)):
            window = src.window(*polygon.bounds)
            data = src.read(1, window=window).flatten()
            if src.nodata is not None:
                data = data[data != src.nodata]
            if len(data) > 0:
                lc_values.append(int(mode(data, keepdims=True).mode[0]))
            else:
                lc_values.append(-1)
                skipped += 1

    grid["lc"] = lc_values
    if grid.crs != original_crs:
        grid = grid.to_crs(original_crs)

    logger.info(f"Cells with no valid LC data: {skipped}/{len(grid)}")

    output_file = out / f"{Path(grid_path).stem}_lc.gdb"
    if output_file.exists():
        shutil.rmtree(output_file)
    grid.to_file(output_file)
    logger.info(f"Saved grid with land cover to {output_file}")


# ---------------------------------------------------------------------------
# Hydra CLI
# ---------------------------------------------------------------------------
@hydra.main(
    version_base=None, config_path=str(CONFIG_PATH), config_name="analyze_sampling"
)
def main(cfg: DictConfig) -> None:
    """Run sampling analysis (Hydra entry point)."""
    logger.info("Starting sampling analysis")

    neg_path = cfg.get("neg_path", None) or None

    if cfg.run_fwi_analysis:
        fwi_analysis(
            pos_path=cfg.pos_path,
            fwi_base_path=cfg.fwi_base_path,
            output_path=cfg.output_path,
            neg_path=neg_path,
            num_bucket=cfg.num_bucket,
        )

    if cfg.run_land_cover_analysis:
        land_cover_analysis(
            pos_path=cfg.pos_path,
            lc_path=cfg.lc_path,
            output_path=cfg.output_path,
            neg_path=neg_path,
        )

    if cfg.run_land_cover_grid:
        land_cover_grid(
            grid_path=cfg.grid_path,
            lc_path=cfg.lc_path,
            output_path=cfg.output_path,
        )

    logger.info("Sampling analysis completed")


if __name__ == "__main__":
    main()
