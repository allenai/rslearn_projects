"""Temporal binning of fire samples.

This module discretises the continuous fire dates from the burned-area
preprocessing step into fixed-length time windows.  For each spatial
grid cell, the fire's start date is snapped to the nearest preceding
bin boundary, producing labelled samples that answer:

    *"Was this cell burned in the next N days?"*

The bin width (``offset``) is configurable (e.g. 8 days).

Input
-----
The spatially-joined, temporally-merged fire–grid samples produced by
:mod:`burned_area` (saved at ``merged_fire_grid_path``).  Expected
columns include at least ``id``, ``grid_id``, ``start_date``,
``end_date``, and ``geometry``.

Output
------
* **Temporal grid samples** – deduplicated ``(grid_id, start_date)``
  pairs where ``start_date`` has been snapped to the temporal grid.
* **Temporal bin mapping** – for each ``(grid_id, snapped_start_date)``
  bin, the list of original sample IDs and their actual fire dates.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from data_preproc_script.constants import CONFIG_PATH


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def build_temporal_grid(
    start_year: int,
    end_year: int,
    offset: int,
) -> list[datetime]:
    """Generate evenly-spaced dates spanning *start_year* to *end_year*.

    Parameters
    ----------
    start_year:
        First calendar year (inclusive).
    end_year:
        Last calendar year (inclusive).
    offset:
        Step size in days between consecutive grid dates.

    Returns:
    -------
    list[datetime]
        Sorted list of bin-boundary dates.
    """
    dates: list[datetime] = []
    for year in range(start_year, end_year + 1):
        current = datetime(year, 1, 1)
        year_end = datetime(year, 12, 31)
        while current <= year_end:
            dates.append(current)
            current += timedelta(days=offset)
    return dates


# Keep the old name available for backward compatibility.
temp_grid = build_temporal_grid


def snap_to_preceding_date(
    grid_dates: list[datetime],
    date: datetime,
) -> datetime:
    """Return the latest grid date that falls on or before *date*.

    Parameters
    ----------
    grid_dates:
        Sorted list of candidate dates (bin boundaries).
    date:
        The date to snap.

    Returns:
    -------
    datetime
        The closest preceding (or equal) grid date.
    """
    return min(
        (d for d in grid_dates if d <= date),
        key=lambda d: abs(d - date),
    )


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------
def temporal_bin(
    input_path: os.PathLike[str] | str,
    samples_out_path: os.PathLike[str] | str,
    mapping_out_path: os.PathLike[str] | str,
    start_year: int,
    end_year: int,
    offset: int,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Snap fire samples to a fixed temporal grid and deduplicate.

    Parameters
    ----------
    input_path:
        Path to the spatially-joined fire–grid samples produced by
        :func:`burned_area.prep_ba` (the ``merged_fire_grid_path``).
    samples_out_path:
        Destination for the deduplicated temporal-grid samples.
    mapping_out_path:
        Destination CSV for the temporal-bin mapping.
    start_year:
        First calendar year for the temporal grid.
    end_year:
        Last calendar year for the temporal grid.
    offset:
        Bin width in days (e.g. ``8``).

    Returns:
    -------
    tuple[gpd.GeoDataFrame, pd.DataFrame]
        ``(samples, bin_mapping)``
    """
    # ------------------------------------------------------------------
    # 1. Load fire–grid samples
    # ------------------------------------------------------------------
    print(f"Loading fire–grid samples from {input_path} ...")
    gdf = gpd.read_file(input_path)

    # ------------------------------------------------------------------
    # 2. Build the temporal grid and snap dates
    # ------------------------------------------------------------------
    print("Building temporal grid ...")
    grid_dates = build_temporal_grid(start_year, end_year, offset)

    # Preserve original fire dates before overwriting start_date
    gdf = gdf.rename(
        columns={"start_date": "start_date_fire", "end_date": "end_date_fire"},
    )
    # Ensure fire dates are proper datetimes (source data may store them as strings)
    gdf["start_date_fire"] = pd.to_datetime(gdf["start_date_fire"])
    gdf["end_date_fire"] = pd.to_datetime(gdf["end_date_fire"])

    tqdm.pandas(desc="Snapping fire dates to temporal grid")
    gdf["start_date"] = gdf["start_date_fire"].progress_apply(
        lambda d: snap_to_preceding_date(grid_dates, d)
    )

    # ------------------------------------------------------------------
    # 3. Deduplicate: one entry per (grid_id, temporal bin)
    # ------------------------------------------------------------------
    samples = (
        gdf.drop(columns=["start_date_fire", "end_date_fire"])
        .sort_values(by=["id"])
        .drop_duplicates(subset=["grid_id", "start_date"], keep="first")
    )

    # ------------------------------------------------------------------
    # 4. Build bin mapping: (grid_id, snapped_date) → original IDs & dates
    # ------------------------------------------------------------------
    bin_mapping = (
        gdf.groupby(["grid_id", "start_date"])
        .agg({"id": list, "start_date_fire": list, "end_date_fire": list})
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    Path(samples_out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(mapping_out_path).parent.mkdir(parents=True, exist_ok=True)

    print("Saving temporal-grid samples and bin mapping ...")
    samples.to_file(samples_out_path)
    bin_mapping.to_csv(mapping_out_path, index=False)

    return samples, bin_mapping


# ---------------------------------------------------------------------------
# CLI / Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(
    version_base=None, config_path=str(CONFIG_PATH), config_name="ba_preprocess"
)
def temporal_preprocess(cfg: DictConfig) -> None:
    """Run temporal binning of fire samples.

    Configuration is supplied via Hydra (``ba_preprocess.yaml``).
    """
    temporal_bin(
        input_path=cfg.merged_fire_grid_path,
        samples_out_path=cfg.temporal_grid_samples_path,
        mapping_out_path=cfg.temporal_bin_mapping_path,
        start_year=cfg.start_year,
        end_year=cfg.end_year,
        offset=cfg.offset,
    )


if __name__ == "__main__":
    temporal_preprocess()
