"""Preprocessing of burned-area polygon data.

This module provides a generic pipeline for converting raw burned-area
shapefiles from any geographic region into a standardised grid-based
sample map.

Input columns
-------------
The pipeline expects three semantic columns (fire ID, start date, end date)
plus geometry.  If the source file already uses the canonical names
(``src_fireid``, ``start_date``, ``end_date``), no extra configuration is
needed.  Otherwise, supply ``fire_id_col``, ``start_date_col``, and
``end_date_col`` so the pipeline can rename them on the fly — eliminating
the need for a separate ``rename_fields`` step.

Universal output schema
-----------------------
After processing, every output uses the following column names:

* ``start_date`` / ``end_date`` – fire temporal extent.
* ``src_fireid`` – original fire identifier from the source dataset.
* ``fire_id`` – sequential integer assigned after exploding multipolygons.
* ``grid_id`` – identifier of the spatial grid cell.
* ``id`` – sequential identifier for each unique (grid cell, time window).
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import hydra
import pandas as pd
import tqdm
from omegaconf import DictConfig

from data_preproc_script.constants import (
    CONFIG_PATH,
    KEY_COLUMNS,
)

#: Columns that **must** be present in the input burned-area file.
REQUIRED_INPUT_COLUMNS: frozenset[str] = frozenset(
    {"src_fireid", "start_date", "end_date"}
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate_required_columns(gdf: gpd.GeoDataFrame) -> None:
    """Check that *gdf* contains every required input column."""
    missing = REQUIRED_INPUT_COLUMNS - set(gdf.columns)
    if missing:
        raise ValueError(
            f"Input data is missing required columns: {sorted(missing)}. "
            f"Expected columns: {sorted(REQUIRED_INPUT_COLUMNS)}. "
            f"Available columns: {sorted(gdf.columns.tolist())}"
        )


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------
def prep_ba(
    input_path: os.PathLike[str] | str,
    out_path: os.PathLike[str] | str,
    grid_path: os.PathLike[str] | str,
    start_date: datetime,
    crs: int,
    fire_id_col: str | None = None,
    start_date_col: str | None = None,
    end_date_col: str | None = None,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess burned-area polygons and join them to a spatial grid.

    If the source file does not already use the canonical column names
    (``src_fireid``, ``start_date``, ``end_date``), pass *fire_id_col*,
    *start_date_col*, and *end_date_col* to rename them automatically.

    Parameters
    ----------
    input_path:
        Path to the raw burned-area shapefile / geodatabase.
    out_path:
        Destination for the filtered & exploded fire polygons.
    grid_path:
        Path to the spatial grid file produced by ``create_grid``.
    start_date:
        Only fires starting on or after this date are kept.
    crs:
        Target EPSG code (e.g. ``4326``).
    fire_id_col:
        Name of the fire-ID column in the source file.  Renamed to
        ``src_fireid``.  If *None* the column must already be named
        ``src_fireid``.
    start_date_col:
        Name of the start-date column in the source file.  Renamed to
        ``start_date``.  If *None* the column must already be named
        ``start_date``.
    end_date_col:
        Name of the end-date column in the source file.  Renamed to
        ``end_date``.  If *None* the column must already be named
        ``end_date``.

    Returns:
    -------
    tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]
        ``(download_map, final_mapping, preliminary_mapping)``
    """
    # ------------------------------------------------------------------
    # 1. Read source data, rename columns if needed, & validate
    # ------------------------------------------------------------------
    gdf = gpd.read_file(input_path)

    if crs is None:
        raise ValueError(f"CRS for input geodataframe '{input_path}' must not be None.")

    # Optional column renaming (replaces the separate rename_fields step)
    rename_map: dict[str, str] = {}
    if fire_id_col is not None:
        rename_map[fire_id_col] = "src_fireid"
    if start_date_col is not None:
        rename_map[start_date_col] = "start_date"
    if end_date_col is not None:
        rename_map[end_date_col] = "end_date"
    if rename_map:
        print(f"Renaming columns: {rename_map}")
        gdf = gdf.rename(columns=rename_map)

    validate_required_columns(gdf)

    # Date preprocessing
    gdf["start_date"] = pd.to_datetime(gdf["start_date"])
    gdf["end_date"] = pd.to_datetime(gdf["end_date"])
    gdf = gdf.dropna(subset=["start_date", "end_date"])
    gdf = gdf[gdf["start_date"] >= start_date]

    # Align CRS
    gdf = gdf.to_crs(epsg=crs)

    # Keep only universal columns
    gdf = gdf[["src_fireid", "start_date", "end_date", "geometry"]]

    # Explode multipolygons and assign sequential IDs
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf["fire_id"] = range(1, len(gdf) + 1)

    # ------------------------------------------------------------------
    # 2. Save filtered data
    # ------------------------------------------------------------------
    print(f"Saving filtered data to {out_path}...")
    _out_path = Path(out_path)
    _out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(_out_path)

    # ------------------------------------------------------------------
    # 3. Spatial join with grid
    # ------------------------------------------------------------------
    print(f"Loading grid from {grid_path} for spatial join...")
    grid_gdf = gpd.read_file(grid_path)

    print("Performing spatial join between burned area and grid...")
    joined = gpd.sjoin(grid_gdf, gdf, how="inner", predicate="intersects")
    joined = joined.drop(columns=["index_right"]).drop_duplicates()

    # Aggregate fire / polygon IDs per (grid cell, date range)
    joined = (
        joined.groupby(["id", "start_date", "end_date", "geometry"])
        .agg(
            {
                "fire_id": list,
                "src_fireid": list,
            }
        )
        .reset_index()
    )

    # ------------------------------------------------------------------
    # 4. Merge temporally overlapping fires within each grid cell
    # ------------------------------------------------------------------
    merged = _merge_overlapping_fires(joined, crs)

    # ------------------------------------------------------------------
    # 5. Build output tables
    # ------------------------------------------------------------------
    joined = joined.rename(columns={"id": "grid_id"})
    merged["id"] = range(1, len(merged) + 1)

    download_map = merged[
        [
            "id",
            "grid_id",
            "start_date",
            "end_date",
            "geometry",
            "minx",
            "miny",
            "maxx",
            "maxy",
            "center_x",
            "center_y",
        ]
    ]
    final_mapping = pd.DataFrame(
        merged[
            [
                "id",
                "grid_id",
                "start_date",
                "end_date",
                "fire_id",
                "src_fireid",
            ]
        ]
    )
    preliminary_mapping = joined[
        [
            "grid_id",
            "start_date",
            "end_date",
            "fire_id",
            "src_fireid",
        ]
    ]

    return download_map, final_mapping, preliminary_mapping


def _merge_overlapping_fires(
    joined: gpd.GeoDataFrame,
    crs: int,
) -> gpd.GeoDataFrame:
    """Merge temporally overlapping fires within each grid cell.

    Fires in the same grid cell whose date ranges overlap are coalesced
    into a single record.

    Parameters
    ----------
    joined:
        The spatially-joined GeoDataFrame (grid x fires).
    crs:
        EPSG code used for the output GeoDataFrame.

    Returns:
    -------
    gpd.GeoDataFrame
        Merged records with bounds and centroids appended.
    """
    joined = joined.sort_values(by=["id", "start_date"])

    merged_rows: list[list[Any]] = []

    cur_id: int | None = None
    cur_start: Any = None
    cur_end: Any = None
    cur_geom: Any = None
    cur_fire_ids: list[Any] = []
    cur_source_ids: list[Any] = []

    def _flush() -> None:
        """Append the current accumulated group to *merged_rows*."""
        merged_rows.append(
            [cur_id, cur_start, cur_end, cur_geom, cur_fire_ids, cur_source_ids]
        )

    for _, row in tqdm.tqdm(joined.iterrows(), total=len(joined)):
        if cur_id is None or row["id"] != cur_id:
            # New grid cell — flush the previous group (if any).
            if cur_id is not None:
                _flush()
            cur_id = row["id"]
            cur_start = row["start_date"]
            cur_end = row["end_date"]
            cur_geom = row["geometry"]
            cur_fire_ids = list(row["fire_id"])
            cur_source_ids = list(row["src_fireid"])

        elif row["start_date"] <= cur_end:
            # Overlapping fire — extend the current group.
            cur_end = max(cur_end, row["end_date"])
            cur_fire_ids += list(row["fire_id"])
            cur_source_ids += list(row["src_fireid"])

        else:
            # Non-overlapping fire in the same grid cell — start new group.
            _flush()
            cur_start = row["start_date"]
            cur_end = row["end_date"]
            cur_geom = row["geometry"]
            cur_fire_ids = list(row["fire_id"])
            cur_source_ids = list(row["src_fireid"])

    # Flush the last group
    if cur_id is not None:
        _flush()

    merged = gpd.GeoDataFrame(
        merged_rows,
        columns=[
            "grid_id",
            "start_date",
            "end_date",
            "geometry",
            "fire_id",
            "src_fireid",
        ],
        crs=crs,
    )

    # Append bounds and centroids
    merged = gpd.GeoDataFrame(
        pd.concat([merged, merged.bounds], axis=1),
        geometry="geometry",
        crs=crs,
    )
    merged["center_x"] = merged.centroid.x
    merged["center_y"] = merged.centroid.y

    return merged


# ---------------------------------------------------------------------------
# Optional region splitting
# ---------------------------------------------------------------------------
def region_split(
    gdf: gpd.GeoDataFrame,
    region_bounds: gpd.GeoDataFrame,
    region_col: str = "PRENAME",
) -> gpd.GeoDataFrame:
    """Assign each grid cell to a geographic region.

    When a grid cell overlaps multiple regions the one with the largest
    intersection area is kept.

    Parameters
    ----------
    gdf:
        The download-map GeoDataFrame (output of :func:`prep_ba`).
    region_bounds:
        A GeoDataFrame of region boundary polygons.  Must contain a column
        named *region_col* and a ``geometry`` column.
    region_col:
        Name of the column in *region_bounds* that holds region labels.

    Returns:
    -------
    gpd.GeoDataFrame
        *gdf* augmented with a ``region`` column.
    """
    # Align CRS
    region_bounds = region_bounds.to_crs(str(gdf.crs))

    # Spatial join between regions and grid cells
    gdf = gpd.sjoin(
        gdf,
        region_bounds[[region_col, "geometry"]],
        how="inner",
        predicate="intersects",
    )

    # Compute intersection area to resolve multi-region overlaps
    region_bounds = region_bounds.rename(columns={"geometry": "geometry_region"})
    gdf = gdf.merge(
        region_bounds[[region_col, "geometry_region"]],
        how="inner",
        on=region_col,
    )
    gdf["area"] = gdf.apply(
        lambda row: row["geometry"].intersection(row["geometry_region"]).area,
        axis=1,
    )

    # Keep the region with the largest overlap
    gdf = gdf.sort_values(
        by=["id", "start_date", "area"],
        ascending=[True, True, False],
    )
    gdf = gdf.drop_duplicates(subset=KEY_COLUMNS, keep="first")

    # Clean up
    gdf = gdf.drop(columns=["index_right", "geometry_region", "area"])
    gdf = gdf.rename(columns={region_col: "region"})

    return gdf


# ---------------------------------------------------------------------------
# CLI / Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(
    version_base=None, config_path=str(CONFIG_PATH), config_name="ba_preprocess"
)
def burned_area_preprocess(cfg: DictConfig) -> None:
    """Run the full burned-area preprocessing pipeline.

    Configuration is supplied via Hydra (``ba_preprocess.yaml``).
    If the source file uses non-standard column names, set
    ``fire_id_col``, ``start_date_col``, and ``end_date_col`` in the
    config so they are renamed automatically.
    """
    download_map, final_mapping, preliminary_mapping = prep_ba(
        input_path=cfg.input_path,
        out_path=cfg.filtered_fires_path,
        grid_path=cfg.grid_path,
        start_date=datetime.strptime(cfg.start_date, "%Y-%m-%d"),
        crs=int(cfg.crs),
        fire_id_col=cfg.get("fire_id_col", None),
        start_date_col=cfg.get("start_date_col", None),
        end_date_col=cfg.get("end_date_col", None),
    )

    # Optional region splitting
    if cfg.get("split_regions", False):
        print("Splitting data into regions...")
        region_bounds = gpd.read_file(cfg.region_bounds_path)
        download_map = region_split(
            download_map,
            region_bounds,
            region_col=cfg.get("region_col", "PRENAME"),
        )

    # Create output directories
    Path(cfg.merged_fire_grid_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.post_merge_fire_mapping_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.pre_merge_fire_mapping_path).parent.mkdir(parents=True, exist_ok=True)

    # Save
    print("Saving data...")
    download_map.to_file(cfg.merged_fire_grid_path)
    final_mapping.to_csv(cfg.post_merge_fire_mapping_path, index=False)
    preliminary_mapping.to_csv(cfg.pre_merge_fire_mapping_path, index=False)


if __name__ == "__main__":
    burned_area_preprocess()
