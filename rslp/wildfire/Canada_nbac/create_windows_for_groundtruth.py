"""Create rslearn windows from the burned-area preprocessing pipeline.

Reads positive fire samples and negative (non-fire) samples produced by the
preprocessing pipeline (data_preproc_script), retrieves fire polygon
geometries for labelling, and creates rslearn Windows with raster labels.

Input files (from burned_area_preproc/):
- temporally_gridded_fire_samples.gdb: positive samples (grid_id, start_date, geometry)
- negative_samples_unified.gdb: negative samples (grid_id, start_date, geometry)
- temporal_bin_sample_mapping.csv: (grid_id, start_date) -> merged sample IDs
- post_merge_fire_sample_mapping.csv: merged sample ID -> fire_id list
- filtered_exploded_fires.gdb: fire_id -> polygon geometry

Windows are created with:
- Time range: [start_date - lookback_days, start_date]
- Sentinel-1, Sentinel-2, Landsat data sources (configured in rslearn dataset config)
- Raster label layer (1=fire, 0=background) from fire polygons

Splits are assigned by year:
- val: 2022
- test: 2023
- train: all other years
"""

import ast
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import shapely
from pyproj import Transformer
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.utils.geometry import Projection
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import (
    GeotiffRasterFormat,
    get_transform_from_projection_and_bounds,
)
from upath import UPath

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PREPROC_DIR = Path("./datasets/Canada_nbac/burned_area_preproc")
DS_PATH = UPath(f"{os.environ['RSLEARN_EAI_ROOT']}/datasets/wildfire/canada_nbac")

POSITIVE_SAMPLES_PATH = PREPROC_DIR / "temporally_gridded_fire_samples.gdb"
NEGATIVE_SAMPLES_PATH = PREPROC_DIR / "negative_samples_unified.gdb"

TEMPORAL_BIN_MAPPING_PATH = PREPROC_DIR / "temporal_bin_sample_mapping.csv"
POST_MERGE_MAPPING_PATH = PREPROC_DIR / "post_merge_fire_sample_mapping.csv"
FILTERED_FIRES_PATH = PREPROC_DIR / "filtered_exploded_fires.gdb"

# ---------------------------------------------------------------------------
# Split configuration (year-based)
# ---------------------------------------------------------------------------
VAL_YEARS: set[int] = {2022}
TEST_YEARS: set[int] = {2023}

# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
NATIVE_RESOLUTION = 10  # meters per pixel


def get_split(year: int) -> str:
    """Assign a sample to train/val/test based on its year."""
    if year in VAL_YEARS:
        return "val"
    if year in TEST_YEARS:
        return "test"
    return "train"


# ---------------------------------------------------------------------------
# Fire polygon lookup
# ---------------------------------------------------------------------------
class FirePolygonLookup:
    """Lookup fire polygons for a (grid_id, start_date) sample.

    Uses three files from the preprocessing pipeline:
    1. temporal_bin_sample_mapping.csv: (grid_id, start_date) -> merged sample IDs
    2. post_merge_fire_sample_mapping.csv: merged sample ID -> fire_id list
    3. filtered_exploded_fires.gdb: fire_id -> polygon geometry
    """

    def __init__(
        self,
        temporal_bin_mapping_path: Path,
        post_merge_mapping_path: Path,
        filtered_fires_path: Path,
    ) -> None:
        """Load the lookup tables needed to map windows back to source fires."""
        # (grid_id, start_date_str) -> list of merged sample IDs
        print(f"Loading temporal bin mapping from {temporal_bin_mapping_path} ...")
        bin_df = pd.read_csv(temporal_bin_mapping_path)
        bin_df["start_date"] = pd.to_datetime(bin_df["start_date"])
        self.bin_to_ids: dict[tuple[int, str], list[int]] = {}
        for _, row in bin_df.iterrows():
            key = (int(row["grid_id"]), row["start_date"].strftime("%Y-%m-%d"))
            ids = row["id"]
            if isinstance(ids, str):
                ids = ast.literal_eval(ids)
            self.bin_to_ids[key] = ids
        print(f"  Indexed {len(self.bin_to_ids)} (grid_id, start_date) bins")

        # merged sample ID -> list of fire_ids
        print(f"Loading post-merge fire mapping from {post_merge_mapping_path} ...")
        merge_df = pd.read_csv(post_merge_mapping_path)
        self.id_to_fire_ids: dict[int, list[int]] = {}
        for _, row in merge_df.iterrows():
            fire_ids = row["fire_id"]
            if isinstance(fire_ids, str):
                fire_ids = ast.literal_eval(fire_ids)
            self.id_to_fire_ids[int(row["id"])] = fire_ids
        print(f"  Indexed {len(self.id_to_fire_ids)} merged-sample -> fire_id mappings")

        # fire_id -> list of polygon geometries
        print(f"Loading fire polygons from {filtered_fires_path} ...")
        fires_gdf = gpd.read_file(filtered_fires_path)
        fires_gdf = fires_gdf.set_crs("EPSG:4326", allow_override=True)
        self.fire_id_to_polygons: dict[int, list[shapely.Geometry]] = {}
        for _, row in fires_gdf.iterrows():
            fid = int(row["fire_id"])
            self.fire_id_to_polygons.setdefault(fid, []).append(row["geometry"])
        print(f"  Indexed {len(self.fire_id_to_polygons)} unique fire_ids")

    def get_fire_polygons(
        self,
        grid_id: int,
        start_date: datetime,
        clip_bounds_wgs84: tuple[float, float, float, float] | None = None,
    ) -> list[shapely.Geometry]:
        """Return fire polygons for a (grid_id, start_date) bin.

        Args:
            grid_id: spatial grid cell ID.
            start_date: snapped bin start date.
            clip_bounds_wgs84: optional (min_lon, min_lat, max_lon, max_lat) to
                clip polygons to the window extent.

        Returns:
            List of fire polygon geometries (WGS84), optionally clipped.
        """
        key = (grid_id, start_date.strftime("%Y-%m-%d"))
        merged_ids = self.bin_to_ids.get(key, [])
        if not merged_ids:
            return []

        all_fire_ids: set[int] = set()
        for mid in merged_ids:
            all_fire_ids.update(self.id_to_fire_ids.get(mid, []))

        if not all_fire_ids:
            return []

        polygons: list[shapely.Geometry] = []
        for fid in all_fire_ids:
            polygons.extend(self.fire_id_to_polygons.get(fid, []))

        if not polygons:
            return []

        if clip_bounds_wgs84 is not None:
            clip_box = shapely.box(*clip_bounds_wgs84)
            clipped = []
            for poly in polygons:
                if poly.intersects(clip_box):
                    c = poly.intersection(clip_box)
                    if not c.is_empty:
                        clipped.append(c)
            return clipped

        return polygons


# ---------------------------------------------------------------------------
# Grid size
# ---------------------------------------------------------------------------
def compute_grid_size(
    center_lon: float,
    center_lat: float,
    geometry: shapely.Geometry,
    label_resolution: int = 100,
) -> int:
    """Derive the grid side length in pixels from a grid cell geometry.

    Converts the geometry bounds to the local UTM zone and divides the
    larger of (width, height) by NATIVE_RESOLUTION. The result is rounded
    to the nearest multiple of ``label_resolution // NATIVE_RESOLUTION``
    so that ``grid_size * NATIVE_RESOLUTION`` is always divisible by
    ``label_resolution``.
    """
    utm_proj = get_utm_ups_projection(
        center_lon, center_lat, NATIVE_RESOLUTION, -NATIVE_RESOLUTION
    )
    transformer = Transformer.from_crs("EPSG:4326", utm_proj.crs, always_xy=True)

    minx, miny, maxx, maxy = geometry.bounds
    utm_min_x, utm_min_y = transformer.transform(minx, miny)
    utm_max_x, utm_max_y = transformer.transform(maxx, maxy)

    width_m = abs(utm_max_x - utm_min_x)
    height_m = abs(utm_max_y - utm_min_y)
    raw = round(max(width_m, height_m) / NATIVE_RESOLUTION)

    scale = label_resolution // NATIVE_RESOLUTION
    return round(raw / scale) * scale


# ---------------------------------------------------------------------------
# Raster label writing
# ---------------------------------------------------------------------------
def write_raster_label(
    window: Window,
    fire_polygons: list[shapely.Geometry],
    win_pixel_bounds: tuple[int, int, int, int],
    win_utm_crs: str,
    is_positive: bool,
    grid_size: int,
    label_resolution: int = 100,
) -> None:
    """Write raster label for a window at a specified resolution.

    Creates a single-band raster where 1 = fire/burned area, 0 = background.
    """
    assert NATIVE_RESOLUTION * grid_size % label_resolution == 0

    label_side_size = int(grid_size * NATIVE_RESOLUTION / label_resolution)
    scale = label_resolution // NATIVE_RESOLUTION

    label_pix_bounds = (
        win_pixel_bounds[0] // scale,
        win_pixel_bounds[1] // scale,
        win_pixel_bounds[2] // scale,
        win_pixel_bounds[3] // scale,
    )

    label_projection = Projection(
        crs=win_utm_crs,
        x_resolution=label_resolution,
        y_resolution=-label_resolution,
    )

    label_raster = np.zeros((label_side_size, label_side_size), dtype=np.uint8)

    if is_positive and fire_polygons:
        transformer = Transformer.from_crs("EPSG:4326", win_utm_crs, always_xy=True)
        utm_shapes = [
            (shapely.ops.transform(transformer.transform, poly), 1)
            for poly in fire_polygons
        ]

        transform = get_transform_from_projection_and_bounds(
            label_projection, label_pix_bounds
        )
        label_raster = rasterio.features.rasterize(
            utm_shapes,
            out_shape=(label_side_size, label_side_size),
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )

    label_array = RasterArray(chw_array=label_raster[np.newaxis, :, :])

    label_layer = f"label_{label_resolution}m"
    raster_dir = window.get_raster_dir(label_layer, [label_layer])
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        label_projection,
        label_pix_bounds,
        label_array,
    )
    window.mark_layer_completed(label_layer)


# ---------------------------------------------------------------------------
# Sample processing
# ---------------------------------------------------------------------------
def process_samples(
    samples: gpd.GeoDataFrame,
    fire_lookup: FirePolygonLookup,
    lookback_days: int = 64,
    offset_days: int = 0,
    grid_size_override: int | None = None,
    label_resolution: int = 100,
    max_samples: int | None = None,
) -> int:
    """Create rslearn windows for all samples (positive and negative).

    Returns the number of windows created.
    """
    if max_samples is not None:
        samples = samples.head(max_samples)
        print(f"  (Limited to {max_samples} samples)")

    windows_created = 0
    positive_count = 0
    negative_count = 0
    positive_with_polygons = 0
    positive_without_polygons = 0

    for _, row in samples.iterrows():
        is_positive = bool(row["is_positive"])
        grid_id = int(row["grid_id"])

        start_date = row["start_date"]
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.to_pydatetime()
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)

        split = get_split(start_date.year)

        center_lon = float(row["center_x"])
        center_lat = float(row["center_y"])

        if grid_size_override is not None:
            grid_size = grid_size_override
        else:
            grid_size = compute_grid_size(
                center_lon, center_lat, row["geometry"], label_resolution
            )

        utm_projection = get_utm_ups_projection(
            center_lon, center_lat, NATIVE_RESOLUTION, -NATIVE_RESOLUTION
        )

        transformer = Transformer.from_crs(
            "EPSG:4326", utm_projection.crs, always_xy=True
        )
        center_utm_x, center_utm_y = transformer.transform(center_lon, center_lat)

        half_size_m = grid_size * NATIVE_RESOLUTION / 2
        window_size_m = grid_size * NATIVE_RESOLUTION

        initial_min_x = center_utm_x - half_size_m
        initial_min_y = center_utm_y - half_size_m

        aligned_min_x = round(initial_min_x / label_resolution) * label_resolution
        aligned_min_y = round(initial_min_y / label_resolution) * label_resolution

        win_utm_bounds = (
            aligned_min_x,
            aligned_min_y,
            aligned_min_x + window_size_m,
            aligned_min_y + window_size_m,
        )

        # y_resolution is negative (-10): UTM max_y -> smaller pixel row
        win_pixel_bounds = (
            int(win_utm_bounds[0] / utm_projection.x_resolution),
            int(win_utm_bounds[3] / utm_projection.y_resolution),
            int(win_utm_bounds[2] / utm_projection.x_resolution),
            int(win_utm_bounds[1] / utm_projection.y_resolution),
        )

        fire_polygons: list[shapely.Geometry] = []
        if is_positive:
            transformer_inv = Transformer.from_crs(
                utm_projection.crs, "EPSG:4326", always_xy=True
            )
            wgs84_min_lon, wgs84_min_lat = transformer_inv.transform(
                win_utm_bounds[0], win_utm_bounds[1]
            )
            wgs84_max_lon, wgs84_max_lat = transformer_inv.transform(
                win_utm_bounds[2], win_utm_bounds[3]
            )

            fire_polygons = fire_lookup.get_fire_polygons(
                grid_id=grid_id,
                start_date=start_date,
                clip_bounds_wgs84=(
                    wgs84_min_lon,
                    wgs84_min_lat,
                    wgs84_max_lon,
                    wgs84_max_lat,
                ),
            )
            if fire_polygons:
                positive_with_polygons += 1
            else:
                positive_without_polygons += 1

        time_start = start_date - timedelta(days=lookback_days)
        time_end = start_date + timedelta(days=offset_days)

        region = str(row.get("region", "unknown")).replace(" ", "_")
        sample_type = "POS" if is_positive else "NEG"
        window_name = (
            f"{region}_{grid_id}_{sample_type}_{start_date.strftime('%Y%m%d')}"
        )

        window_storage = FileWindowStorage(DS_PATH)
        window = Window(
            storage=window_storage,
            group=split,
            name=window_name,
            projection=utm_projection,
            bounds=win_pixel_bounds,
            time_range=(time_start, time_end),
            options={
                "split": split,
                "region": region,
                "grid_id": grid_id,
                "is_positive": is_positive,
                "start_date": start_date.isoformat(),
                "num_fire_polygons": len(fire_polygons),
                "grid_size": grid_size,
            },
        )
        window.save()

        write_raster_label(
            window,
            fire_polygons,
            win_pixel_bounds,
            str(utm_projection.crs),
            is_positive,
            grid_size=grid_size,
            label_resolution=label_resolution,
        )

        windows_created += 1
        if is_positive:
            positive_count += 1
        else:
            negative_count += 1

        if windows_created % 100 == 0:
            print(f"  Created {windows_created} windows...")

    print(f"Finished: {windows_created} windows created")
    print(
        f"  Positive: {positive_count} "
        f"({positive_with_polygons} with polygons, "
        f"{positive_without_polygons} without)"
    )
    print(f"  Negative: {negative_count}")

    return windows_created


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(
    lookback_days: int = 64,
    offset_days: int = 0,
    grid_size: int | None = None,
    label_resolution: int = 100,
    max_samples: int | None = None,
    test_mode: bool = False,
    splits: list[str] | None = None,
) -> None:
    """Create label rasters for the requested Canada NBAC splits."""
    DS_PATH.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading fire polygon lookup data...")
    print("=" * 60)
    fire_lookup = FirePolygonLookup(
        temporal_bin_mapping_path=TEMPORAL_BIN_MAPPING_PATH,
        post_merge_mapping_path=POST_MERGE_MAPPING_PATH,
        filtered_fires_path=FILTERED_FIRES_PATH,
    )

    print("\n" + "=" * 60)
    print("Loading samples...")
    print("=" * 60)
    pos_samples = gpd.read_file(POSITIVE_SAMPLES_PATH)
    pos_samples["is_positive"] = True
    print(f"  Positive samples: {len(pos_samples)}")

    neg_samples = gpd.read_file(NEGATIVE_SAMPLES_PATH)
    neg_samples["is_positive"] = False
    print(f"  Negative samples: {len(neg_samples)}")

    common_cols = [
        "grid_id",
        "start_date",
        "geometry",
        "center_x",
        "center_y",
        "region",
        "is_positive",
    ]
    pos_subset = pos_samples[[c for c in common_cols if c in pos_samples.columns]]
    neg_subset = neg_samples[[c for c in common_cols if c in neg_samples.columns]]
    all_samples = gpd.GeoDataFrame(
        pd.concat([pos_subset, neg_subset], ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    print(f"  Total samples: {len(all_samples)}")

    if test_mode:
        max_samples = max_samples or 10
        print(f"TEST MODE: limiting to {max_samples} samples")

    all_samples["start_date"] = pd.to_datetime(all_samples["start_date"])
    all_samples["_split"] = all_samples["start_date"].dt.year.apply(get_split)
    print("\nSplit distribution (all data):")
    for split_name in ["train", "val", "test"]:
        mask = all_samples["_split"] == split_name
        n_pos = (mask & all_samples["is_positive"]).sum()
        n_neg = (mask & ~all_samples["is_positive"]).sum()
        print(f"  {split_name}: {n_pos} positive, {n_neg} negative")

    if splits is not None:
        all_samples = all_samples[all_samples["_split"].isin(splits)].copy()
        print(f"\nFiltered to splits {splits}: {len(all_samples)} samples")
    all_samples = all_samples.drop(columns=["_split"])

    print(f"\nLabel resolution: {label_resolution}m (layer: label_{label_resolution}m)")
    print(f"Lookback: {lookback_days} days, Offset: {offset_days} days")
    if grid_size is not None:
        print(f"Grid size override: {grid_size} pixels")
    else:
        print("Grid size: dynamically derived from geometry")
    print("=" * 60)

    total_windows = process_samples(
        all_samples,
        fire_lookup,
        lookback_days=lookback_days,
        offset_days=offset_days,
        grid_size_override=grid_size,
        label_resolution=label_resolution,
        max_samples=max_samples,
    )

    print(f"\n{'=' * 60}")
    print(f"Total windows created: {total_windows}")
    print("\nNext steps:")
    print("1. Materialize Sentinel-1, Sentinel-2, and Landsat data:")
    print(f"   rslearn dataset prepare --root {DS_PATH} --workers 32 --batch-size 8")
    print(
        f"   rslearn dataset ingest --root {DS_PATH} --workers 32 --no-use-initial-job"
    )
    print(
        f"   rslearn dataset materialize --root {DS_PATH} --workers 32"
        f" --no-use-initial-job"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create rslearn windows from preprocessed burned-area data"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (limited samples)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max total samples to process",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=64,
        help="Days of satellite imagery before bin start date (default: 64)",
    )
    parser.add_argument(
        "--offset-days",
        type=int,
        default=0,
        help="Days after bin start date (default: 0, window ends at start_date)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Override grid side length in pixels (default: derived from geometry)",
    )
    parser.add_argument(
        "--label-resolution",
        type=int,
        default=100,
        help="Label raster resolution in meters (default: 100)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=None,
        help="Only create windows for these splits (default: all)",
    )
    args = parser.parse_args()

    main(
        lookback_days=args.lookback_days,
        offset_days=args.offset_days,
        grid_size=args.grid_size,
        label_resolution=args.label_resolution,
        max_samples=args.max_samples,
        test_mode=args.test,
        splits=args.splits,
    )
