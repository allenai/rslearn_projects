"""Create rslearn windows with dense target labels from MapBiomas coverage rasters.

Each window is target_size x target_size pixels at 10m resolution. A patch of
(target_size / 3) pixels is read from the corresponding year's MapBiomas raster
(~30m) and upscaled 3x via nearest-neighbor to fill the target. Omitted classes
are remapped to no-data (0); windows where all pixels are no-data after
remapping are skipped.

Usage::

    python -m rslp.mapbiomas.create_windows_dense_raster \
        --ds-name mapbiomas_3k \
        --omit-classes 31 0
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from rslp.utils.windows import calculate_bounds

MY_ROOT = Path(os.environ.get("MY_ROOT", "."))

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label_raster"
BAND_NAME = "label"
NODATA_VALUE = 0
GROUP_NAME = "mapbiomas_dense_raster"
UPSCALE_FACTOR = 3

DEFAULT_CSV = (
    MY_ROOT / "rslearn_projects/rslp/mapbiomas/subsampling/sample_dense_raster_4k.csv"
)
DEFAULT_RASTER_DIR = MY_ROOT / "datasets/mapbiomas/data"
DEFAULT_DS_NAME = "mapbiomas_3k"


def create_window(
    csv_row: pd.Series,
    ds_path: UPath,
    target_size: int,
    omit_classes: set[int],
    raster_dir: Path,
) -> str:
    """Create one dense-label window from a MapBiomas coverage raster patch.

    Returns a status string: "created", "omit_class", or "omit_nodata".
    """
    target_id = int(csv_row["TARGETID"])
    year = int(csv_row["YEAR"])
    split = csv_row["split"]
    window_name = f"{target_id}_{year}"

    # LON/LAT columns are swapped in the source CSV
    longitude = csv_row["LAT"]
    latitude = csv_row["LON"]

    raster_path = raster_dir / f"brazil_coverage_{year}.tif"
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    raster_patch_size = target_size // UPSCALE_FACTOR
    half_patch = raster_patch_size // 2

    with rasterio.open(raster_path) as src:
        r, c = src.index(longitude, latitude)
        rio_window = rasterio.windows.Window(
            col_off=c - half_patch,
            row_off=r - half_patch,
            width=raster_patch_size,
            height=raster_patch_size,
        )
        patch = src.read(1, window=rio_window, boundless=True, fill_value=NODATA_VALUE)

    # Remap omitted classes to no-data
    if omit_classes:
        for cls in omit_classes:
            patch[patch == cls] = NODATA_VALUE

    if np.all(patch == NODATA_VALUE):
        print(
            f"Omitted window {window_name}: all pixels are no-data after class omission"
        )
        return "omit_nodata"

    # Upscale via nearest-neighbor (repeat each pixel 3x3)
    upscaled = np.repeat(
        np.repeat(patch, UPSCALE_FACTOR, axis=0), UPSCALE_FACTOR, axis=1
    )

    # Trim to exact target_size in case of rounding (e.g. target_size not divisible by 3)
    upscaled = upscaled[:target_size, :target_size]

    # Project point to UTM and build rslearn window
    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, target_size)

    start_time = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(year, 12, 31, tzinfo=timezone.utc)

    dataset = Dataset(ds_path)
    window = Window(
        storage=dataset.storage,
        group=GROUP_NAME,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(start_time, end_time),
        options={"split": split},
    )
    window.save()

    raster = upscaled[np.newaxis, :, :].astype(np.uint8)

    raster_out_dir = window.get_raster_dir(LABEL_LAYER, [BAND_NAME])
    GeotiffRasterFormat().encode_raster(
        raster_out_dir,
        window.projection,
        window.bounds,
        RasterArray(chw_array=raster),
    )
    window.mark_layer_completed(LABEL_LAYER)
    return "created"


def create_windows_from_csv(
    csv_path: UPath,
    ds_name: str,
    target_size: int,
    omit_classes: set[int],
    raster_dir: Path,
    workers: int,
) -> None:
    """Create dense-label windows for all rows in the subsample CSV."""
    rslearn_root = os.environ.get("RSLEARN_EAI_ROOT")
    if not rslearn_root:
        raise RuntimeError("RSLEARN_EAI_ROOT environment variable is not set")
    ds_path = UPath(rslearn_root) / ds_name

    raster_patch_size = target_size // UPSCALE_FACTOR

    df = pd.read_csv(csv_path)
    csv_rows = [row for _, row in df.iterrows()]

    print(f"Loaded {len(csv_rows)} rows from {csv_path}")
    print(f"Dataset path: {ds_path}")
    print(f"Target window size: {target_size}x{target_size} at 10m")
    print(
        f"Raster patch size:  {raster_patch_size}x{raster_patch_size} at ~30m (upscale {UPSCALE_FACTOR}x)"
    )
    if omit_classes:
        print(f"Omitting classes: {sorted(omit_classes)}")

    jobs = [
        dict(
            csv_row=row,
            ds_path=ds_path,
            target_size=target_size,
            omit_classes=omit_classes,
            raster_dir=raster_dir,
        )
        for row in csv_rows
    ]

    counts = {"created": 0, "omit_class": 0, "omit_nodata": 0}
    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, create_window, jobs)
    for status in tqdm.tqdm(outputs, total=len(jobs)):
        counts[status] += 1
    p.close()

    total_omitted = counts["omit_class"] + counts["omit_nodata"]
    print("\n" + "=" * 60)
    print("WINDOW CREATION SUMMARY (dense raster)")
    print("=" * 60)
    print(f"  Created:              {counts['created']}")
    print(f"  Omitted (class):      {counts['omit_class']}")
    print(f"  Omitted (all nodata): {counts['omit_nodata']}")
    print(f"  Total omitted:        {total_omitted}")
    print(f"  Total processed:      {len(jobs)}")
    print("=" * 60)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(
        description="Create rslearn windows with dense raster labels for MapBiomas.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV),
        help="Path to the dense raster subsample CSV.",
    )
    parser.add_argument(
        "--ds-name",
        type=str,
        default=DEFAULT_DS_NAME,
        help="Dataset name, appended to RSLEARN_EAI_ROOT (default: %(default)s).",
    )
    parser.add_argument(
        "--raster-dir",
        type=str,
        default=str(DEFAULT_RASTER_DIR),
        help="Directory containing brazil_coverage_{year}.tif rasters.",
    )
    parser.add_argument(
        "--omit-classes",
        type=int,
        nargs="*",
        default=[],
        help="Class IDs to remap to no-data (windows left all-nodata are skipped).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=48,
        help="Window size in 10m pixels (default: 48). "
        "A patch of (target_size / 3) pixels is read from the ~30m raster.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of multiprocessing workers (default: 32).",
    )
    args = parser.parse_args()

    create_windows_from_csv(
        csv_path=UPath(args.csv_path),
        ds_name=args.ds_name,
        target_size=args.target_size,
        omit_classes=set(args.omit_classes),
        raster_dir=Path(args.raster_dir),
        workers=args.workers,
    )
