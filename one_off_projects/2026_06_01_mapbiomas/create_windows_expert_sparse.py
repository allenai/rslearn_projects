"""Create rslearn windows with sparse target labels from expert validation points.

Each window is an *extended* window of size (2 * target_size - 3) pixels at 10m
resolution, centered on the expert point. The expert label is placed as a 3x3
block at the window center (representing one ~30m validation pixel). All other
pixels are set to the no-data value (0).

The extended sizing guarantees that any random target_size x target_size crop
from the window will always fully contain the 3x3 labeled block.

Usage::

    python -m rslp.mapbiomas.create_windows_expert_sparse \
        --ds-name mapbiomas_3k \
        --omit-classes 29 5 25
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
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
GROUP_NAME = "mapbiomas_expert_sparse"
LABEL_BLOCK_SIZE = 3

DEFAULT_CSV = (
    MY_ROOT / "rslearn_projects/rslp/mapbiomas/subsampling/sample_expert_points_4k.csv"
)
DEFAULT_DS_NAME = "mapbiomas_3k"


def compute_extended_window_size(target_size: int) -> int:
    """Compute extended window size so every target_size crop contains the 3x3 center."""
    return 2 * target_size - LABEL_BLOCK_SIZE


def create_window(
    csv_row: pd.Series,
    ds_path: UPath,
    window_size: int,
    omit_classes: set[int],
) -> str:
    """Create one sparse-label window from an expert validation point.

    Returns a status string: "created", "omit_class", or "omit_nodata".
    """
    class_id = int(csv_row["CLASS"])
    target_id = int(csv_row["TARGETID"])
    year = int(csv_row["YEAR"])
    split = csv_row["split"]
    window_name = f"{target_id}_{year}"

    if class_id in omit_classes:
        print(f"Omitted window {window_name}: CLASS {class_id} in omit list")
        return "omit_class"

    # LON/LAT columns are swapped in the source CSV
    longitude = csv_row["LAT"]
    latitude = csv_row["LON"]

    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

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

    raster_h = bounds[3] - bounds[1]
    raster_w = bounds[2] - bounds[0]
    raster = np.full((1, raster_h, raster_w), NODATA_VALUE, dtype=np.uint8)

    cy, cx = raster_h // 2, raster_w // 2
    half = LABEL_BLOCK_SIZE // 2
    y_lo = max(cy - half, 0)
    y_hi = min(cy + half + 1, raster_h)
    x_lo = max(cx - half, 0)
    x_hi = min(cx + half + 1, raster_w)
    raster[0, y_lo:y_hi, x_lo:x_hi] = class_id

    raster_dir = window.get_raster_dir(LABEL_LAYER, [BAND_NAME])
    GeotiffRasterFormat().encode_raster(
        raster_dir,
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
    workers: int,
) -> None:
    """Create sparse-label windows for all rows in the expert subsample CSV."""
    rslearn_root = os.environ.get("RSLEARN_EAI_ROOT")
    if not rslearn_root:
        raise RuntimeError("RSLEARN_EAI_ROOT environment variable is not set")
    ds_path = UPath(rslearn_root) / ds_name

    window_size = compute_extended_window_size(target_size)

    df = pd.read_csv(csv_path)
    csv_rows = [row for _, row in df.iterrows()]

    print(f"Loaded {len(csv_rows)} rows from {csv_path}")
    print(f"Dataset path: {ds_path}")
    print(f"Target crop size: {target_size}x{target_size}")
    print(
        f"Extended window size: {window_size}x{window_size}  (2*{target_size} - {LABEL_BLOCK_SIZE})"
    )
    if omit_classes:
        print(f"Omitting classes: {sorted(omit_classes)}")

    jobs = [
        dict(
            csv_row=row,
            ds_path=ds_path,
            window_size=window_size,
            omit_classes=omit_classes,
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
    print("WINDOW CREATION SUMMARY (expert sparse)")
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
        description="Create rslearn windows with sparse expert labels for MapBiomas.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV),
        help="Path to the expert subsample CSV.",
    )
    parser.add_argument(
        "--ds-name",
        type=str,
        default=DEFAULT_DS_NAME,
        help="Dataset name, appended to RSLEARN_EAI_ROOT (default: %(default)s).",
    )
    parser.add_argument(
        "--omit-classes",
        type=int,
        nargs="*",
        default=[],
        help="Class IDs to treat as no-data (windows with only omitted classes are skipped).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=48,
        help="Target crop size in 10m pixels (default: 48). "
        "The actual window will be (2*target_size - 3) to ensure any "
        "random crop always includes the 3x3 labeled center.",
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
        workers=args.workers,
    )
