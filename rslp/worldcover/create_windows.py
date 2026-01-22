"""Create windows for the Tolbi project."""

import argparse
import hashlib
import multiprocessing
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shapely
import tqdm
from rslearn.config.dataset import StorageConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from rslp.utils.windows import calculate_bounds

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label_raster"
BAND_NAME = "label"
CLASS_NAMES = [
    "invalid",  # "Not sure" belongs to invalid
    "bare",  # 1
    "burnt",  # 2
    "crops",  # 3
    "fallow/shifting cultivation",  # 4
    "grassland",  # 5
    "Lichen and moss",  # 6
    "shrub",  # 7
    "snow and ice",  # 8
    "tree",  # 9
    "urban/built-up",  # 10
    "water",  # 11
    "wetland (herbaceous)",  # 12
]
# index from 0 to 12 (12 classes)


def create_window(
    group: pd.DataFrame,
    ds_path: UPath,
    group_name: str,
    window_size: int,
) -> None:
    """Create windows for the Worldcover project.

    Args:
        group: a dataframe with 100 points (10x10 grid) with sampleid, latitude, longitude, class_name, reference_year
        ds_path: path to the dataset
        group_name: name of the group
        window_size: window size in pixels
    """
    # Per sampleid, there should be exactly 10x10 samples
    if len(group) != 100:
        raise ValueError(f"Group {group_name} has {len(group)} samples, expected 100")

    center_lat, center_lon = group["latitude"].mean(), group["longitude"].mean()

    # Get metadata from the first row (assuming all rows in group have same year, etc.)
    first_row = group.iloc[0]
    year = int(first_row["reference_year"]) + 1
    sample_id = first_row.get("sampleid", group_name)

    # The center of the 10x10 grid will be used to create the window
    src_point = shapely.Point(center_lon, center_lat)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(center_lon, center_lat)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

    # Spatial split (1/4 for val, 3/4 for train)
    tile = (bounds[0] // 1024, bounds[1] // 1024)
    grid_cell_id = f"{dst_projection.crs}_{tile[0]}_{tile[1]}"
    first_hex_char_in_hash = hashlib.sha256(grid_cell_id.encode()).hexdigest()[0]
    if first_hex_char_in_hash in ["0", "1"]:
        split = "val"
    elif first_hex_char_in_hash in ["2", "3"]:
        split = "test"
    else:
        split = "train"

    window_name = f"{sample_id}_{round(center_lat, 6)}_{round(center_lon, 6)}"
    start_time = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(year, 12, 31, tzinfo=timezone.utc)

    window = Window(
        storage=StorageConfig()
        .instantiate_window_storage_factory()
        .get_storage(ds_path),
        group=group_name,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(start_time, end_time),
        options={
            "split": split,
        },
    )
    window.save()

    # Add the label raster for all 100 points in the 10x10 grid
    # Bounds are in pixels
    raster_height = window.bounds[3] - window.bounds[1]
    raster_width = window.bounds[2] - window.bounds[0]
    raster = np.zeros((1, raster_height, raster_width), dtype=np.uint8)

    # Process each point in the 10x10 grid
    for _, row in group.iterrows():
        point_lat, point_lon = row["latitude"], row["longitude"]
        point_category = row["tag_name"]

        # Get class_id: use 0 for "Not sure", otherwise use CLASS_NAMES.index
        if point_category == "Not sure":
            point_class_id = 0
        else:
            point_class_id = CLASS_NAMES.index(point_category)

        point_src = shapely.Point(point_lon, point_lat)
        point_src_geometry = STGeometry(WGS84_PROJECTION, point_src, None)
        point_dst_geometry = point_src_geometry.to_projection(dst_projection)

        # Calculate pixel coordinates in the raster
        # Geometry coordinates are already in pixels
        pixel_x = int(point_dst_geometry.shp.x - window.bounds[0])
        # Y-axis is flipped in raster (top-left origin)
        pixel_y = int(point_dst_geometry.shp.y - window.bounds[1])

        # Ensure pixel coordinates are within bounds
        if 0 <= pixel_y < raster_height and 0 <= pixel_x < raster_width:
            raster[0, pixel_y, pixel_x] = point_class_id

    raster_dir = window.get_raster_dir(LABEL_LAYER, [BAND_NAME])
    GeotiffRasterFormat().encode_raster(
        raster_dir, window.projection, window.bounds, raster
    )
    window.mark_layer_completed(LABEL_LAYER)


def create_windows_from_csv(
    csv_path: UPath,
    ds_path: UPath,
    group_name: str,
    window_size: int,
) -> None:
    """Create windows from csv.

    Args:
        csv_path: path to the csv file
        ds_path: path to the dataset
        group_name: name of the group
        window_size: window size
    """
    df = pd.read_csv(csv_path)
    df_grouped = df.groupby("sampleid")

    sample_groups = []
    for _, group in df_grouped:
        sample_groups.append(group)

    jobs = [
        dict(
            group=group,
            ds_path=ds_path,
            group_name=group_name,
            window_size=window_size,
        )
        for group in sample_groups
    ]
    p = multiprocessing.Pool(32)
    outputs = star_imap_unordered(p, create_window, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(description="Create windows from csv")
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the csv file",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--group_name",
        type=str,
        required=True,
        help="Name of the group",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        required=True,
        help="Window size",
    )
    args = parser.parse_args()
    create_windows_from_csv(
        UPath(args.csv_path),
        UPath(args.ds_path),
        group_name=args.group_name,
        window_size=args.window_size,
    )

# Try to have window size: 53 x 53, crop size: 32 x 32
