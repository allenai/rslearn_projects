"""Create windows for the Tolbi project."""

import argparse
import hashlib
import multiprocessing
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import shapely
import tqdm
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
    "invalid",  # 0 as invalid
    "cacao",
    "palmoil",
    "rubber",
    "tree",
    "shrub",
    "others",
]


def create_window(
    csv_row: pd.Series,
    ds_path: UPath,
    group_name: str,
    window_size: int,
) -> None:
    """Create windows for the Tolbi project.

    Args:
        csv_row: a row of the dataframe
        ds_path: path to the dataset
        group_name: name of the group
        window_size: window size
    """
    # Get sample metadata
    latitude, longitude = csv_row["latitude"], csv_row["longitude"]
    category = csv_row["class_name"]

    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

    # Spatial split (1/4 for val, 3/4 for train)
    tile = (bounds[0] // 256, bounds[1] // 256)
    grid_cell_id = f"{dst_projection.crs}_{tile[0]}_{tile[1]}"
    first_hex_char_in_hash = hashlib.sha256(grid_cell_id.encode()).hexdigest()[0]
    if first_hex_char_in_hash in ["0", "1", "2", "3"]:
        split = "val"
    else:
        split = "train"

    window_name = f"{csv_row.index}_{latitude}_{longitude}"
    start_time = datetime(int(csv_row["reference_year"]), 1, 1, tzinfo=timezone.utc)
    end_time = datetime(int(csv_row["reference_year"]), 12, 31, tzinfo=timezone.utc)

    window = Window(
        path=Window.get_window_root(ds_path, group_name, window_name),
        group=group_name,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(start_time, end_time),
        options={
            "split": split,
            "category": category,
        },
    )
    window.save()

    # Add the label raster.
    class_id = CLASS_NAMES.index(category)
    raster = np.zeros(
        (1, window.bounds[3] - window.bounds[1], window.bounds[2] - window.bounds[0]),
        dtype=np.uint8,
    )
    raster[:, raster.shape[1] // 2, raster.shape[2] // 2] = class_id
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
    df_sampled = pd.read_csv(csv_path)
    csv_rows = []
    for _, row in df_sampled.iterrows():
        csv_rows.append(row)

    jobs = [
        dict(
            csv_row=row,
            ds_path=ds_path,
            group_name=group_name,
            window_size=window_size,
        )
        for row in csv_rows
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
        required=False,
        help="Window size",
        default=1,
    )
    args = parser.parse_args()
    create_windows_from_csv(
        UPath(args.csv_path),
        UPath(args.ds_path),
        group_name=args.group_name,
        window_size=args.window_size,
    )
