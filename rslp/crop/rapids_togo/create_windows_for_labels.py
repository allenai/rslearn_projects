"""Create windows for crop type mapping.

Data from https://zenodo.org/records/3836629
"""

import argparse
import hashlib
import multiprocessing
from datetime import datetime, timezone

import pandas as pd
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label"

# data was collected in May 2020, so we consider the 6 months before and after may
# we pick the center month; the actual range will be managed by the offset in the config.
START_TIME = datetime(2020, 5, 1, tzinfo=timezone.utc)
END_TIME = datetime(2020, 5, 31, tzinfo=timezone.utc)


def create_window(
    csv_row: pd.Series, ds_path: UPath, group_name: str, window_size: int, is_test: bool
) -> None:
    """Create windows for crop type mapping.

    Args:
        csv_row: a row of the dataframe
        ds_path: path to the dataset
        group_name: name of the group
        window_size: window size
        is_test: whether or not this is a test window
    """
    # Get sample metadata
    polygon_id = csv_row["unique_id"]
    latitude, longitude = csv_row["latitude"], csv_row["longitude"]
    is_crop = csv_row["is_crop"]
    category = is_crop

    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)

    # This is specific for window size = 1.
    if window_size == 1:
        bounds = (
            int(dst_geometry.shp.x),
            int(dst_geometry.shp.y) - window_size,
            int(dst_geometry.shp.x) + window_size,
            int(dst_geometry.shp.y),
        )
    else:
        bounds = (
            int(dst_geometry.shp.x),
            int(dst_geometry.shp.y),
            int(dst_geometry.shp.x) + window_size // 2,
            int(dst_geometry.shp.y) + window_size // 2,
        )

    group = f"{group_name}_window_{window_size}"
    window_name = f"{polygon_id}_{latitude}_{longitude}"
    if not is_test:
        # Check if train or val.
        # If split by polygon id, no samples from the same polygon will be in the same split.
        is_val = hashlib.sha256(str(window_name).encode()).hexdigest()[0] in ["0", "1"]

        if is_val:
            split = "val"
        else:
            split = "train"
    else:
        split = "test"

    window = Window(
        path=Window.get_window_root(ds_path, group, window_name),
        group=group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(START_TIME, END_TIME),
        options={
            "split": split,
            "is_crop": is_crop,
            "category": category,
        },
    )
    window.save()

    # Add the label.
    feature = Feature(
        window.get_geometry(),
        {
            "category": category,
        },
    )
    layer_dir = window.get_layer_dir(LABEL_LAYER)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(LABEL_LAYER)


def create_windows_from_csv(
    csv_paths: UPath,
    ds_path: UPath,
    group_name: str,
    window_size: int,
) -> None:
    """Create windows from csv.

    Args:
        csv_paths: path to the csv files
        ds_path: path to the dataset
        group_name: name of the group
        window_size: window size
    """
    for filename in [
        "crop_merged_v2.csv",
        "noncrop_merged_v2.csv",
        "togo_test_majority.csv",
    ]:
        df_sampled = pd.read_csv(csv_paths / filename)
        csv_rows = []
        for _, row in df_sampled.iterrows():
            csv_rows.append(row)

        jobs = [
            dict(
                csv_row=row,
                ds_path=ds_path,
                group_name=group_name,
                window_size=window_size,
                is_test="test" in filename,
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
        "--csv_paths",
        type=str,
        default="gs://ai2-helios-us-central1/evaluations/crop_type_mapping/togo_2020",
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
        required=False,
        help="Name of the group",
        default="groundtruth",
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
        UPath(args.csv_paths),
        UPath(args.ds_path),
        args.group_name,
        window_size=args.window_size,
    )
