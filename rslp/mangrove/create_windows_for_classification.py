"""Create windows for crop type mapping."""

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

# We want to get the whole year of 2020
START_TIME = datetime(2020, 6, 15, tzinfo=timezone.utc)
END_TIME = datetime(2020, 7, 15, tzinfo=timezone.utc)


def create_window(
    csv_row: pd.Series, ds_path: UPath, window_size: int, group_name: str
) -> None:
    """Create windows for crop type mapping.

    Args:
        csv_row: a row of the dataframe
        ds_path: path to the dataset
        window_size: window size
        group_name: group name
    """
    # Get sample metadata
    sample_id = int(csv_row["fid"])
    latitude, longitude = csv_row["latitude"], csv_row["longitude"]
    label = str(csv_row["ref_cls"])

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

    # Check if train or val.
    group = group_name
    window_name = f"{sample_id}_{latitude}_{longitude}"

    is_val = hashlib.sha256(str(window_name).encode()).hexdigest()[0] in ["0", "1"]

    if is_val:
        split = "val"
    else:
        split = "train"

    window = Window(
        path=Window.get_window_root(ds_path, group, window_name),
        group=group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(START_TIME, END_TIME),
        options={
            "split": split,
            "sample_id": sample_id,
            "label": label,
            "weight": 1,
        },
    )
    window.save()

    # Add the label.
    feature = Feature(
        window.get_geometry(),
        {
            "category": label,
        },
    )
    layer_dir = window.get_layer_dir(LABEL_LAYER)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(LABEL_LAYER)


def create_windows_from_csv(
    csv_path: UPath,
    ds_path: UPath,
    window_size: int,
    is_reference: bool,
    group_name: str,
) -> None:
    """Create windows from csv.

    Args:
        csv_path: path to the csv file
        ds_path: path to the dataset
        window_size: window size
        is_reference: whether this is reference points
        group_name: group name
    """
    df = pd.read_csv(csv_path)
    df.rename(columns={"x": "longitude", "y": "latitude"}, inplace=True)
    # Extract steps for reference points
    if is_reference:
        df.index.name = "fid"
        df.reset_index(inplace=True)
        # There's no Water in the reference points, Water and Other are combined.
        df["ref_cls"] = df["ref_col"]
        df_sampled = df
    else:
        df_sampled = df.sample(100000, random_state=42)
        cls_lookup = {
            1: "Mangrove",
            2: "Water",
            3: "Other",
        }
        df_sampled["ref_cls"] = df_sampled["ref_cls"].apply(
            lambda x: cls_lookup[int(x)]
        )
    #     ref_cls
    # 1    45850
    # 2    24177
    # 3    29973

    csv_rows = []
    for _, row in df_sampled.iterrows():
        csv_rows.append(row)

    jobs = [
        dict(
            csv_row=row,
            ds_path=ds_path,
            window_size=window_size,
            group_name=group_name,
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
        "--is_reference",
        action="store_true",
        help="Whether this is reference points",
    )
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
        help="Group name",
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
        window_size=args.window_size,
        is_reference=args.is_reference,
        group_name=args.group_name,
    )
