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

from rslp.utils.windows import calculate_bounds

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label"

# Center month between "2022-09-30" and "2023-09-30"
START_TIME = datetime(2023, 3, 1, tzinfo=timezone.utc)
END_TIME = datetime(2023, 3, 31, tzinfo=timezone.utc)


def process_csv(csv_path: UPath) -> pd.DataFrame:
    """Create windows for crop type mapping.

    Args:
        csv_path: path to the csv file
    """
    df = pd.read_csv(csv_path)
    print(df.groupby("tag_name").size())

    # Random sample 20K points from the csv file, 72K points in total
    df = df.sample(n=20000, random_state=42)

    return df


# feature_id,task_name,tag_name,lon,lat,utm_easting,utm_northing,utm_epsg,utm_zone
# 94a1099a-78f5-4701-9963-1a8948943bd7,Nandi County round 2,Trees,35.01508785066754,0.1054712094358493,724275.0,11665.0,EPSG:32636,36


def create_window(
    csv_row: pd.Series, ds_path: UPath, group_name: str, window_size: int
) -> None:
    """Create windows for crop type mapping.

    Args:
        csv_row: a row of the dataframe
        ds_path: path to the dataset
        group_name: name of the group
        window_size: window size
    """
    # Get sample metadata
    unique_id = csv_row.name
    latitude, longitude = csv_row["lat"], csv_row["lon"]
    category = csv_row["tag_name"]

    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

    group = f"{group_name}_window_{window_size}"
    window_name = f"{unique_id}_{latitude}_{longitude}"

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
    df_sampled = process_csv(csv_path)
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
        help="Window group name",
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
        args.group_name,
        args.window_size,
    )
