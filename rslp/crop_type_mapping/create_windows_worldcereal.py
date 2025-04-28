"""Create windows for crop type mapping."""

import argparse
import hashlib
import multiprocessing
from datetime import datetime, timedelta, timezone

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


def create_window(
    csv_row: pd.Series, ds_path: UPath, window_size: int
) -> None:
    """Create windows for crop type mapping.

    Args:
        csv_row: a row of the dataframe
        ds_path: path to the dataset
        window_size: window size
    """
    # Get sample metadata
    sample_id = csv_row["sample_id"]
    latitude, longitude = csv_row["latitude"], csv_row["longitude"]
    
    valid_time = datetime.strptime(csv_row["valid_time"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_time, end_time = valid_time - timedelta(days=15), valid_time + timedelta(days=15)
    
    level_123 = str(csv_row["level_123"])
    ewoc_code = str(csv_row["ewoc_code"])
    h3_l3_cell = str(csv_row["h3_l3_cell"])
    quality_score_lc = csv_row["quality_score_lc"]
    quality_score_ct = csv_row["quality_score_ct"]
    
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
    group = "h3_sample100_66K"
    window_name = f"{h3_l3_cell}_{latitude}_{longitude}"

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
        time_range=(start_time, end_time),
        options={
            "split": split,
            "sample_id": sample_id,
            "ewoc_code": ewoc_code,
            "level_123": level_123,
            "h3_l3_cell": h3_l3_cell,
            "quality_score_lc": quality_score_lc,
            "quality_score_ct": quality_score_ct,
            "weight": 1,
        },
    )
    window.save()

    # Add the label.
    feature = Feature(
        window.get_geometry(),
        {
            "category": level_123,
        },
    )
    layer_dir = window.get_layer_dir(LABEL_LAYER)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(LABEL_LAYER)


def create_windows_from_csv(
    csv_path: UPath,
    ds_path: UPath,
    window_size: int,
) -> None:
    """Create windows from csv.

    Args:
        csv_path: path to the csv file
        ds_path: path to the dataset
        window_size: window size
    """
    df_sampled = pd.read_csv(csv_path)
    # print(df_sampled["valid_time"].min(), df_sampled["valid_time"].max())  # 2017-01-06 2023-10-20
    # remove rows with latitude > 60, due to S2 images are not available
    # df_sampled = df_sampled[df_sampled["latitude"] < 60]
    csv_rows = []
    for _, row in df_sampled.iterrows():
        csv_rows.append(row)

    jobs = [
        dict(
            csv_row=row,
            ds_path=ds_path,
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
        required=False,
        help="Path to the csv file",
        default="/weka/dfive-default/yawenz/rslearn_projects/rslp/crop_type_mapping/csv/worldcereal_points_filtered_level_123.csv",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=False,
        help="Path to the dataset",
        default="/weka/dfive-default/rslearn-eai/datasets/crop_type_mapping/20250422_worldcereal",
    )
    parser.add_argument(
        "--window_size", type=int, required=False, help="Window size", default=1
    )
    args = parser.parse_args()
    create_windows_from_csv(
        UPath(args.csv_path),
        UPath(args.ds_path),
        window_size=args.window_size,
    )
