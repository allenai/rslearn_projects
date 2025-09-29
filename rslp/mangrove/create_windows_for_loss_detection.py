"""Create windows for mangrove loss detection."""

import argparse
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

from rslp.utils.windows import calculate_bounds

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label"


def create_window(csv_row: pd.Series, ds_path: UPath, window_size: int) -> None:
    """Create windows for crop type mapping.

    Args:
        csv_row: a row of the dataframe
        ds_path: path to the dataset
        window_size: window size
    """
    # Get sample metadata
    sample_id = csv_row.fid
    latitude, longitude = csv_row["latitude"], csv_row["longitude"]
    # Date format: 2021/01/30 23:00:00+00
    first_obs_date, scr5_obs_date = (
        csv_row["first_obs_date"].split(" ")[0],
        csv_row["scr5_obs_date"].split(" ")[0],
    )
    scr5_obs_year = scr5_obs_date.split("/")[0]

    sampling_date = datetime.strptime(scr5_obs_date, "%Y/%m/%d").replace(
        tzinfo=timezone.utc
    )
    start_time, end_time = (
        sampling_date - timedelta(days=15),
        sampling_date + timedelta(days=15),
    )

    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

    # Check if train or val.
    group = "sample_188K_temporal_split"
    window_name = f"{sample_id}_{latitude}_{longitude}_{scr5_obs_year}"

    is_val = scr5_obs_year in ["2024", "2025"]

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
            "latitude": latitude,
            "longitude": longitude,
            "first_obs_date": first_obs_date,
            "scr5_obs_date": scr5_obs_date,
        },
    )
    window.save()

    # Add the label.
    feature = Feature(
        window.get_geometry(),
        {
            "label": csv_row["label"],
        },
    )
    layer_dir = window.get_layer_dir(LABEL_LAYER)
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed(LABEL_LAYER)


def create_windows_from_csv(
    true_positives_csv_path: UPath,
    false_positives_csv_path: UPath,
    ds_path: UPath,
    window_size: int,
) -> None:
    """Create windows from csv.

    Args:
        true_positives_csv_path: path to the csv file for true positives
        false_positives_csv_path: path to the csv file for false positives
        ds_path: path to the dataset
        window_size: window size
    """
    df_true_positives = pd.read_csv(true_positives_csv_path)
    df_false_positives = pd.read_csv(false_positives_csv_path)
    df_true_positives["label"] = "correct"
    df_false_positives["label"] = "incorrect"

    # Sample equal number of true positives and false positives
    df_true_positives = df_true_positives.sample(
        len(df_false_positives), random_state=42
    )
    df_sampled = pd.concat([df_true_positives, df_false_positives])

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
        "--true_positives_csv_path",
        type=str,
        required=True,
        help="Path to the csv file for true positives",
    )
    parser.add_argument(
        "--false_positives_csv_path",
        type=str,
        required=True,
        help="Path to the csv file for false positives",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset",
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
        UPath(args.true_positives_csv_path),
        UPath(args.false_positives_csv_path),
        UPath(args.ds_path),
        window_size=args.window_size,
    )
