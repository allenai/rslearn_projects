"""Create windows for cropland classification."""

import argparse
import hashlib
import multiprocessing
from datetime import datetime, timedelta, timezone

import pandas as pd
import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.utils.windows import calculate_bounds

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label"


def create_window(csv_row: pd.Series, dataset: Dataset, window_size: int) -> None:
    """Create windows for crop type mapping.

    Args:
        csv_row: a row of the dataframe
        dataset: the dataset to create the window in
        window_size: window size
    """
    # Get sample metadata
    sample_id = csv_row["sample_id"]
    latitude, longitude = csv_row["latitude"], csv_row["longitude"]

    valid_time = datetime.strptime(csv_row["valid_time"], "%Y-%m-%d").replace(
        tzinfo=timezone.utc
    )
    start_time, end_time = (
        valid_time - timedelta(days=15),
        valid_time + timedelta(days=15),
    )

    level_1 = str(csv_row["level_1"])
    cropland_classes = ["10", "11", "12", "14", "15"]

    # Treat as a binary classification task
    category = "Cropland" if level_1 in cropland_classes else "Non-Cropland"

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
    bounds = calculate_bounds(dst_geometry, window_size)
    # Check if train or val.
    group = "h3_sample100_66K"
    # Adding the valid_time to the window name to avoid duplicate windows
    window_name = f"{h3_l3_cell}_{latitude}_{longitude}_{csv_row['valid_time']}"

    is_val = hashlib.sha256(str(window_name).encode()).hexdigest()[0] in ["0", "1"]

    if is_val:
        split = "val"
    else:
        split = "train"

    window = Window(
        storage=dataset.storage,
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
    window_size: int,
) -> None:
    """Create windows from csv.

    Args:
        csv_path: path to the csv file
        ds_path: path to the dataset
        window_size: window size
    """
    df_sampled = pd.read_csv(csv_path)
    df_sampled = df_sampled.drop_duplicates(
        subset=["h3_l3_cell", "latitude", "longitude", "valid_time"]
    )
    csv_rows = []
    for _, row in df_sampled.iterrows():
        csv_rows.append(row)

    dataset = Dataset(ds_path)
    jobs = [
        dict(
            csv_row=row,
            dataset=dataset,
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
    )
