"""Create windows for LFMC estimation."""

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

from rslp.lfmc.constants import CUTOFF_VALUE
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
    lfmc_value = csv_row["lfmc_value"]
    if lfmc_value > CUTOFF_VALUE:
        lfmc_value = CUTOFF_VALUE

    # Get sample metadata
    sample_id = csv_row.name
    site_name, state_region, country = (
        csv_row["site_name"],
        csv_row["state_region"],
        csv_row["country"],
    )
    latitude, longitude = csv_row["latitude"], csv_row["longitude"]

    sampling_date = datetime.strptime(csv_row["sampling_date"], "%Y-%m-%d").replace(
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
    group = "globe_lfmc"
    window_name = f"{sample_id}_{latitude}_{longitude}"

    is_val = hashlib.sha256(str(window_name).encode()).hexdigest()[0] in ["0", "1"]

    if is_val:
        split = "val"
    else:
        split = "train"

    is_site_name_val = hashlib.sha256(site_name.encode()).hexdigest()[0] in ["0", "1"]

    if is_site_name_val:
        site_name_split = "val"
    else:
        site_name_split = "train"

    window = Window(
        storage=dataset.storage,
        group=group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=(start_time, end_time),
        options={
            "split": split,
            "site_name_split": site_name_split,
            "site_name": site_name,
            "state_region": state_region,
            "country": country,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    window.save()

    # Add the label.
    feature = Feature(
        window.get_geometry(),
        {
            "lfmc_value": lfmc_value,
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
    multiprocessing.set_start_method("spawn")
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
