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


def process_csv(
    csv_path: UPath, num_pixels: int = 10, postprocess_categories: bool = False
) -> pd.DataFrame:
    """Create windows for crop type mapping.

    Args:
        csv_path: path to the csv file
        num_pixels: number of points to sample from each polygon
        postprocess_categories: whether to postprocess categories
    """
    df = pd.read_csv(csv_path)
    df["latitude"], df["longitude"] = df["y"], df["x"]

    df = df[
        [
            "unique_id",
            "latitude",
            "longitude",
            "LR_plantin",
            "LR_Harvest",
            "LR_harvetd",
            "Category",
        ]
    ]
    print(df.groupby("Category").size())
    print(df["unique_id"].nunique())  # 812 in total

    # Sample per polygon
    df_sampled = (
        df.groupby("unique_id")
        .apply(
            lambda x: x.sample(num_pixels, random_state=42)
            if len(x) > num_pixels
            else x
        )
        .reset_index(drop=True)
    )

    if postprocess_categories:
        # Post-process on category.
        df_sampled.loc[df_sampled["Category"] == "Exoticetrees/forest", "Category"] = (
            "Trees"
        )
        df_sampled.loc[df_sampled["Category"] == "Nativetrees/forest", "Category"] = (
            "Trees"
        )
        # df_sampled = df_sampled[~df_sampled["Category"].isin(["Vegetables", "Legumes"])]

    print(df_sampled.shape)
    print(df_sampled.groupby("Category").size())
    print(df_sampled["unique_id"].nunique())

    output_path = csv_path.with_name(csv_path.stem + "_sampled.csv")
    df_sampled.reset_index(drop=True).to_csv(output_path, index=True)

    return df_sampled


def create_window(
    csv_row: pd.Series,
    ds_path: UPath,
    group_name: str,
    split_by_polygon: bool,
    window_size: int,
) -> None:
    """Create windows for crop type mapping.

    Args:
        csv_row: a row of the dataframe
        ds_path: path to the dataset
        group_name: name of the group
        split_by_polygon: whether to split by polygon
        window_size: window size
    """
    # Get sample metadata
    polygon_id = csv_row["unique_id"]
    latitude, longitude = csv_row["latitude"], csv_row["longitude"]
    planted_date, harvested_or_not, harvested_date = (
        csv_row["LR_plantin"],
        csv_row["LR_Harvest"],
        csv_row["LR_harvetd"],
    )
    category = csv_row["Category"]

    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)
    bounds = calculate_bounds(dst_geometry, window_size)

    # Check if train or val.
    if split_by_polygon:
        group = f"{group_name}_polygon_split_window_{window_size}"
    else:
        group = f"{group_name}_random_split_window_{window_size}"
    window_name = f"{polygon_id}_{latitude}_{longitude}"

    # If split by polygon id, no samples from the same polygon will be in the same split.
    if split_by_polygon:
        is_val = hashlib.sha256(str(polygon_id).encode()).hexdigest()[0] in ["0", "1"]
    else:
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
            "planted_date": planted_date,
            "harvested_or_not": harvested_or_not,
            "harvested_date": harvested_date,
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
    split_by_polygon: bool,
    window_size: int,
    num_pixels: int,
    postprocess_categories: bool,
) -> None:
    """Create windows from csv.

    Args:
        csv_path: path to the csv file
        ds_path: path to the dataset
        group_name: name of the group
        split_by_polygon: whether to split by polygon
        window_size: window size
        num_pixels: number of pixels to sample from each polygon
        postprocess_categories: whether to postprocess categories
    """
    df_sampled = process_csv(csv_path, num_pixels, postprocess_categories)
    csv_rows = []
    for _, row in df_sampled.iterrows():
        csv_rows.append(row)

    jobs = [
        dict(
            csv_row=row,
            ds_path=ds_path,
            group_name=group_name,
            split_by_polygon=split_by_polygon,
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
        required=False,
        help="Name of the group",
        default="groundtruth",
    )
    parser.add_argument(
        "--postprocess_categories",
        type=bool,
        required=False,
        help="Postprocess categories",
        default=True,
    )
    parser.add_argument(
        "--window_size",
        type=int,
        required=False,
        help="Window size",
        default=1,
    )
    parser.add_argument(
        "--num_pixels",
        type=int,
        required=False,
        help="Number of pixels to sample from each polygon",
        default=10,
    )
    parser.add_argument(
        "--split_by_polygon",
        type=bool,
        required=False,
        help="Split by polygon",
        default=True,
    )
    args = parser.parse_args()
    create_windows_from_csv(
        UPath(args.csv_path),
        UPath(args.ds_path),
        args.group_name,
        split_by_polygon=args.split_by_polygon,
        window_size=args.window_size,
        num_pixels=args.num_pixels,
        postprocess_categories=args.postprocess_categories,
    )
