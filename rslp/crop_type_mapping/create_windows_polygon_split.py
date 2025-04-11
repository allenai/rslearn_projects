"""Create windows for crop type mapping."""

from upath import UPath
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import sys
from datetime import datetime, timedelta, timezone

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.feature import Feature
from rslearn.utils.raster_format import get_raster_projection_and_bounds, GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat


WINDOW_SIZE = 32
WINDOW_RESOLUTION = 10

LABEL_LAYER = "label"

# Use center month between "2022-09-30" and "2023-09-30"
START_TIME = datetime(2023, 3, 1, tzinfo=timezone.utc)
END_TIME = datetime(2023, 3, 31, tzinfo=timezone.utc)


def process_csv(csv_path: UPath, num_pixels: int = 10) -> pd.DataFrame:
    """Create windows for crop type mapping.

    Args:
        csv_path: path to the csv file
        num_points: number of points to sample from each polygon
    """
    df = pd.read_csv(csv_path)
    df["latitude"], df["longitude"] = df["y"], df["x"]

    df = df[["unique_id", "latitude", "longitude", "LR_plantin", "LR_Harvest", "LR_harvetd", "Category"]]
    print(df.groupby("Category").size())
    print(df["unique_id"].nunique())  # 812 (819 in total)
    
    # Sample per polygon
    df_sampled = df.groupby("unique_id").apply(lambda x: x.sample(num_pixels, random_state=42) if len(x) > num_pixels else x).reset_index(drop=True)
    print(df_sampled.shape)
    print(df_sampled.groupby("Category").size())
    print(df_sampled["unique_id"].nunique())

    # # Category stats:
    # # Coffee                  977
    # # Exoticetrees/forest     695
    # # Grassland              1020
    # # Legumes                 440
    # # Maize                  1336
    # # Nativetrees/forest      231
    # # Sugarcane               964
    # # Tea                     979
    # # Vegetables              282

    # # Sample per category, allow replacement to better mimic the RF training samples
    # df_sampled = df.groupby("Category").apply(lambda x: x.sample(num_pixels, random_state=42, replace=True)).reset_index(drop=True)
    # print(df_sampled.shape)
    # print(df_sampled.groupby("Category").size())
    # print(df_sampled)

    return df_sampled


def create_window(csv_row: pd.Series, ds_path: UPath):
    """Create windows for crop type mapping.

    Args:
        csv_row: a row of the dataframe
        ds_path: path to the dataset
    """
    # Get sample metadata
    index = csv_row.name
    polygon_id = csv_row["unique_id"]
    latitude, longitude = csv_row["latitude"], csv_row["longitude"]
    planted_date, harvested_or_not, harvested_date = csv_row["LR_plantin"], csv_row["LR_Harvest"], csv_row["LR_harvetd"]
    category = csv_row["Category"]

    src_point = shapely.Point(longitude, latitude)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(longitude, latitude)
    dst_projection = Projection(dst_crs, WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    dst_geometry = src_geometry.to_projection(dst_projection)

    # bounds = (
    #     int(dst_geometry.shp.x),
    #     int(dst_geometry.shp.y),
    #     int(dst_geometry.shp.x) + WINDOW_SIZE,
    #     int(dst_geometry.shp.y) + WINDOW_SIZE,
    # )
    bounds = (
        int(dst_geometry.shp.x) - WINDOW_SIZE // 2,
        int(dst_geometry.shp.y) - WINDOW_SIZE // 2,
        int(dst_geometry.shp.x) + WINDOW_SIZE // 2,
        int(dst_geometry.shp.y) + WINDOW_SIZE // 2,
    )

    # Check if train or val.
    group = "window_32_split_by_polygon"
    window_name = f"{polygon_id}_{latitude}_{longitude}"
    window_path = ds_path / "windows" / group / window_name
    
    # Split train/val by polygon id.
    is_val = hashlib.md5(str(polygon_id).encode()).hexdigest()[0] in ["0", "1"]
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
            "weight": 1
        }
    )
    window.save()
    
    # # Add the label.
    # feature = Feature(window.get_geometry(), {
    #     "category": category,
    # })
    # layer_dir = window.get_layer_dir(LABEL_LAYER)
    # GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    # window.mark_layer_completed(LABEL_LAYER)


def create_windows_from_csv(csv_path: UPath, ds_path: UPath):
    df_sampled = process_csv(csv_path)
    csv_rows = []
    for _, row in df_sampled.iterrows():
        csv_rows.append(row)

    jobs = [dict(csv_row=row, ds_path=ds_path) for row in csv_rows]
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
        default="gs://ai2-helios-us-central1/evaluations/crop_type_mapping/cgiar/NandiGroundTruthPoints.csv"
    )
    parser.add_argument(
        "--ds_path", 
        type=str, 
        required=False, 
        help="Path to the dataset",
        default="/weka/dfive-default/rslearn-eai/datasets/crop_type_mapping/20250409_kenya_nandi"
    )
    args = parser.parse_args()
    
    create_windows_from_csv(
        UPath(args.csv_path),
        UPath(args.ds_path)
    )



