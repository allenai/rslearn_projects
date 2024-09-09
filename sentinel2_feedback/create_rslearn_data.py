import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import shapely
from pydantic import BaseModel
from pyproj import Transformer
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset.window import Window
from rslearn.utils import get_utm_ups_crs
from rslearn.utils.geometry import Projection, STGeometry
from upath import UPath

point_geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [[32, 32]],
            },
            "properties": {
                "label": None,
            },
        }
    ],
    "properties": None,
}


class ArgsModel(BaseModel):
    dataset_csv: str
    out_dir: str


class Record(BaseModel):
    event_id: str
    label: str
    lat: float
    lon: float
    chip_path: str
    time: str


def latlon_to_utm_zone(lat, lon):
    """Determine the UTM zone for a given latitude and longitude."""
    zone_number = math.floor((lon + 180) / 6) + 1
    if lat >= 0:
        epsg_code = 32600 + zone_number  # Northern Hemisphere
    else:
        epsg_code = 32700 + zone_number  # Southern Hemisphere
    return epsg_code


def calculate_bounds(
    record: Record, projection: Projection
) -> tuple[int, int, int, int]:
    window_size = 128
    point = shapely.Point(record.lon, record.lat)
    stgeometry = STGeometry(WGS84_PROJECTION, point, None)
    geometry = stgeometry.to_projection(projection)

    bounds = [
        int(geometry.shp.x) - window_size // 2,
        int(geometry.shp.y) - window_size // 2,
        int(geometry.shp.x) + window_size // 2,
        int(geometry.shp.y) + window_size // 2,
    ]
    return bounds


def get_label_data(record: Record, window: Window):
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [[32, 32]],
                },
                "properties": {
                    "label": record.label,
                },
            }
        ],
        "properties": window.projection.serialize(),
    }


def create_rslearn_data(args: ArgsModel):
    with open(args.dataset_csv, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            record = Record(**row)
            pixel_size = 10  # 10 meters per pixel for Sentinel-2
            crs = get_utm_ups_crs(record.lat, record.lon)
            projection = Projection(
                crs=crs, x_resolution=pixel_size, y_resolution=-pixel_size
            )

            bounds = calculate_bounds(record, projection)
            timestamp = datetime.fromisoformat(record.time)
            window_root = UPath(f"{args.out_dir}/windows/sentinel2/{record.event_id}")
            os.makedirs(window_root, exist_ok=True)

            # Create the Window object
            window = Window(
                path=window_root,
                group=record.label,
                name=record.event_id,
                projection=projection,
                bounds=bounds,
                time_range=(
                    timestamp - timedelta(minutes=20),
                    timestamp + timedelta(minutes=20),
                ),
            )
            window.save()

            # Populate the sentinel2 layer
            image_layer_dir = os.path.join(window_root, "layers", "sentinel2", "R_G_B")
            os.makedirs(image_layer_dir, exist_ok=True)
            Path(f"{image_layer_dir}/completed").touch()

            # Populate the label layer
            label_layer_dir = os.path.join(window_root, "layers", "label")
            os.makedirs(label_layer_dir, exist_ok=True)
            with open(os.path.join(label_layer_dir, "data.geojson"), "w") as f:
                json.dump(get_label_data(record, window), f)
            Path(f"{label_layer_dir}/completed").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates rslearn data from a CSV of events."
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        required=True,
        help="Dataset CSV file which was the --output_csv from retrieve_dataset.py.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Location of the rslearn dataset.",
    )
    parsed_args = parser.parse_args()
    args = ArgsModel(**vars(parsed_args))  # convert parsed args to pydantic model
    os.makedirs(args.out_dir, exist_ok=True)

    # Copy the model architecture definition to the window directory
    shutil.copyfile("config.json", os.path.join(args.out_dir, "config.json"))
    shutil.copyfile("model_config.yaml", os.path.join(args.out_dir, "config.yaml"))

    create_rslearn_data(args)
