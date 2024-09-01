import argparse
import csv
import math
import os
import shutil
from datetime import datetime, timedelta

from pydantic import BaseModel
from pyproj import CRS, Transformer
from rslearn.dataset import Window
from rslearn.utils import LocalFileAPI, Projection
from rslearn.utils.raster_format import GeotiffRasterFormat


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


def create_projection(lat, lon, pixel_size=10):
    """Creates a Projection object based on the center latitude and longitude."""
    epsg_code = latlon_to_utm_zone(lat, lon)
    crs = CRS.from_epsg(epsg_code)
    return Projection(crs=crs, x_resolution=pixel_size, y_resolution=pixel_size)


def calculate_bounds(
    lat, lon, pixel_width, pixel_height, pixel_size, projection
) -> tuple[int, int, int, int]:
    """Calculate the bounds of the image in the projected coordinates."""
    transformer = Transformer.from_crs("epsg:4326", projection.crs, always_xy=True)
    center_x, center_y = transformer.transform(lon, lat)

    half_width = (pixel_width / 2) * pixel_size
    half_height = (pixel_height / 2) * pixel_size

    min_x = center_x - half_width
    max_x = center_x + half_width
    min_y = center_y - half_height
    max_y = center_y + half_height

    return (min_x, min_y, max_x, max_y)


def create_rslearn_data(args: ArgsModel):
    with open(args.dataset_csv, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            record = Record(**row)
            projection = create_projection(record.lat, record.lon)

            # Define pixel size and image dimensions (adjust if necessary)
            pixel_size = 10  # 10 meters per pixel for Sentinel-2
            chip_size = 128  # 128x128 pixel image

            # Calculate the geographic bounds of the PNG image
            bounds = calculate_bounds(
                record.lat,
                record.lon,
                chip_size,
                chip_size,
                pixel_size,
                projection,
            )

            timestamp = datetime.fromisoformat(record.time)
            window_root = os.path.join(
                args.out_dir, "windows", record.event_id, record.label
            )
            os.makedirs(window_root, exist_ok=True)

            # Create the Window object
            window = Window(
                file_api=LocalFileAPI(window_root),
                group="images",
                name=record.event_id,
                projection=projection,
                bounds=bounds,
                time_range=(
                    timestamp - timedelta(minutes=1),
                    timestamp + timedelta(minutes=1),
                ),
            )
            window.save()

            """
            populate the chip layer

            this works by copying the chip image to the chips directory.
            """
            crop = None
            file_api = window.file_api.get_folder("layers", "chips", "R_G_B")
            GeotiffRasterFormat().encode_raster(file_api, projection, bounds, crop)
            dst_chip_path = os.path.join(file_api.to_str(), "chip.png")
            shutil.copyfile(record.chip_path, dst_chip_path)
            complete_path = os.path.join(window_root, "layers", "chips", "completed")
            os.system(f"touch {complete_path}")

            """
            populate the label layer

            this works by creating a dummy geotiff raster with the same bounds as the chip image
            and a property to denote the label.
            """
            file_api = window.file_api.get_folder("layers", "label", "label")
            GeotiffRasterFormat().encode_raster(file_api, projection, bounds, crop)
            # TODO write labels.json with the label
            complete_path = os.path.join(window_root, "layers", "label", "completed")
            os.system(f"touch {complete_path}")


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
    shutil.copyfile("config.json", os.path.join(args.out_dir, "config.json"))
    create_rslearn_data(args)
