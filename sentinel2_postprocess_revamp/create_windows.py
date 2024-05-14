"""
Hunter sent a CSV with heading, length, width, ship type, and other attributes.
Here we just want to create rslearn windows to get the vessel crops (it also has time/lat/lon).
And then also write some of the metadata from the CSV into a file in the window dirs.
"""
import csv
from datetime import datetime, timedelta
import json
import os
import random

import shapely
import tqdm

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_projection

from ship_types import ship_types

in_fname = "/home/favyenb/sentinel2_vessel_labels_with_metadata.csv"
out_dir = "/data/favyenb/rslearn_sentinel2_vessel_postprocess"
pixel_size = 10
group = "vessels"
window_size = 64

with open(in_fname) as f:
    reader = csv.DictReader(f)
    csv_rows = list(reader)

csv_rows = [csv_row for csv_row in csv_rows if csv_row["length"] and csv_row["cog"] and csv_row["width"] and csv_row["ship_type"]]
csv_rows = random.sample(csv_rows, 100)
for idx, csv_row in enumerate(tqdm.tqdm(csv_rows)):
    ts = datetime.fromisoformat(csv_row["timestamp"])
    lat = float(csv_row["latitude"])
    lon = float(csv_row["longitude"])
    if csv_row["ship_type"]:
        ship_type = ship_types.get(int(csv_row["ship_type"]), "unknown")
    else:
        ship_type = "unknown"

    def get_optional_float(k):
        if csv_row[k]:
            return float(csv_row[k])
        else:
            return None
    vessel_length = get_optional_float("length")
    vessel_width = get_optional_float("width")
    vessel_cog = get_optional_float("cog")
    vessel_sog = get_optional_float("sog")

    src_point = shapely.Point(lon, lat)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_projection(lon, lat)
    dst_projection = Projection(dst_crs, pixel_size, -pixel_size)
    dst_geometry = src_geometry.to_projection(dst_projection)

    bounds = (
        int(dst_geometry.shp.x) - window_size // 2,
        int(dst_geometry.shp.y) - window_size // 2,
        int(dst_geometry.shp.x) + window_size // 2,
        int(dst_geometry.shp.y) + window_size // 2,
    )
    time_range = (ts - timedelta(hours=1), ts + timedelta(hours=1))

    window_name = f"vessel_{idx}"
    window_root = os.path.join(out_dir, "windows", group, window_name)
    window = Window(
        window_root=window_root,
        group=group,
        name=window_name,
        projection=dst_projection,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()

    # Save metadata.
    with open(os.path.join(window_root, "info.json"), "w") as f:
        json.dump({
            "length": vessel_length,
            "width": vessel_width,
            "cog": vessel_cog,
            "type": ship_type,
        }, f)
