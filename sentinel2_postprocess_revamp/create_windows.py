"""
This script creates rslearn windows to get vessel crops based on a CSV with columns:
- timestamp
- latitude
- longitude
- ship_type
- length
- width
- cog
- sog

Or with columns:
- event_id
- event_time
- lat
- lon
- vessel_type
- vessel_length
- vessel_width
- ais_course
- ais_speed
Here we just want to create rslearn windows to get the vessel crops (it also has time/lat/lon).
And then also write some of the metadata from the CSV into a file in the window dirs.

We assign some images to validation split based on their MD5 hash.
"""
import csv
from datetime import datetime, timedelta
import hashlib
import json
import multiprocessing
import os
import sys

import shapely
import tqdm

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Feature, Projection, STGeometry, get_utm_ups_crs

from ship_types import ship_types

in_fname = sys.argv[1]
out_dir = sys.argv[2]
group = sys.argv[3]

pixel_size = 10
window_size = 128
vessel_categories = ["cargo", "tanker", "passenger", "service", "pleasure", "fishing", "enforcement", "sar"]

with open(in_fname) as f:
    reader = csv.DictReader(f)
    csv_rows = list(reader)

def process_row(csv_row):
    def get_optional_float(k):
        if csv_row[k]:
            return float(csv_row[k])
        else:
            return None

    if "event_time" in csv_row:
        event_id = csv_row["event_id"]
        ts = datetime.fromisoformat(csv_row["event_time"])
        lat = float(csv_row["lat"])
        lon = float(csv_row["lon"])
        if csv_row["vessel_category"]:
            ship_type = csv_row["vessel_category"]
        else:
            ship_type = "unknown"
        vessel_length = get_optional_float("vessel_length")
        vessel_width = get_optional_float("vessel_width")
        vessel_cog = get_optional_float("ais_course")
        vessel_cog_avg = get_optional_float("course")
        vessel_sog = get_optional_float("ais_speed")
    else:
        ts = datetime.fromisoformat(csv_row["timestamp"])
        lat = float(csv_row["latitude"])
        lon = float(csv_row["longitude"])
        if csv_row["ship_type"]:
            ship_type = ship_types.get(int(csv_row["ship_type"]), "unknown")
        else:
            ship_type = "unknown"

        vessel_length = get_optional_float("length")
        vessel_width = get_optional_float("width")
        vessel_cog = get_optional_float("cog")
        vessel_cog_avg = None
        vessel_sog = get_optional_float("sog")
        event_id = f"{csv_row['timestamp']}_{csv_row['longitude']}_{csv_row['latitude']}"

    src_point = shapely.Point(lon, lat)
    src_geometry = STGeometry(WGS84_PROJECTION, src_point, None)
    dst_crs = get_utm_ups_crs(lon, lat)
    dst_projection = Projection(dst_crs, pixel_size, -pixel_size)
    dst_geometry = src_geometry.to_projection(dst_projection)

    bounds = (
        int(dst_geometry.shp.x) - window_size // 2,
        int(dst_geometry.shp.y) - window_size // 2,
        int(dst_geometry.shp.x) + window_size // 2,
        int(dst_geometry.shp.y) + window_size // 2,
    )
    time_range = (ts - timedelta(hours=1), ts + timedelta(hours=1))

    window_name = event_id
    # Check if train or val.
    cur_group = group
    if hashlib.md5(window_name.encode()).hexdigest()[0] in ["0", "1"]:
        cur_group = f"{group}_val"
    window_root = os.path.join(out_dir, "windows", cur_group, window_name)
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
            "event_id": event_id,
            "length": vessel_length,
            "width": vessel_width,
            "cog": vessel_cog,
            "cog_avg": vessel_cog_avg,
            "sog": vessel_sog,
            "type": ship_type,
        }, f)

    info_dir = os.path.join(window_root, "layers", "info")
    gt_layer_fname = os.path.join(info_dir, "data.geojson")
    os.makedirs(info_dir, exist_ok=True)
    properties = {
        "event_id": event_id,
    }
    if vessel_length and vessel_length >= 5 and vessel_length < 460:
        properties["length"] = vessel_length
    if vessel_width and vessel_width >= 2 and vessel_width < 120:
        properties["width"] = vessel_width
    if vessel_cog and vessel_sog and vessel_sog > 5 and vessel_sog < 50 and vessel_cog >= 0 and vessel_cog < 360:
        properties["cog"] = vessel_cog
    if vessel_sog and vessel_sog > 0 and vessel_sog < 60:
        properties["sog"] = vessel_sog
    if ship_type and ship_type in vessel_categories:
        properties["type"] = ship_type
    feat = Feature(dst_geometry, properties)
    with open(gt_layer_fname, "w") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": [feat.to_geojson()],
        }, f)

p = multiprocessing.Pool(32)
outputs = p.imap_unordered(process_row, csv_rows)
for _ in tqdm.tqdm(outputs, total=len(csv_rows)):
    pass
p.close()
