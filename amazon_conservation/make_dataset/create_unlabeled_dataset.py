"""
Create new dataset corresponding to randomly sampled GLAD alerts.
The alerts must be supplied.
Can by configured to avoid sampling alerts that overlap existing labels.
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import numpy as np
import os
from PIL import Image
import random
import rasterio
from rasterio.crs import CRS
import rasterio.features
import shapely.geometry
import shapely.ops
import shapely.wkt
import tqdm

import rsdh.util

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry

# Regions (alert tif, alertdate tif).
# We have these hardcoded so we don't forget the options.
regions = {
    "brazil": [
        '/multisat/datasets/amazon_conservation/2023-12-21-glad-unlabeled/alert_060W_10S_050W_00N.tif',
        '/multisat/datasets/amazon_conservation/2023-12-21-glad-unlabeled/alertDate_060W_10S_050W_00N.tif',
    ],
    "peru": [
        '/multisat/datasets/amazon_conservation/2023-12-21-glad-unlabeled/alert_080W_10S_070W_00N.tif',
        '/multisat/datasets/amazon_conservation/2023-12-21-glad-unlabeled/alertDate_080W_10S_070W_00N.tif',
    ],
}

# Dates in the GDAL GeoTIFF are measured in days since 2019-01-01.
date_base = datetime(2019, 1, 1, tzinfo=timezone.utc)
# Minimum distance from sampled alert to an existing example.
# This is roughly 10 km away, in degrees lat/lon.
existing_distance_threshold = 10000 / 111111
# Confidence threshold for the GLAD alert confidence GeoTIFF.
# 2 corresponds to low confidence.
min_confidence = 2

web_mercator_crs = CRS.from_epsg(3857)
web_mercator_m = 2 * math.pi * 6378137
pixel_size = web_mercator_m / (2**13) / 512
web_mercator_projection = Projection(web_mercator_crs, pixel_size, -pixel_size)

parser = argparse.ArgumentParser()
parser.add_argument("--region", help="Which region (e.g. peru or brazil)")
parser.add_argument("--window_size", help="Size of windows to write in rslearn dataset", type=int, default=128)
parser.add_argument("--count", help="Number of GLAD alerts to save into the dataset (per date range)", type=int, default=200)
parser.add_argument("--min_area", help="Minimum area in pixels to consider a GLAD alert", type=int, default=16)
parser.add_argument("--existing_dirs", help="Optional comma-separated list of existing multisat-formatted training folders, to skip alerts close to existing examples", type=str, default=None)
parser.add_argument("--dates", help="Date ranges to get forest loss alerts in, e.g. 2020-01-01,2020-03-01,2020-03-01,2020-05-05 for Jan/Feb and Mar/Apr 2020")
parser.add_argument("--window_days", help="Number of days each window should span (will be from beginning of specified date range)", type=int, default=30)
parser.add_argument("--out_dir", help="Path to output rslearn dataset")
parser.add_argument("--group", help="Which group to add the windows to", default="default")
args = parser.parse_args()

conf_fname, date_fname = regions[args.region]

def parse_date(s):
    parts = s.split('-')
    return datetime(int(parts[0]), int(parts[1]), int(parts[2]), tzinfo=timezone.utc)

date_ranges = []
arg_dates = args.dates.split(",")
for i in range(0, len(arg_dates), 2):
    date1 = parse_date(arg_dates[i])
    date2 = parse_date(arg_dates[i+1])
    days1 = (date1 - date_base).days
    days2 = (date2 - date_base).days
    date_ranges.append((days1, days2))

# Load MultiPolygon from the existing_dirs.
# It should contain examples like feat_idx_col_row_...
# So we just use the column/row.
existing_points = []
if args.existing_dirs:
    for existing_dir in args.existing_dirs.split(","):
        for example_id in os.listdir(existing_dir):
            parts = example_id.split('_')
            point = (int(parts[2]), int(parts[3]))
            point = rsdh.util.mercator_to_geo(point, zoom=13, pixels=512)
            existing_points.append(point)

def is_existing(point):
    for existing_point in existing_points:
        distance = math.sqrt((point[0] - existing_point[0])**2 + (point[1] - existing_point[1])**2)
        if distance <= existing_distance_threshold:
            return True
    return False

print('read confidences')
conf_raster = rasterio.open(conf_fname)
conf_data = conf_raster.read(1)
print('read dates')
date_raster = rasterio.open(date_fname)
date_data = date_raster.read(1)

for days1, days2 in date_ranges:
    print(days1, days2)

    # Extract centers of polygons corresponding to forest loss detections with at least low confidence.
    mask = (conf_data >= min_confidence) & (date_data >= days1) & (date_data < days2)
    mask = mask.astype(np.uint8)

    print('extract shapes')
    shapes = list(rasterio.features.shapes(mask))
    random.shuffle(shapes)
    num_processed = 0

    for feat_idx, (shp, value) in enumerate(shapes):
        # Discard shapes corresponding to the background.
        if value != 1:
            continue

        shp = shapely.geometry.shape(shp)
        if shp.area < args.min_area:
            continue

        # Get center point (clipped to shape) and note the corresponding date.
        center = shp.centroid
        center_clipped, _ = shapely.ops.nearest_points(shp, center)
        img_point = (int(center_clipped.x), int(center_clipped.y))
        julian_days = int(date_data[img_point[1], img_point[0]])
        cur_date = date_base + timedelta(days=julian_days)

        # Transform the center to Web-Mercator so we can get image around it.
        projection_pos = conf_raster.xy(img_point[1], img_point[0])
        projection_shp = shapely.Point(projection_pos[0], projection_pos[1])
        projection_geom = STGeometry(Projection(conf_raster.crs, 1, 1), projection_shp, None)
        wgs84_geom = projection_geom.to_projection(WGS84_PROJECTION)
        cur_point = (wgs84_geom.shp.centroid.x, wgs84_geom.shp.centroid.y)
        if is_existing(cur_point):
            continue
        web_mercator_shp = wgs84_geom.to_projection(web_mercator_projection).shp

        # WebMercator point to include in the name.
        # The annotation website will use this so we have adjusted this to have the same offset as multisat/other code.
        mercator_point = (
            int(web_mercator_shp.x) + 512 * (2**12),
            int(web_mercator_shp.y) + 512 * (2**12),
        )
        # While the bounds is for rslearn.
        bounds = (
            int(web_mercator_shp.x) - args.window_size // 2,
            int(web_mercator_shp.y) - args.window_size // 2,
            int(web_mercator_shp.x) + args.window_size // 2,
            int(web_mercator_shp.y) + args.window_size // 2,
        )
        time_range = (
            date_base + timedelta(days=days1),
            date_base + timedelta(days=days1 + args.window_days),
        )

        # Create the new rslearn windows.
        window_name = f"feat_{feat_idx}_{mercator_point[0]}_{mercator_point[1]}_{img_point[0]}_{img_point[1]}"
        window = Window(
            window_root=os.path.join(args.out_dir, "windows", args.group, window_name),
            group=args.group,
            name=window_name,
            projection=web_mercator_projection,
            bounds=bounds,
            time_range=time_range,
        )
        window.save()

        # Output some metadata.
        with open(os.path.join(window.window_root, "info.json"), "w") as f:
            json.dump({
                "date1": (date_base + timedelta(days=days1)).isoformat(),
                "date2": (date_base + timedelta(days=days2)).isoformat(),
                "pixel_date": cur_date.isoformat(),
            }, f)

        # Get pixel coordinates of the mask.
        def raster_pixel_to_proj(points):
            for i in range(points.shape[0]):
                points[i, 0:2] = conf_raster.xy(points[i, 1], points[i, 0])
            return points
        projection_shp = shapely.transform(shp, raster_pixel_to_proj)
        projection_polygon = STGeometry(Projection(conf_raster.crs, 1, 1), projection_shp, None)
        web_mercator_shp = projection_polygon.to_projection(web_mercator_projection).shp
        def to_out_pixel(points):
            points[:, 0] -= bounds[0]
            points[:, 1] -= bounds[1]
            return points
        pixel_shp = shapely.transform(web_mercator_shp, to_out_pixel)
        mask_im = rasterio.features.rasterize(
            [(pixel_shp, 255)],
            out_shape=(args.window_size, args.window_size),
        )
        Image.fromarray(mask_im).save(os.path.join(window.window_root, "mask.png"))

        # Add to existing points so we don't add duplicate.
        existing_points.append(cur_point)

        num_processed += 1
        if num_processed >= args.count:
            break
