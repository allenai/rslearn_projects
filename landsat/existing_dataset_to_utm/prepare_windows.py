"""This script prepares UTM rslearn windows corresponding to the existing WebMercator landsat windows.
But it also produces:
1. File containing four corners of rectangle of original window in the new coordinate system.
   The image should be blacked out outside of this quadrilateral.
2. File containing vessel positions in the new window.
"""

import json
import math
import multiprocessing
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import shapely
import skimage.draw
import tqdm
from PIL import Image
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs

in_dir = "/data/favyenb/landsat8-data/multisat_vessel_labels_fixed/"
image_db_fname = "/data/favyenb/landsat8-data/siv.sqlite3"
out_dir = "/data/favyenb/rslearn_landsat/windows/"
group = "labels_utm"

pixels_per_tile = 1024
src_crs = CRS.from_epsg(3857)
web_mercator_m = 2 * math.pi * 6378137
total_pixels = (2**13) * 512
src_pixel_size = web_mercator_m / total_pixels
src_projection = Projection(src_crs, src_pixel_size, -src_pixel_size)

dst_pixel_size = 15

image_conn = sqlite3.connect(image_db_fname)
image_conn.isolation_level = None
image_db = image_conn.cursor()

# Get mapping from window ID to image timestamp.
image_db.execute(
    "SELECT w.id, im.time FROM windows AS w, images AS im WHERE w.image_id = im.id"
)
window_id_to_time = {}
for w_id, im_time_str in image_db.fetchall():
    im_time = datetime.strptime(im_time_str.split("+")[0], "%Y-%m-%d %H:%M:%S")
    im_time = im_time.replace(tzinfo=timezone.utc)
    window_id_to_time[w_id] = im_time

example_ids = os.listdir(in_dir)


def handle(example_id):
    # Extract polygon in source projection coordinates from the example folder name.
    parts = example_id.split("_")
    col = int(parts[0]) - total_pixels // 2
    row = int(parts[1]) - total_pixels // 2
    image_uuid = parts[2]
    window_id = int(parts[3])
    src_polygon = shapely.Polygon(
        [
            [col, row],
            [col + pixels_per_tile, row],
            [col + pixels_per_tile, row + pixels_per_tile],
            [col, row + pixels_per_tile],
        ]
    )

    # Now identify the appropriate UTM projection for the polygon, and transform it.
    src_geom = STGeometry(src_projection, src_polygon, None)
    wgs84_geom = src_geom.to_projection(WGS84_PROJECTION)
    # We apply abs() on the latitude because Landsat only uses northern UTM zones.
    dst_crs = get_utm_ups_crs(wgs84_geom.shp.centroid.x, abs(wgs84_geom.shp.centroid.y))
    dst_projection = Projection(dst_crs, dst_pixel_size, -dst_pixel_size)
    dst_geom = src_geom.to_projection(dst_projection)
    dst_polygon = dst_geom.shp

    # (1) Write the window itself.
    bounds = [
        int(dst_polygon.bounds[0]),
        int(dst_polygon.bounds[1]),
        int(dst_polygon.bounds[2]),
        int(dst_polygon.bounds[3]),
    ]
    # Make the bounds multiple of two since some bands are 1/2 the resolution.
    if (bounds[2] - bounds[0]) % 2 == 1:
        bounds[2] += 1
    if (bounds[3] - bounds[1]) % 2 == 1:
        bounds[3] += 1
    ts = window_id_to_time[window_id]
    time_range = (
        ts - timedelta(minutes=1),
        ts + timedelta(minutes=1),
    )
    assert time_range[0].tzinfo is not None and time_range[1].tzinfo is not None
    window_root = os.path.join(out_dir, group, example_id)
    window = Window(
        window_root=window_root,
        group=group,
        name=example_id,
        projection=dst_projection,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()

    # (2) Write the vessel positions.
    vessel_points = []
    with open(os.path.join(in_dir, example_id, "gt.json")) as f:
        for x1, y1, x2, y2, category in json.load(f):
            # Existing label.
            cx = col + (x1 + x2) / 2
            cy = row + (y1 + y2) / 2
            # Warp to new projection.
            src_vessel = STGeometry(src_projection, shapely.Point(cx, cy), None)
            dst_vessel = src_vessel.to_projection(dst_projection)
            cx = int(dst_vessel.shp.x) - bounds[0]
            cy = int(dst_vessel.shp.y) - bounds[1]
            vessel_points.append(
                [
                    cx - 10,
                    cy - 10,
                    cx + 10,
                    cy + 10,
                    category,
                ]
            )
    with open(os.path.join(window_root, "gt.json"), "w") as f:
        json.dump(vessel_points, f)

    # (3) Write mask corresponding to old window projected onto new window.
    mask = np.zeros((bounds[3] - bounds[1], bounds[2] - bounds[0]), dtype=np.uint8)
    assert len(dst_polygon.exterior.coords) == 5
    assert len(dst_polygon.interiors) == 0
    polygon_rows = [coord[1] - bounds[1] for coord in dst_polygon.exterior.coords]
    polygon_cols = [coord[0] - bounds[0] for coord in dst_polygon.exterior.coords]
    rr, cc = skimage.draw.polygon(polygon_rows, polygon_cols, shape=mask.shape)
    mask[rr, cc] = 255
    Image.fromarray(mask).save(os.path.join(window_root, "mask.png"))


p = multiprocessing.Pool(64)
outputs = p.imap_unordered(handle, example_ids)
for _ in tqdm.tqdm(outputs, total=len(example_ids)):
    pass
p.close()
