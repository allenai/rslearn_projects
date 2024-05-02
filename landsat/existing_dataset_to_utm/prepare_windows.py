"""
This script prepares UTM rslearn windows corresponding to the existing WebMercator landsat windows.
But it also produces:
1. File containing four corners of rectangle of original window in the new coordinate system.
   The image should be blacked out outside of this quadrilateral.
2. File containing vessel positions in the new window.
"""
from datetime import datetime
import math
import sqlite3
import os

from rasterio.crs import CRS
import shapely
import tqdm

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_projection

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
image_db.execute("SELECT w.id, im.time FROM windows AS w, images AS im WHERE w.image_id = im.id")
window_id_to_time = {}
for w_id, im_time_str in image_db.fetchall():
    im_time = datetime.strptime(im_time_str, "%Y-%m-%d %H:%M:%S.%f")
    window_id_to_time[w_id] = im_time

example_ids = os.listdir(in_dir)

for example_id in tqdm.tqdm(example_ids):
    # Extract polygon in source projection coordinates from the example folder name.
    parts = example_id.split("_")
    col = int(parts[0]) - total_pixels // 2
    row = int(parts[1]) - total_pixels // 2
    image_uuid = parts[2]
    src_polygon = shapely.Polygon([
        [col, row],
        [col + pixels_per_tile, row],
        [col + pixels_per_tile, row + pixels_per_tile],
        [col, row + pixels_per_tile],
    ])

    # Now identify the appropriate UTM projection for the polygon, and transform it.
    src_geom = STGeometry(src_projection, src_polygon, None)
    wgs84_geom = src_geom.to_projection(WGS84_PROJECTION)
    dst_crs = get_utm_ups_projection(wgs84_geom.shp.centroid.x, wgs84_geom.shp.centroid.y)
    dst_projection = Projection(dst_crs, dst_pixel_size, -dst_pixel_size)
    dst_geom = src_geom.to_projection(dst_projection)
    dst_polygon = dst_geom.shp

    assert len(dst_polygon.exterior.coords) == 4
    assert len(dst_polygon.interiors) == 0

    # (1) Write the window itself.
    bounds = [
        int(dst_polygon.bounds[0]),
        int(dst_polygon.bounds[1]),
        int(dst_polygon.bounds[2]),
        int(dst_polygon.bounds[3]),
    ]
    window = Window(
        window_root=os.path.join(out_dir, group, example_id),
        group=group,
        name=example_id,
        projection=dst_projection,
        bounds=bounds,
        time_range=,
    )
