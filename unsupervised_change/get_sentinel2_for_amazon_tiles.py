"""
I want to send the tiles with forest loss from Amazon Conservation project to the unsupervised change team.
But we can no longer download tiles using old method from GCS.
So this script is similar to landsat/random_landsat_windows.py but instead for creating Sentinel-2 windows.
"""

from datetime import datetime, timedelta, timezone
import json
import math
import os
import random
import shapely

from rasterio.crs import CRS

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry

tiles_fname = "/data/favyenb/rslearn_change_amazon_images/tiles.json"
out_dir = "/data/favyenb/rslearn_change_amazon_images/windows/"
GROUP = "default"

webmercator_crs = CRS.from_epsg(3857)
pixels_per_tile = 512
web_mercator_m = 2 * math.pi * 6378137
pixel_size = web_mercator_m / (2**13) / 512
projection = Projection(webmercator_crs, pixel_size, -pixel_size)

with open(tiles_fname) as f:
    tiles = json.load(f)

for tile in tiles:
    bounds = (
        (tile[0] - 4096) * pixels_per_tile,
        (tile[1] - 4096) * pixels_per_tile,
        (tile[0] - 4096 + 1) * pixels_per_tile,
        (tile[1] - 4096 + 1) * pixels_per_tile,
    )

    for year in range(2016, 2024):
        for month in range(1, 13, 2):
            start_time = datetime(year, month, 1, tzinfo=timezone.utc)
            end_time = datetime(year, month + 1, 28, tzinfo=timezone.utc)
            window_name = f"{tile[0]}_{tile[1]}_{year:04d}_{month:02d}"
            window = Window(
                window_root=os.path.join(out_dir, GROUP, window_name),
                group=GROUP,
                name=window_name,
                projection=projection,
                bounds=bounds,
                time_range=(start_time, end_time),
            )
            window.save()
