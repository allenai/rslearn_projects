"""Just pick a bunch of random Landsat windows.
In WebMercator projection since that's how the vessel detection model data is.
Yeah this is for vessel detection, just testing to see how the model does elsewhere.
"""

import math
import os
import random
from datetime import datetime, timedelta, timezone

import shapely
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry

out_dir = "/data/favyenb/rslearn_landsat/windows/"
GROUP = "default"
YEARS = [2021, 2022, 2023]

webmercator_crs = CRS.from_epsg(3857)
pixels_per_tile = 1024
web_mercator_m = 2 * math.pi * 6378137
pixel_size = web_mercator_m / (2**13) / 512

seen = set()
while len(seen) < 1000:
    lat = random.randint(-70, 70)
    lon = random.randint(-180, 180)
    point = shapely.Point(lon, lat)
    geometry = STGeometry(WGS84_PROJECTION, point, None)
    projection = Projection(webmercator_crs, pixel_size, -pixel_size)
    geometry = geometry.to_projection(projection)
    tile_col = int(geometry.shp.x) // pixels_per_tile
    tile_row = int(geometry.shp.y) // pixels_per_tile

    window_name = f"{tile_col}_{tile_row}"
    if window_name in seen:
        continue
    seen.add(window_name)

    bounds = (
        tile_col * pixels_per_tile,
        tile_row * pixels_per_tile,
        (tile_col + 1) * pixels_per_tile,
        (tile_row + 1) * pixels_per_tile,
    )

    start_time = datetime(
        random.choice(YEARS), random.randint(1, 12), 1, tzinfo=timezone.utc
    )
    end_time = start_time + timedelta(days=30)

    window = Window(
        window_root=os.path.join(out_dir, GROUP, window_name),
        group=GROUP,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=(start_time, end_time),
    )
    window.save()
