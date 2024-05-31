"""
This obtains Sentinel-2 images for a few non-US windows that Piper sent.
To test super-resolution model that seemed to be working pretty well.
"""
from datetime import datetime, timezone
import math
import os

from rasterio.crs import CRS
import shapely

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs

zoom13_tiles = [
    [1845, 3650],
    [4103, 2729],
    [6526, 4241],
    [1407, 3270],
    [2047, 3380],
    [2210, 3517],
    [2641, 4476],
    [4150, 2834],
    [4515, 4890],
    [4780, 2728],
    [7276, 3225],
]
zoom15_tiles = [
    [22971, 14217],
    [22971, 14232],
    [22972, 14232],
    [22983, 14233],
    [22984, 14229],
    [22985, 14227],
    [22987, 14226],
    [22995, 14225],
    [22998, 14209],
]
all_tiles = [(tile[0], tile[1], 13) for tile in zoom13_tiles] + [(tile[0], tile[1], 15) for tile in zoom15_tiles]

out_dir = "/data/favyenb/rslearn_superres_non_us/windows/"
group = "default"
web_mercator_crs = CRS.from_epsg(3857)
web_mercator_m = 2 * math.pi * 6378137
utm_pixel_size = 10
pixels_per_tile = 512
time_range = (
    datetime(2023, 3, 1, tzinfo=timezone.utc),
    datetime(2023, 9, 1, tzinfo=timezone.utc),
)

for col, row, zoom in all_tiles:
    # Get source geometry.
    total_pixels = (2**zoom) * pixels_per_tile
    pixel_size = web_mercator_m / total_pixels
    src_projection = Projection(web_mercator_crs, pixel_size, -pixel_size)
    src_shp = shapely.Point(
        (col + 0.5) * pixels_per_tile - total_pixels / 2,
        (row + 0.5) * pixels_per_tile - total_pixels / 2,
    )
    src_geom = STGeometry(src_projection, src_shp, None)

    # Get appropriate UTM geometry and write window.
    wgs84_geom = src_geom.to_projection(WGS84_PROJECTION)
    print(wgs84_geom)
    utm_crs = get_utm_ups_crs(wgs84_geom.shp.x, wgs84_geom.shp.y)
    utm_projection = Projection(utm_crs, utm_pixel_size, -utm_pixel_size)
    utm_geom = src_geom.to_projection(utm_projection)

    tile = (int(utm_geom.shp.x) // pixels_per_tile, int(utm_geom.shp.y) // pixels_per_tile)
    bounds = [
        tile[0] * pixels_per_tile,
        tile[1] * pixels_per_tile,
        (tile[0] + 1) * pixels_per_tile,
        (tile[1] + 1) * pixels_per_tile,
    ]

    window_name = f"{utm_crs.to_string()}_{tile[0]}_{tile[1]}"
    window = Window(
        window_root=os.path.join(out_dir, group, window_name),
        group=group,
        name=window_name,
        projection=utm_projection,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()
