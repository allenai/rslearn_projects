"""
We deleted the Sentinel-2 images from GCS so Viraj cannot get the images from there anymore.
So now need to get it from rslearn.
This script takes GeoJSONs he sent and populates windows in rslearn dataset.
"""

from datetime import datetime, timezone
import json
import math
import os

from rasterio import CRS
import shapely

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_projection

geojson_fnames = [
    "/tmp/observation_research.geojson",
    "/tmp/oil_derrick_rig.geojson",
    "/tmp/production_platform.geojson",
]
time_ranges = [
    [datetime(2023, 12, 1, tzinfo=timezone.utc), datetime(2023, 12, 31, tzinfo=timezone.utc)],
    [datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 1, 31, tzinfo=timezone.utc)],
    [datetime(2024, 2, 1, tzinfo=timezone.utc), datetime(2024, 2, 29, tzinfo=timezone.utc)],
]

out_dir = "/data/favyenb/rslearn_sentinel2_platforms/windows/"
WEBMERCATOR_GROUP = "web_mercator"
UTM_GROUP = "utm"

webmercator_crs = CRS.from_epsg(3857)
web_mercator_m = 2 * math.pi * 6378137
pixel_size = web_mercator_m / (2**13) / 512
web_mercator_projection = Projection(webmercator_crs, pixel_size, -pixel_size)

for geojson_fname in geojson_fnames:
    with open(geojson_fname) as f:
        data = json.load(f)
    for feat_idx, feat in enumerate(data["features"]):
        geometry = feat["geometry"]
        assert geometry["type"] == "Point" and len(geometry["coordinates"]) == 2
        lon, lat = geometry["coordinates"]
        category = feat["properties"]["finer_category"].replace(" ", "_").replace("/", "_")
        for time_range in time_ranges:
            geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), time_range)
            geom = geom.to_projection(web_mercator_projection)
            bounds = [
                geom.shp.x - 32,
                geom.shp.y - 32,
                geom.shp.x + 32,
                geom.shp.y + 32,
            ]
            window_name = f"{category}_{feat_idx}_{lon}_{lat}_{time_range[0].year}_{time_range[0].month}"
            window = Window(
                window_root=os.path.join(out_dir, WEBMERCATOR_GROUP, window_name),
                group=WEBMERCATOR_GROUP,
                name=window_name,
                projection=web_mercator_projection,
                bounds=bounds,
                time_range=time_range,
            )
            window.save()

        for time_range in time_ranges:
            geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), time_range)
            utm_crs = get_utm_ups_projection(lon, lat)
            utm_projection = Projection(utm_crs, 10 ,-10)
            geom = geom.to_projection(utm_projection)
            bounds = [
                geom.shp.x - 32,
                geom.shp.y - 32,
                geom.shp.x + 32,
                geom.shp.y + 32,
            ]
            window_name = f"{category}_{feat_idx}_{lon}_{lat}_{time_range[0].year}_{time_range[0].month}_utm"
            window = Window(
                window_root=os.path.join(out_dir, UTM_GROUP, window_name),
                group=UTM_GROUP,
                name=window_name,
                projection=utm_projection,
                bounds=bounds,
                time_range=time_range,
            )
            window.save()
