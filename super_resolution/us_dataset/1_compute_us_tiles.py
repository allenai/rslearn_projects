"""
Populate windows in remsen super-resolution dataset that span the entire continental US.
1. Extract US polygon from Natural Earth shapefile.
2. Sample points every 5 km.
3. Find appropriate UTM projection and tile for each point.
4. Get the set of distinct tiles and populate them.
"""

import fiona
import json
import multiprocessing
import numpy as np
from rasterio.crs import CRS
import rasterio.warp
import shapely.geometry
import tqdm

from remsen.dataset.window import Window
from remsen.utils import get_utm_ups_projection

# Meters per degree at equator (maximum).
DEGREE_METERS = 111111

# Grid size in pixel coordinates.
GRID_SIZE = 512

METERS_PER_PIXEL = 10

us_feature = None
with fiona.open("/home/ubuntu/remsen/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp") as f:
    for feat in f:
        if feat["properties"]["ADM0_ISO"] != "USA":
            continue
        us_feature = feat
        break

# The continental US should be first polygon last time I checked.
# Not sure how else to get it.
geom = shapely.geometry.shape(us_feature["geometry"])
polygon = geom.geoms[0]

# Sample points on 5 km grid.
# For each point, find the appropriate UTM zone and tile.
src_crs = CRS.from_epsg(4326)
def get_batch_tiles(batch):
    tiles = set()
    for lon, lat in batch:
        point = shapely.geometry.Point(lon, lat)
        if not polygon.contains(point):
            continue

        utm_crs = get_utm_ups_projection(lon, lat)
        utm_point = rasterio.warp.transform_geom(src_crs, utm_crs, point)
        utm_x, utm_y = utm_point["coordinates"]
        tile = (
            utm_crs.to_epsg(),
            int(utm_x / METERS_PER_PIXEL) // GRID_SIZE,
            int(utm_y / METERS_PER_PIXEL) // GRID_SIZE,
        )
        tiles.add(tile)
    return tiles

bounds = polygon.bounds
points = []
for lon in np.arange(bounds[0], bounds[2], GRID_SIZE * METERS_PER_PIXEL / DEGREE_METERS):
    for lat in np.arange(bounds[1], bounds[3], GRID_SIZE * METERS_PER_PIXEL / DEGREE_METERS):
        points.append((lon, lat))
batches = []
for i in range(0, len(points), 100):
    batches.append(points[i:i+1000])
p = multiprocessing.Pool(64)
outputs = p.imap_unordered(get_batch_tiles, batches)
tiles = set()
print(f"processing {len(points)} points in {len(batches)} batches")
for output in tqdm.tqdm(outputs, desc="Collecting tiles", total=len(batches)):
    tiles |= output
p.close()

with open("continental_us_utm_tiles.json", "w") as f:
    json.dump(list(tiles), f)
