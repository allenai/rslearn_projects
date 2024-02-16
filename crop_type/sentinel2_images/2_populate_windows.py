from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
import sys
import tqdm

from rasterio.crs import CRS

from rslearn.dataset import Window
from rslearn.utils import Projection


tile_fname = sys.argv[1]
group = sys.argv[2]

out_dir = "/data/favyenb/rslearn_crop_type/windows/"

with open(tile_fname) as f:
    tile_years = json.load(f)

webmercator_crs = CRS.from_epsg(3857)
webmercator_m = 2 * math.pi * 6378137
zoom = 7
pixels_per_tile = 32768
pixel_size = webmercator_m / (2**zoom) / pixels_per_tile

def make_window(tile_year):
    tile, year = tile_year
    col = tile[0]
    row = -tile[1] - 1
    window_name = f"{col}_{row}_{year}"
    projection = Projection(webmercator_crs, pixel_size, -pixel_size)
    bounds = (col*32768, row*32768, (col+1)*32768, (row+1)*32768)
    time_range = (datetime(year, 7, 1, tzinfo=timezone.utc), datetime(year, 8, 1, tzinfo=timezone.utc))
    window = Window(
        window_root=os.path.join(out_dir, group, window_name),
        group=group,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()

p = multiprocessing.Pool(64)
outputs = p.imap_unordered(make_window, tile_years)
for _ in tqdm.tqdm(outputs, total=len(tile_years)):
    pass
p.close()
