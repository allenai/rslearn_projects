"""
We want to have images in the same year and location as the labels, with one image per calendar month.

We will first identify a list of (tile, year) that are needed.
The tiles will be in WebMercator at 10 m/pixel (not any particular zoom level), with resolution of each tile at 32768x32768.

For now there are three datasets:

- CDL consists of .tif files. We'll split it up into a grid and just download the grid cells with non-zero label. Only use 2017-2023 due to Sentinel-2 time range.
- NCCM is similar but only three .tif files.
- EuroCrops consists of shapefiles. Each shapefile covers a certain country and has a specific year, we'll iterate over the polygons and add the (tile, year) tuples.

Assumption is that this torchgeo code has been run:

from torchgeo.datasets import CDL, EuroCrops, NCCM, SouthAmericaSoybean
CDL(paths='./data/cdl/', download=True, checksum=True, years=[2023, 2022, 2021, 2020, 2019, 2018, 2017])
EuroCrops(paths='./data/eurocrops/', download=True, checksum=True)
NCCM(paths="./data/nccm/", download=True, checksum=True)
SouthAmericaSoybean(paths="./data/sas/", download=True, checksum=True)
"""

import glob
import json
import multiprocessing
import os

import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.warp
import shapely
import tqdm

CHIP_SIZE = 32

data_paths = [
    ("data/cdl/", "data/sentinel2/crop_type_tiles_cdl.json"),
    ("data/eurocrops/", "data/sentinel2/crop_type_tiles_eurocrops.json"),
    ("data/nccm/", "data/sentinel2/crop_type_tiles_nccm.json"),
    ("data/sas/", "data/sentinel2/crop_type_tiles_sas.json"),
    ("data/agrifieldnet/field_ids/", "data/sentinel2/crop_type_tiles_agrifieldnet.json"),
    ("data/southafrica/field_ids/", "data/sentinel2/crop_type_tiles_southafrica.json"),
]
years = range(2017, 2024)

def get_year(fname):
    for year in years:
        if str(year) in fname:
            return year
    if "agrifieldnet" in fname:
        return 2021
    if "southafrica" in fname:
        return 2017
    raise ValueError(f"no year found in {fname}")

webmercator_fiona_crs = fiona.crs.from_epsg(3857)
webmercator_rasterio_crs = rasterio.crs.CRS.from_epsg(3857)

pixels_per_tile = 256
pixel_size = 10

def get_shp_tiles(fname):
    year = get_year(fname)
    print(f"processing {fname} with detected year {year}")
    tiles = set()
    with fiona.open(fname, "r") as src:
        xs = []
        ys = []
        for feature in src:
            shp = shapely.geometry.shape(feature["geometry"])
            center = shp.centroid
            xs.append(center.x)
            ys.append(center.y)
        xs, ys = fiona.transform.transform(src.crs, webmercator_fiona_crs, xs, ys)
        for x, y in zip(xs, ys):
            tile = (int(x/pixel_size)//pixels_per_tile, int(y/pixel_size)//pixels_per_tile)
            tiles.add((tile, year))

    print(f"found {len(tiles)} tiles in {fname}")
    return tiles

def get_tif_tiles(data_path, fname):
    year = get_year(fname)
    print(f"processing {fname} with detected year {year}")
    tiles = set()
    with rasterio.open(fname) as src:
        data = src.read(1)

        if 'cdl' in data_path:
            data = (data == 1) | (data == 3) | (data == 5) | (data == 12) | (data == 13) | (data == 22) | (data == 23) | (data == 24)
            data = data.astype(np.uint8)

        if 'nccm' in data_path:
            # paddy rice - keep
            data[data == 0] = 4
            # nodata - ignore
            data[data == 15] = 0

        rows = []
        cols = []
        for row in range(0, src.height, CHIP_SIZE):
            print(f"{fname} {row}/{src.height}")
            for col in range(0, src.width, CHIP_SIZE):
                crop = data[row:row+CHIP_SIZE, col:col+CHIP_SIZE]
                if np.count_nonzero(crop) < 128 and 'agrifieldnet' not in fname and 'southafrica' not in fname:
                    continue
                rows.append(row + CHIP_SIZE//2)
                cols.append(col + CHIP_SIZE//2)
        xs, ys = src.xy(rows, cols)
        xs, ys = rasterio.warp.transform(src.crs, webmercator_rasterio_crs, xs, ys)
        for x, y in zip(xs, ys):
            tile = (int(x/pixel_size)//pixels_per_tile, int(y/pixel_size)//pixels_per_tile)
            tiles.add((tile, year))

    print(f"found {len(tiles)} tiles in {fname}")
    return tiles

def get_tiles(job):
    data_path, fname = job
    if fname.endswith(".shp"):
        return get_shp_tiles(fname)
    elif fname.endswith(".tif"):
        return get_tif_tiles(data_path, fname)
    else:
        raise Exception("bad fname " + fname)

p = multiprocessing.Pool(64)

for data_path, out_fname in data_paths:
    print(data_path)
    tiles = set()

    shp_fnames = glob.glob(os.path.join(data_path, "**/*.shp"), recursive=True)
    tif_fnames = glob.glob(os.path.join(data_path, "**/*.tif"), recursive=True)
    fnames = shp_fnames + tif_fnames
    jobs = [(data_path, fname) for fname in fnames]
    outputs = p.imap_unordered(get_tiles, jobs)
    for cur_tiles in tqdm.tqdm(outputs, total=len(jobs)):
        tiles = tiles.union(cur_tiles)

    with open(out_fname, "w") as f:
        json.dump(list(tiles), f)

p.close()
