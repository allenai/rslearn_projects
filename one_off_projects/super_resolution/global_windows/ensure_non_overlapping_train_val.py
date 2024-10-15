"""This makes sure that Piper's train and val sets are non-overlapping.
For initial super-resolution run.
"""

import multiprocessing

import geopy.distance
import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils import Projection, STGeometry

train_fname = "/data/piperw/projects/satlas-super-resolution/train_tiles.txt"
val_fname = "/data/piperw/projects/satlas-super-resolution/val_tiles.txt"

pixel_size = 1.25
pixels_per_tile = 512


def get_center(line):
    parts = line.split("_")
    crs = CRS.from_epsg(int(parts[0]))
    col = int(parts[1])
    row = int(parts[2])

    projection = Projection(crs, pixel_size, -pixel_size)
    shp = shapely.Point((col + 0.5) * pixels_per_tile, (row + 0.5) * pixels_per_tile)
    src_geom = STGeometry(projection, shp, None)
    wgs84_geom = src_geom.to_projection(WGS84_PROJECTION)
    return (wgs84_geom.shp.y, wgs84_geom.shp.x, line)


def evaluate(job):
    train_centers, val_center = job
    for train_center in train_centers:
        distance = geopy.distance.distance(val_center[0:2], train_center[0:2]).meters
        if distance < 200:
            print(train_center, val_center)
            return False
    return True


p = multiprocessing.Pool(64)


def get_centers(fname):
    print("read", fname)
    with open(fname) as f:
        lines = f.readlines()
    centers = []
    outputs = p.imap_unordered(get_center, lines)
    for center in tqdm.tqdm(outputs, total=len(lines)):
        centers.append(center)
    return centers


train_centers = get_centers(train_fname)
val_centers = get_centers(val_fname)

print("compare")
outputs = p.imap_unordered(
    evaluate, [(train_centers, val_center) for val_center in val_centers]
)
num_good = 0
num_bad = 0
for output in tqdm.tqdm(outputs, total=len(val_centers)):
    if output:
        num_good += 1
    else:
        num_bad += 1
