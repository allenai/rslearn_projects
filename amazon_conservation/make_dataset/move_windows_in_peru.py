"""
Initial focus area is Peru.
But we sometimes get annotations in Ecuador, Brazil, etc. in the Peru set since we just
do it based on GLAD alert grid.
So here we move windows that are actually in Peru to a new group.
"""

import json
import math
import os
import sys

import fiona
import shapely.geometry

def mercator_to_geo(p, zoom=13, pixels=512):
    n = 2**zoom
    x = p[0] / pixels
    y = p[1] / pixels
    x = x * 360.0 / n - 180
    y = math.atan(math.sinh(math.pi * (1 - 2.0 * y / n)))
    y = y * 180 / math.pi
    return (x, y)

ds_root = sys.argv[1]
src_group = sys.argv[2]
dst_group = sys.argv[3]

country_fname = "/multisat/datasets/natural_earth_countries/ne_10m_admin_0_countries.shp"

peru_shp = None
with fiona.open(country_fname) as src:
    for feat in src:
        if feat["properties"]["ISO_A2"] != "PE":
            continue
        cur_shp = shapely.geometry.shape(feat["geometry"])
        if peru_shp:
            peru_shp = peru_shp.union(cur_shp)
        else:
            peru_shp = cur_shp

good_features = []
for example_id in os.listdir(os.path.join(ds_root, "windows", src_group)):
    parts = example_id.split("_")
    point = (int(parts[2]), int(parts[3]))
    point = mercator_to_geo(point, zoom=13, pixels=512)
    point = shapely.Point(point[0], point[1])
    if not peru_shp.contains(point):
        continue
    dst_window_dir = os.path.join(ds_root, "windows", dst_group, example_id)
    os.rename(
        os.path.join(ds_root, "windows", src_group, example_id),
        dst_window_dir,
    )
    with open(os.path.join(dst_window_dir, "metadata.json")) as f:
        metadata = json.load(f)
    metadata["group"] = dst_group
    with open(os.path.join(dst_window_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)
