"""Convert the remaining labels, which should be the ones that are computed by the NAIP
model and used to supervise the Sentinel-2 model.
"""

import json
import os
import random
from datetime import datetime, timedelta, timezone

import shapely
from upath import UPath

from ..lib import convert_window

in_dir = "/multisat/labels/renewable_infra_point_naip_supervision/"
out_path = UPath("/multisat/datasets/rslearn_datasets_satlas/wind_turbine/")
group = "naip"

# Get the existing labels in the label group.
# We will only add windows in in_dir that don't already exist.
existing = set()
for window_name in os.listdir(os.path.join(out_path, "windows", "label")):
    parts = window_name.split("_")
    existing.add((int(parts[0]) // 512, int(parts[1]) // 512))

for fname in os.listdir(in_dir):
    parts = fname.split(".")[0].split("_")
    tile = (int(parts[0]), int(parts[1]))
    if tile in existing:
        continue

    bounds = [tile[0] * 512, tile[1] * 512, (tile[0] + 1) * 512, (tile[1] + 1) * 512]

    month = random.randint(1, 1)
    ts = datetime(2020, month, 1, tzinfo=timezone.utc)
    time_range = (ts, ts + timedelta(days=90))

    labels = []
    with open(os.path.join(in_dir, fname)) as f:
        for x1, y1, x2, y2, category in json.load(f):
            point = shapely.Point(
                bounds[0] + (x1 + x2) / 2,
                bounds[1] + (y1 + y2) / 2,
            )
            properties = {"category": "turbine"}
            labels.append((point, properties))

    convert_window(
        root_dir=out_path,
        group=group,
        zoom=13,
        bounds=bounds,
        labels=labels,
        time_range=time_range,
    )
