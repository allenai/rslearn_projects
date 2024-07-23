"""
Get a sample of 1000 vessel labels.
"""

import csv
from datetime import timedelta
import json
import os
import random

import numpy as np
from PIL import Image
import shapely

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import LocalFileAPI, STGeometry

in_dir = "/multisat/datasets/rslearn_landsat/"
group = "labels_utm"
count = 1000
out_csv_fname = "/multisat/datasets/rslearn_landsat/2024-07-18-joe-check-training-phase1/data.csv"
out_ds_dir = "/multisat/datasets/rslearn_landsat/2024-07-18-joe-check-training-phase1/"

group_dir = os.path.join(in_dir, "windows", group)
# Pick up to one vessel label per window.
options = []
for window_id in os.listdir(group_dir):
    window = Window.load(LocalFileAPI(os.path.join(group_dir, window_id)))
    with open(os.path.join(group_dir, window_id, "gt.json")) as f:
        labels = json.load(f)
    if len(labels) == 0:
        continue
    if not os.path.exists(os.path.join(group_dir, window_id, "layers", "landsat", "B8", "image.png")):
        continue

    # Some labels are out of the image bounds for some reason.
    good_labels = []
    for label in labels:
        x1, y1, x2, y2, category = label
        cx = (x1+x2)//2
        cy = (y1+y2)//2
        if cx < 0 or cy < 0 or cx >= window.bounds[2] - window.bounds[0] or cy >= window.bounds[3] - window.bounds[1]:
            continue
        good_labels.append(label)

    x1, y1, x2, y2, category = random.choice(good_labels)
    cx = (x1+x2)//2
    cy = (y1+y2)//2
    src_geom = STGeometry(
        window.projection,
        shapely.Point(window.bounds[0] + cx, window.bounds[1] + cy),
        None,
    )
    dst_geom = src_geom.to_projection(WGS84_PROJECTION)
    lon = dst_geom.shp.x
    lat = dst_geom.shp.y
    format_string = "%Y-%m-%dT%H:%M:%SZ"
    from_time = (window.time_range[0] - timedelta(minutes=20)).strftime(format_string)
    to_time = (window.time_range[1] + timedelta(minutes=20)).strftime(format_string)
    url = f"https://apps.sentinel-hub.com/eo-browser/?zoom=16&lat={lat}&lng={lon}&fromTime={from_time}&toTime={to_time}&datasetId=AWS_LOTL1&layerId=2_TRUE_COLOR_PANSHARPENED"
    options.append({
        "lon": lon,
        "lat": lat,
        "window_id": window_id,
        "col": cx,
        "row": cy,
        "url": url,
    })

    # Get pan-sharpened RGB image.
    images = {}
    for band in ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]:
        images[band] = np.array(Image.open(os.path.join(in_dir, "windows", group, window_id, "layers", "landsat", band, "image.png")))
        if band != "B8":
            images[band] = images[band].repeat(repeats=2, axis=0).repeat(repeats=2, axis=1)
    for band in ["B2", "B3", "B4"]:
        images[band + "_sharp"] = images[band].astype(np.int32)
    total = np.clip((images["B2_sharp"] + images["B3_sharp"] + images["B4_sharp"])//3, 1, 255)
    for band in ["B2", "B3", "B4"]:
        images[band + "_sharp"] = np.clip(images[band + "_sharp"] * images["B8"] // total, 0, 255).astype(np.uint8)
    images["rgb"] = np.stack([images["B4_sharp"], images["B3_sharp"], images["B2_sharp"]], axis=2)
    # Pad to make it easier to extract crops around vessels.
    for band in images.keys():
        if len(images[band].shape) == 3:
            images[band] = np.pad(images[band], [(32, 32), (32, 32), (0, 0)])
        else:
            images[band] = np.pad(images[band], [(32, 32), (32, 32)])

    # Write each label as a separate window.
    # We first put in in default.
    # Then later we'll move to "selected" group if we picked it.
    for x1, y1, x2, y2, category in good_labels:
        cx = (x1+x2)//2
        cy = (y1+y2)//2

        crop_window_name = f"{window_id}_{cx}_{cy}"
        crop_window_root = os.path.join(out_ds_dir, "windows", "default", crop_window_name)
        os.makedirs(crop_window_root, exist_ok=True)
        crop_window = Window(
            file_api=LocalFileAPI(crop_window_root),
            group="default",
            name=crop_window_name,
            projection=window.projection,
            bounds=[0, 0, 64, 64],
            time_range=window.time_range,
        )
        crop_window.save()

        # Select crop appropriately in the padded image.
        crop = images["rgb"][cy:cy+64, cx:cx+64, :]
        image_layer_dir = os.path.join(crop_window_root, "layers", "landsat", "R_G_B")
        os.makedirs(image_layer_dir, exist_ok=True)
        with open(os.path.join(image_layer_dir, "completed"), "w") as f:
            pass
        Image.fromarray(crop).save(os.path.join(image_layer_dir, "image.png"))

        for band in ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]:
            crop = images[band][cy:cy+64, cx:cx+64]
            image_layer_dir = os.path.join(crop_window_root, "layers", "landsat", band)
            os.makedirs(image_layer_dir, exist_ok=True)
            Image.fromarray(crop).save(os.path.join(image_layer_dir, "image.png"))

        label_layer_dir = os.path.join(crop_window_root, "layers", "label")
        os.makedirs(label_layer_dir, exist_ok=True)
        with open(os.path.join(label_layer_dir, "completed"), "w") as f:
            pass
        if not os.path.exists(os.path.join(label_layer_dir, "data.geojson")):
            with open(os.path.join(label_layer_dir, "data.geojson"), "w") as f:
                json.dump({
                    "type": "FeatureCollection",
                    "features": [{
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [[32, 32]],
                        },
                        "properties": {
                            "label": "unknown",
                        },
                    }],
                    "properties": crop_window.projection.serialize(),
                }, f)

csv_rows = random.sample(options, 1000)
with open(out_csv_fname, "w") as f:
    writer = csv.DictWriter(f, ["lon", "lat", "window_id", "col", "row", "url"])
    writer.writeheader()
    for csv_row in csv_rows:
        writer.writerow({k: str(v) for k, v in csv_row.items()})
        window_id, col, row = csv_row["window_id"], csv_row["col"], csv_row["row"]
        crop_window_name = f"{window_id}_{col}_{row}"
        os.rename(
            os.path.join(out_ds_dir, "windows", "default", crop_window_name),
            os.path.join(out_ds_dir, "windows", "selected", crop_window_name),
        )
