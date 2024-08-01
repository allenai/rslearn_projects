"""This is based on 2024_03_27_recompute_masks in multisat amazon_conservation code.

This script will do a similar recompute process but instead of updating the mask,
the output is rslearn windows with the info.json and mask.png similar to
create_unlabeled_dataset.json.
"""

import json
import math
import multiprocessing
import os
import shutil
from datetime import datetime, timedelta, timezone

import rasterio
import rasterio.features
import tqdm
from rasterio.crs import CRS
from rslearn.dataset import Window
from rslearn.utils import Projection

# Tuples (alert tif, alertdate tif, example directory, annotation file(s)).
tasks = {
    "brazil": [
        "/multisat/datasets/amazon_conservation/2023-12-21-glad-unlabeled/alert_060W_10S_050W_00N.tif",
        "/multisat/datasets/amazon_conservation/2023-12-21-glad-unlabeled/alertDate_060W_10S_050W_00N.tif",
        "/multisat/labels/amazon_conservation_brazil/amazon_conservation/",
        ["/multisat/datasets/amazon_conservation/2024-01-17-annotation/brazil.json"],
    ],
    "peru": [
        "/multisat/datasets/amazon_conservation/2023-12-21-glad-unlabeled/alert_080W_10S_070W_00N.tif",
        "/multisat/datasets/amazon_conservation/2023-12-21-glad-unlabeled/alertDate_080W_10S_070W_00N.tif",
        "/multisat/labels/amazon_conservation_peru/amazon_conservation/",
        [
            "/multisat/datasets/amazon_conservation/2024-01-17-annotation/peru.json",
            "/multisat/datasets/amazon_conservation/2024-03-19-annotation/peru_subset.json",
        ],
    ],
    "peru2": [
        "/multisat/datasets/amazon_conservation/2023-12-21-glad-unlabeled/alert_080W_10S_070W_00N.tif",
        "/multisat/datasets/amazon_conservation/2023-12-21-glad-unlabeled/alertDate_080W_10S_070W_00N.tif",
        "/multisat/labels/amazon_conservation_peru2/amazon_conservation/",
        [],
    ],
}
group = "peru"
conf_fname, date_fname, label_dir, annotation_fnames = tasks[group]

# Dates in the GDAL GeoTIFF are measured in days since 2019-01-01.
date_base = datetime(2019, 1, 1, tzinfo=timezone.utc)

crop_size = 128
out_dir = "/data/favyenb/rslearn_amazon_conservation_closetime/"

web_mercator_crs = CRS.from_epsg(3857)
web_mercator_m = 2 * math.pi * 6378137
web_mercator_total_pixels = 2**13 * 512
pixel_size = web_mercator_m / web_mercator_total_pixels
web_mercator_projection = Projection(web_mercator_crs, pixel_size, -pixel_size)

# print('read confidences')
# conf_raster = rasterio.open(conf_fname)
# conf_data = conf_raster.read(1)
print("read dates")
date_raster = rasterio.open(date_fname)
date_data = date_raster.read(1)


def handle(example_id):
    parts = example_id.split("_")
    mercator_col = int(parts[2])
    mercator_row = int(parts[3])
    orig_col = int(parts[4])
    orig_row = int(parts[5])

    center_days = int(date_data[orig_row, orig_col])
    center_date = date_base + timedelta(days=center_days)

    # Create the new rslearn windows.
    bounds = (
        int(mercator_col) - web_mercator_total_pixels // 2 - crop_size // 2,
        int(mercator_row) - web_mercator_total_pixels // 2 - crop_size // 2,
        int(mercator_col) - web_mercator_total_pixels // 2 + crop_size // 2,
        int(mercator_row) - web_mercator_total_pixels // 2 + crop_size // 2,
    )
    time_range = (
        center_date,
        center_date + timedelta(days=60),
    )
    window = Window(
        window_root=os.path.join(out_dir, "windows", group, example_id),
        group=group,
        name=example_id,
        projection=web_mercator_projection,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()

    # Copy the mask.
    shutil.copyfile(
        os.path.join(label_dir, example_id, "images", "image_0", "mask.png"),
        os.path.join(window.window_root, "mask.png"),
    )


jobs = os.listdir(label_dir)
p = multiprocessing.Pool(32)
outputs = p.imap_unordered(handle, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs)):
    pass
p.close()

seen = set()
for annotation_fname in annotation_fnames:
    with open(annotation_fname) as f:
        for example_id, annot_data in json.load(f):
            # Only overwrite label in earlier annotation fname if current is not unlabeled.
            new_label = annot_data["new_label"]
            if new_label == "unlabeled" and example_id in seen:
                continue
            seen.add(example_id)

            with open(
                os.path.join(out_dir, "windows", group, example_id, "label.json"), "w"
            ) as f:
                json.dump(annot_data, f)
