"""Picks least cloudy images based on RGB and copies to another output directory."""

import multiprocessing
import os
import shutil

import numpy as np
import rasterio
import tqdm

in_dir = "/data/favyenb/rslearn_crop_type/windows/sact/"
out_dir = "/data/favyenb/rslearn_crop_type/sact_outputs/"

jobs = {}
for window_name in os.listdir(in_dir):
    label = "_".join(window_name.split("_")[:-1])
    if label not in jobs:
        jobs[label] = []
    jobs[label].append(window_name)


def handle_example(job):
    label, window_names = job
    candidates = []
    for window_name in window_names:
        image_fname = os.path.join(
            in_dir,
            window_name,
            "layers/sentinel2/B01_B02_B03_B04_B05_B06_B07_B08_B09_B10_B11_B12_B8A/geotiff.tif",
        )
        if not os.path.exists(image_fname):
            continue
        with rasterio.open(image_fname) as src:
            im = src.read()[1:3, :, :]
            bad_pixels = np.count_nonzero(
                (im.max(axis=0) == 0) | (im.min(axis=0) > 1400)
            )
            candidates.append((image_fname, bad_pixels))
    candidates.sort(key=lambda t: t[1])
    if len(candidates) == 0:
        return
    shutil.copyfile(candidates[0][0], os.path.join(out_dir, f"{label}.tif"))


p = multiprocessing.Pool(64)
outputs = p.imap_unordered(handle_example, list(jobs.items()))
for _ in tqdm.tqdm(outputs, total=len(jobs)):
    pass
p.close()
