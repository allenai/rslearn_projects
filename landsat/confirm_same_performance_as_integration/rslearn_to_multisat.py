"""Create a custom dataset without labels for the downloaded Landsat images."""

import json
import os

import numpy as np
from PIL import Image

in_dir = "/data/favyenb/rslearn_landsat_inference_tmp/windows/default/"
out_dir = "/data/favyenb/rslearn_landsat_inference_tmp/multisat/landsat_vessels/"
split_fname = "/data/favyenb/rslearn_landsat_inference_tmp/multisat/split.json"
bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]

example_ids = []

for window_id in os.listdir(in_dir):
    for band in bands:
        im = np.array(
            Image.open(
                os.path.join(in_dir, window_id, "layers", "landsat", band, "image.png")
            )
        )
        if band != "B8":
            im = im.repeat(repeats=2, axis=0).repeat(repeats=2, axis=1)

        for col in range(0, im.shape[1], 512):
            for row in range(0, im.shape[0], 512):
                example_id = f"{window_id}_{col}_{row}"
                crop = im[row : row + 512, col : col + 512]
                cur_img_dir = os.path.join(out_dir, example_id, "images", "x")
                os.makedirs(cur_img_dir, exist_ok=True)
                Image.fromarray(crop).save(
                    os.path.join(cur_img_dir, band.lower() + ".png")
                )
                example_ids.append(example_id)

with open(split_fname, "w") as f:
    json.dump(example_ids, f)
