"""Reformat from rslearn to multisat format.

This is because this project is more urgent than the other conversion ones.
So we really want to get the UTM model out.
(Instead of using rslearn to train the model.)
"""

import json
import multiprocessing
import os

import numpy as np
import tqdm
from PIL import Image

in_dir = "/data/favyenb/rslearn_datasets_satlas/sentinel2_vessel/"
out_dir = (
    "/data/favyenb/rslearn_datasets_satlas/sentinel2_vessel/multisat/sentinel2_vessel/"
)
bands = {
    "R_G_B": "tci",
    "B08": "b08",
    "B11": "b11",
    "B12": "b12",
}


def reformat(job):
    group, example_id = job
    cur_in_dir = os.path.join(in_dir, "windows", group, example_id)
    for src_band in bands.keys():
        if not os.path.exists(
            os.path.join(cur_in_dir, "layers", "sentinel2", src_band, "image.png")
        ):
            print(f"warning: missing {example_id}")
            return

    cur_out_dir = os.path.join(out_dir, example_id)
    cur_img_dir = os.path.join(cur_out_dir, "images", "x")
    os.makedirs(cur_img_dir, exist_ok=True)

    # Copy the vessel labels.
    with open(os.path.join(cur_in_dir, "metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(cur_in_dir, "layers", "label", "data.geojson")) as f:
        fc = json.load(f)
    boxes = []
    for feat in fc["features"]:
        geometry = feat["geometry"]
        assert geometry["type"] == "Point"
        assert feat["properties"]["category"] == "vessel"
        col = int(geometry["coordinates"][0] - metadata["bounds"][0])
        row = int(geometry["coordinates"][1] - metadata["bounds"][1])
        boxes.append((col - 15, row - 15, col + 15, row + 15, "vessel"))
    with open(os.path.join(cur_out_dir, "gt.json"), "w") as f:
        json.dump(boxes, f)

    # Mask each image to erase parts that were not in the WebMercator image.
    mask = np.array(
        Image.open(os.path.join(cur_in_dir, "layers", "mask", "mask", "image.png"))
    )
    mask[mask == 255] = 1
    for src_band, dst_band in bands.items():
        in_img_fname = os.path.join(
            cur_in_dir, "layers", "sentinel2", src_band, "image.png"
        )
        out_img_fname = os.path.join(cur_img_dir, f"{dst_band}.png")
        im = np.array(Image.open(in_img_fname))
        if len(im.shape) == 3:
            im *= mask[:, :, None]
        else:
            im *= mask
        Image.fromarray(im).save(out_img_fname)


jobs = []
for group in os.listdir(os.path.join(in_dir, "windows")):
    group_dir = os.path.join(in_dir, "windows", group)
    for example_id in os.listdir(group_dir):
        jobs.append((group, example_id))
p = multiprocessing.Pool(64)
outputs = p.imap_unordered(reformat, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs)):
    pass
p.close()
