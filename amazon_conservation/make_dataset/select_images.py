"""Picks which images are least cloudy based on RGB and write them to a separate layer.
The new layers are prefixed with "best_" like "best_pre_0".
"""

import glob
import json
import multiprocessing
import os
import shutil

import numpy as np
import tqdm
from PIL import Image

ds_root = "/multisat/datasets/rslearn_amazon_conservation_closetime/"
group = "peru_interesting"
num_outs = 3
min_choices = 5


def handle_example(window_dir):
    if not os.path.exists(os.path.join(window_dir, "items.json")):
        return

    # Get the timestamp of each expected layer.
    layer_times = {}
    with open(os.path.join(window_dir, "items.json")) as f:
        item_data = json.load(f)
        for layer_data in item_data:
            layer_name = layer_data["layer_name"]
            if "planet" in layer_name:
                continue
            for group_idx, group in enumerate(layer_data["serialized_item_groups"]):
                if group_idx == 0:
                    cur_layer_name = layer_name
                else:
                    cur_layer_name = f"{layer_name}.{group_idx}"
                layer_times[cur_layer_name] = group[0]["geometry"]["time_range"][0]

    # Find best pre and post images.
    image_lists = {"pre": [], "post": []}
    options = glob.glob("layers/*/R_G_B/image.png", root_dir=window_dir)
    for fname in options:
        # "pre" or "post"
        k = fname.split("/")[-3].split(".")[0].split("_")[0]
        if "planet" in k:
            continue
        im = np.array(Image.open(os.path.join(window_dir, fname)))[32:96, 32:96, :]
        image_lists[k].append((im, fname))

    # Copy the images to new "best" layer.
    # Keep track of the timestamps and write them to a separate file.
    best_times = {}
    for k, image_list in image_lists.items():
        if len(image_list) < min_choices:
            return
        image_list.sort(
            key=lambda t: np.count_nonzero(
                (t[0].max(axis=2) == 0) | (t[0].min(axis=2) > 140)
            )
        )
        for idx, (im, fname) in enumerate(image_list[0:num_outs]):
            dst_layer = f"best_{k}_{idx}"
            dst_dir = os.path.join(window_dir, "layers", dst_layer, "R_G_B")
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copyfile(
                os.path.join(window_dir, fname),
                os.path.join(dst_dir, "image.png"),
            )

            src_layer = fname.split("/")[-3]
            layer_time = layer_times[src_layer]
            best_times[dst_layer] = layer_time

    with open(os.path.join(window_dir, "best_times.json"), "w") as f:
        json.dump(best_times, f)


window_dirs = glob.glob(os.path.join(ds_root, "windows", group, "*"))
p = multiprocessing.Pool(64)
outputs = p.imap_unordered(handle_example, window_dirs)
for _ in tqdm.tqdm(outputs, total=len(window_dirs)):
    pass
p.close()
