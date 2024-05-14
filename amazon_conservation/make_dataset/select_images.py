"""
Picks which images are least cloudy based on RGB and writes them to JSON file in the window dir.
"""
from datetime import datetime, timedelta
import glob
import json
import multiprocessing
import os

import numpy as np
from PIL import Image
import tqdm

ds_root = "/data/favyenb/rslearn_amazon_conservation_closetime/"
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
        im = np.array(Image.open(os.path.join(window_dir, fname)))[32:96, 32:96, :]
        image_lists[k].append((im, fname))
    best_images = {}
    for k, image_list in image_lists.items():
        if len(image_list) < min_choices:
            return
        image_list.sort(key=lambda t: np.count_nonzero((t[0].max(axis=2) == 0) | (t[0].min(axis=2) > 140)))
        best_images[k] = []
        for im, fname in image_list[0:num_outs]:
            layer_name = fname.split("/")[-3]
            layer_time = layer_times[layer_name]
            best_images[k].append((fname, layer_time))

    with open(os.path.join(window_dir, "good_images.json"), "w") as f:
        json.dump(best_images, f)

window_dirs = glob.glob(os.path.join(ds_root, "windows/*/*"))
p = multiprocessing.Pool(64)
outputs = p.imap_unordered(handle_example, window_dirs)
for _ in tqdm.tqdm(outputs, total=len(window_dirs)):
    pass
p.close()
