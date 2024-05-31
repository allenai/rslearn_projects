"""
Quick script to figure out how many items there are across the windows.
Its done in parallel so it can be faster.
"""

import json
import multiprocessing
import os

import tqdm

root_dir = "/mnt/landsat_1/windows/"

def get_windows(group):
    paths = []
    for window_name in os.listdir(os.path.join(root_dir, group)):
        paths.append(os.path.join(root_dir, group, window_name, "items.json"))
    return paths

def get_item_names(fname):
    if not os.path.exists(fname):
        return None
    with open(fname) as f:
        data = json.load(f)
    layer_by_name = {layer["layer_name"]: layer for layer in data}
    if "sentinel2" not in layer_by_name:
        return None
    item_names = set()
    for group in layer_by_name["sentinel2"]["serialized_item_groups"]:
        for item in group:
            item_names.add(item["name"])
    return item_names

p = multiprocessing.Pool(64)

groups = os.listdir(root_dir)
paths = []
outputs = p.imap_unordered(get_windows, groups)
for output in tqdm.tqdm(outputs, total=len(groups)):
    paths.extend(output)

item_names = set()
outputs = p.imap_unordered(get_item_names, paths)
for output in tqdm.tqdm(outputs, total=len(paths)):
    item_names.update(output)

print(len(item_names))
