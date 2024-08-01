"""Supposing we have created a split.json file mapping from the window_name to split (train/val).
Then this will update the window tags to match this JSON file.
"""

import glob
import json
import os

import tqdm

in_fname = "/multisat/datasets/rslearn_amazon_conservation_closetime/split.json"
ds_root = "/multisat/datasets/rslearn_amazon_conservation_closetime/"

with open(in_fname) as f:
    split_data = json.load(f)

window_metadatas = glob.glob(
    os.path.join(ds_root, "windows", "*", "*", "metadata.json")
)

for metadata_fname in tqdm.tqdm(window_metadatas):
    window_name = metadata_fname.split("/")[-2]
    if window_name not in split_data:
        continue
    with open(metadata_fname) as f:
        metadata = json.load(f)
    if "options" not in metadata or metadata["options"] is None:
        metadata["options"] = {}
    metadata["options"]["split"] = split_data[window_name]
    with open(metadata_fname, "w") as f:
        json.dump(metadata, f)
