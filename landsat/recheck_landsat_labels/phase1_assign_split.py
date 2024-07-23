import glob
import json
import os

import random
import tqdm

ds_root = "/multisat/datasets/rslearn_landsat/2024-07-18-joe-check-training-phase1/"
val_fraction = 0.3

window_metadatas = glob.glob(os.path.join(ds_root, "windows", "*", "*", "metadata.json"))
random.shuffle(window_metadatas)
num_val = int(len(window_metadatas) * val_fraction)
val_windows = window_metadatas[0:num_val]
train_windows = window_metadatas[num_val:]

for window_list, split in [(val_windows, "val"), (train_windows, "train")]:
    print(f"applying split {split} with {len(window_list)} windows")
    applied = 0
    for metadata_fname in tqdm.tqdm(window_list):
        # Only apply split if label is good, otherwise set split to bad.
        window_dir = os.path.dirname(metadata_fname)
        with open(os.path.join(window_dir, "layers", "label", "data.geojson")) as f:
            fc = json.load(f)
        label = fc["features"][0]["properties"]["label"]

        if label in ["correct", "incorrect"]:
            cur_split = split
            applied += 1
        else:
            cur_split = "bad"

        if label == "incorrect":
            weight = 16
        else:
            weight = 1

        with open(metadata_fname) as f:
            metadata = json.load(f)
        if "options" not in metadata or metadata["options"] is None:
            metadata["options"] = {}
        metadata["options"]["split"] = cur_split
        metadata["options"]["weight"] = weight
        with open(metadata_fname + ".tmp", "w") as f:
            json.dump(metadata, f)
        os.rename(metadata_fname + ".tmp", metadata_fname)

    print(f"success on {applied} windows")
