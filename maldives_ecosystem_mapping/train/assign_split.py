import glob
import json
import os
import sys

import random
import tqdm

ds_root = sys.argv[1]
num_val = int(sys.argv[2])

window_metadatas = glob.glob(os.path.join(ds_root, "windows", "crops", "*", "metadata.json"))
random.shuffle(window_metadatas)
val_windows = window_metadatas[0:num_val]
train_windows = window_metadatas[num_val:]

for window_list, split in [(val_windows, "val"), (train_windows, "train")]:
    print(f"applying split {split} with {len(window_list)} windows")
    for metadata_fname in tqdm.tqdm(window_list):
        with open(metadata_fname) as f:
            metadata = json.load(f)
        if "options" not in metadata or metadata["options"] is None:
            metadata["options"] = {}
        metadata["options"]["split"] = split
        with open(metadata_fname + ".tmp", "w") as f:
            json.dump(metadata, f)
        os.rename(metadata_fname + ".tmp", metadata_fname)
