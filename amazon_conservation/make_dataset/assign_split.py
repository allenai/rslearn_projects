import glob
import hashlib
import json
import os
import tqdm

ds_root = "/multisat/datasets/rslearn_amazon_conservation_closetime/"

example_id_to_label = {}
fnames = glob.glob(os.path.join(ds_root, "windows/*/*/layers/label/data.geojson"))
for fname in tqdm.tqdm(fnames):
    with open(fname) as f:
        category = json.load(f)["features"][0]["properties"]["new_label"]
        if category in ["unknown", "unlabeled", "human", "natural"]:
            continue
        parts = fname.split("/")
        group = parts[-5]
        example_id = parts[-4]
        example_id_to_label[(group, example_id)] = category

split_data = {}
for group, example_id in example_id_to_label.keys():
    if group in ["peru3", "peru3_flagged_in_peru", "peru_interesting"]:
        is_val = hashlib.sha256(example_id.encode()).hexdigest()[0] in ["0", "1", "2", "3", "4", "5", "6", "7"]
        if is_val:
            split_data[example_id] = "val"
        else:
            split_data[example_id] = "train"
    elif group in ["nadia2", "nadia3", "brazil_interesting"]:
        split_data[example_id] = "train"
