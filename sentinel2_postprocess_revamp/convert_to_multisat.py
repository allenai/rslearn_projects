"""
Convert the materialized rslearn windows initialized via create_windows.py to multisat format.
Specifically, we output to space-like dataset so that we can set up multiple tasks.
Since we want to predict category, length, width, course, and speed with different heads.
"""

import glob
import hashlib
import json
import os
import shutil

import tqdm

ds_root = "/data/favyenb/rslearn_sentinel2_vessel_postprocess/"
bands = ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "b08", "b09", "b10", "b11", "b12", "b8a"]
vessel_categories = ["cargo", "tanker", "passenger", "service", "pleasure", "fishing", "enforcement", "sar"]
length_buckets = [10, 20, 30, 50, 80, 120, 180, 250, 320]
width_buckets = [4, 8, 15, 30, 60, 90]
speed_buckets = [5, 10]
heading_buckets = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

def get_label_from_buckets(buckets, value):
    for idx, bucket in enumerate(buckets):
        if value < bucket:
            return idx
    return len(buckets)

for band in bands:
    os.makedirs(os.path.join(ds_root, "images", "x", band), exist_ok=True)
os.makedirs(os.path.join(ds_root, "images", "x", "tci"), exist_ok=True)
for task in ["category", "length", "width", "speed", "heading"]:
    os.makedirs(os.path.join(ds_root, "labels", task), exist_ok=True)

window_paths = glob.glob(os.path.join(ds_root, "windows/*/*"))
window_paths.sort()
train_split = []
val_split = []
for idx, window_path in enumerate(tqdm.tqdm(window_paths)):
    path_parts = window_path.split("/")
    window_name = path_parts[-1]
    is_val = hashlib.md5(window_name.encode()).hexdigest()[0] in ["0", "1"]
    fake_tile_name = f"0_{idx}"

    info_fname = os.path.join(window_path, "info.json")
    if not os.path.exists(info_fname):
        continue
    with open(info_fname) as f:
        info = json.load(f)

    # Skip if any bands don't exist for whatever reason.
    all_bands_exist = True
    for band in bands:
        src_fname = os.path.join(window_path, "layers", "sentinel2", band.upper(), "image.png")
        dst_fname = os.path.join(ds_root, "images", "x", band, f"{fake_tile_name}.png")
        if not os.path.exists(src_fname):
            all_bands_exist = False
            print(src_fname)
            continue
        if os.path.exists(dst_fname):
            continue
        shutil.copyfile(src_fname, dst_fname)

        if band == "b01":
            # Need these tci files to exist for multisat dataset to load it properly.
            tci_fname = os.path.join(ds_root, "images", "x", "tci", f"{fake_tile_name}.png")
            if not os.path.exists(tci_fname):
                shutil.copyfile(src_fname, tci_fname)
    if not all_bands_exist:
        continue

    if info["type"] and info["type"] in vessel_categories:
        with open(os.path.join(ds_root, "labels", "category", f"{fake_tile_name}.txt"), "w") as f:
            f.write(info["type"])

    if info["length"] and info["length"] >= 5 and info["length"] < 460:
        with open(os.path.join(ds_root, "labels", "length", f"{fake_tile_name}.txt"), "w") as f:
            f.write(str(get_label_from_buckets(length_buckets, info["length"])))

    if info["width"] and info["width"] >= 2 and info["width"] < 120:
        with open(os.path.join(ds_root, "labels", "width", f"{fake_tile_name}.txt"), "w") as f:
            f.write(str(get_label_from_buckets(width_buckets, info["width"])))

    if info["sog"] and info["sog"] > 0 and info["sog"] < 60:
        with open(os.path.join(ds_root, "labels", "speed", f"{fake_tile_name}.txt"), "w") as f:
            f.write(str(get_label_from_buckets(speed_buckets, info["sog"])))

    if info["cog"] and info["cog"] >= 0 and info["cog"] <= 360 and info["sog"] and info["sog"] > 5 and info["sog"] < 50:
        with open(os.path.join(ds_root, "labels", "heading", f"{fake_tile_name}.txt"), "w") as f:
            f.write(str(get_label_from_buckets(heading_buckets, info["cog"])))

    if is_val:
        val_split.append((0, idx))
    else:
        train_split.append((0, idx))

os.makedirs(os.path.join(ds_root, "splits"), exist_ok=True)
with open(os.path.join(ds_root, "splits", "train.json"), "w") as f:
    json.dump(train_split, f)
with open(os.path.join(ds_root, "splits", "val.json"), "w") as f:
    json.dump(val_split, f)
