"""Simple script to copy the images.

Just need to run `model predict` with visualize_dir set to utm and webm respectively.
"""

import os
import shutil

webm_images = {}
for fname in os.listdir("webm"):
    if not fname.endswith("_gt.png"):
        continue
    parts = fname.split("_gt.png")[0].split("_")
    tile = (int(parts[0]), int(parts[1]))
    webm_images[tile] = fname
utm_images = {}
for fname in os.listdir("utm"):
    if not fname.endswith("_gt.png"):
        continue
    parts = fname.split("_gt.png")[0].split("_")
    tile = (int(parts[0]) // 512, int(parts[1]) // 512)
    utm_images[tile] = fname
good_keys = set(webm_images.keys()).intersection(utm_images.keys())
for tile in good_keys:
    shutil.copyfile(
        f"webm/{webm_images[tile]}", f"out/{tile[0]}_{tile[1]}_webmercator.png"
    )
    shutil.copyfile(f"utm/{utm_images[tile]}", f"out/{tile[0]}_{tile[1]}_utm.png")
