import os

import numpy as np
from PIL import Image

target_dir = "/multisat/datasets/rslearn_landsat/2024-07-18-joe-check-training-phase1/windows/phase2a/"

for window_id in os.listdir(target_dir):
    images = {}
    missing = False
    for band in ["B2", "B3", "B4", "B8"]:
        band_fname = os.path.join(target_dir, window_id, "layers", "landsat", band, "image.png")
        if not os.path.exists(band_fname):
            missing = True
            break
        images[band] = np.array(Image.open(band_fname))
        if band != "B8":
            images[band] = images[band].repeat(repeats=2, axis=0).repeat(repeats=2, axis=1)
    if missing:
        continue
    for band in ["B2", "B3", "B4"]:
        images[band + "_sharp"] = images[band].astype(np.int32)
    total = np.clip((images["B2_sharp"] + images["B3_sharp"] + images["B4_sharp"])//3, 1, 255)
    for band in ["B2", "B3", "B4"]:
        images[band + "_sharp"] = np.clip(images[band + "_sharp"] * images["B8"] // total, 0, 255).astype(np.uint8)
    rgb = np.stack([images["B4_sharp"], images["B3_sharp"], images["B2_sharp"]], axis=2)
    rgb_fname = os.path.join(target_dir, window_id, "layers", "landsat", "R_G_B", "image.png")
    os.makedirs(os.path.dirname(rgb_fname), exist_ok=True)
    Image.fromarray(rgb).save(rgb_fname)
