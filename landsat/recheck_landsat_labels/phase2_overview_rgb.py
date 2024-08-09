import multiprocessing
import os

import numpy as np
import tqdm
from PIL import Image

target_dir = "/data/favyenb/rslearn_landsat/2024-07-18-joe-check-training-phase1/windows/phase2a_zoomout/"


def process(window_id):
    # Read images.
    images = {}
    for band in ["B2", "B3", "B4", "B8"]:
        band_fname = os.path.join(
            target_dir, window_id, "layers", "landsat", band, "image.png"
        )
        if not os.path.exists(band_fname):
            return
        images[band] = np.array(Image.open(band_fname))
        if band != "B8":
            images[band] = (
                images[band].repeat(repeats=2, axis=0).repeat(repeats=2, axis=1)
            )

    # Pan-sharpen.
    for band in ["B2", "B3", "B4"]:
        images[band + "_sharp"] = images[band].astype(np.int32)
    total = np.clip(
        (images["B2_sharp"] + images["B3_sharp"] + images["B4_sharp"]) // 3, 1, 255
    )
    for band in ["B2", "B3", "B4"]:
        images[band + "_sharp"] = np.clip(
            images[band + "_sharp"] * images["B8"] // total, 0, 255
        ).astype(np.uint8)
    rgb = np.stack([images["B4_sharp"], images["B3_sharp"], images["B2_sharp"]], axis=2)

    # Initially we made 2048x2048 windows because we were confused.
    if rgb.shape[0] == 2048:
        rgb = rgb[512:1536, 512:1536, :]

    # Write 1024x1024.
    rgb_fname = os.path.join(
        target_dir, window_id, "layers", "1024", "R_G_B", "image.png"
    )
    os.makedirs(os.path.dirname(rgb_fname), exist_ok=True)
    Image.fromarray(rgb).save(rgb_fname)

    # Write 256x256.
    rgb_256 = rgb[384:640, 384:640, :]
    rgb_256_fname = os.path.join(
        target_dir, window_id, "layers", "256", "R_G_B", "image.png"
    )
    os.makedirs(os.path.dirname(rgb_256_fname), exist_ok=True)
    Image.fromarray(rgb_256).save(rgb_256_fname)


p = multiprocessing.Pool(32)
window_ids = os.listdir(target_dir)
outputs = p.imap_unordered(process, window_ids)
for _ in tqdm.tqdm(outputs, total=len(window_ids)):
    pass
p.close()
