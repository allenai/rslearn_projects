"""
This is a quick script to visualize the outputs from create_windows.py
(after running prepare/ingest/materialize in rslearn to get the images).

Just visualizing the ground truth data.
- Heading, length in the image.
- Ship type in the filename.
"""

import json
import math
import os

import numpy as np
from PIL import Image
import tqdm

mask_size = 256
pixel_size = 2.5

example_ids = os.listdir(".")
for example_id in tqdm.tqdm(example_ids):
    im_fname = os.path.join(example_id, "layers", "sentinel2", "R_G_B", "image.png")
    if not os.path.exists(im_fname):
        continue
    im = np.array(Image.open(im_fname))

    factor = mask_size // im.shape[0]
    im = im.repeat(axis=0, repeats=factor).repeat(axis=1, repeats=factor)

    with open(os.path.join(example_id, "info.json")) as f:
        info = json.load(f)

    # Create vector from length and cog angle.
    radians = math.radians(info["cog"] - 90)
    vector = (
        int(math.cos(radians) * info["length"] / pixel_size / 2),
        int(math.sin(radians) * info["length"] / pixel_size / 2),
    )

    #mask = np.zeros((mask_size, mask_size, 3), dtype=np.uint8)
    center = mask_size // 2
    front = (
        center + vector[0],
        center + vector[1],
    )
    back = (
        center - vector[0],
        center - vector[1],
    )
    im[front[1]-2:front[1]+2, front[0]-2:front[0]+2, :] = [255, 0, 0]
    im[back[1]-2:back[1]+2, back[0]-2:back[0]+2, :] = [255, 255, 0]
    im[center-2:center+2, center-2:center+2, :] = [255, 255, 255]

    Image.fromarray(im).save(f"out/{example_id}_im_{info["type"]}.png")
    #Image.fromarray(mask).save(f"out/{example_id}_mask.png")
