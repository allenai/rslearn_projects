"""Quick script to visualize the wind turbine images and labels."""

import json
import os

import numpy as np
import rasterio
import skimage.io
import tqdm

window_names = os.listdir("in/")
out_dir = "out/"

for window_name in tqdm.tqdm(window_names):
    label_fname = os.path.join("in", window_name, "layers", "label", "data.geojson")
    img_fname = os.path.join(
        "in", window_name, "layers", "sentinel2", "B02_B03_B04_B08", "geotiff.tif"
    )

    if not os.path.exists(label_fname) or not os.path.exists(img_fname):
        continue

    with open(os.path.join("in", window_name, "metadata.json")) as f:
        metadata = json.load(f)
    bounds = metadata["bounds"]
    window_width = bounds[2] - bounds[0]
    window_height = bounds[3] - bounds[1]

    label_im = np.zeros((window_height, window_width), dtype=np.uint8)
    with open(label_fname) as f:
        for feat in json.load(f)["features"]:
            col, row = feat["geometry"]["coordinates"]
            rect = (
                int(col) - bounds[0] - 10,
                int(row) - bounds[1] - 10,
                int(col) - bounds[0] + 10,
                int(row) - bounds[1] + 10,
            )
            rect = (
                np.clip(rect[0], 0, window_width),
                np.clip(rect[1], 0, window_height),
                np.clip(rect[2], 0, window_width),
                np.clip(rect[3], 0, window_height),
            )
            label_im[rect[1] : rect[3], rect[0] : rect[2]] = 255
    skimage.io.imsave(
        os.path.join(out_dir, f"{window_name}_label.png"),
        label_im,
        check_contrast=False,
    )

    with rasterio.open(img_fname) as src:
        data = src.read()
    vis_im = np.clip(data[(2, 1, 0), :, :].transpose(1, 2, 0) // 10, 0, 255).astype(
        np.uint8
    )
    skimage.io.imsave(
        os.path.join(out_dir, f"{window_name}_vis.png"), vis_im, check_contrast=False
    )
