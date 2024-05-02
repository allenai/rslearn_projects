"""
Name it so it's a bit more consistent with the data previously sent to Piper.
"""
import glob
import numpy as np
import os

import rasterio
import tqdm

in_dir = "/data/favyenb/rslearn_superres_non_us/windows/default/"
out_dir = "/data/favyenb/rslearn_superres_non_us/renamed/"

bands = [
    ("B02_B03_B04_B08", "_8.tif"),
    ("B05_B06_B07_B11_B12_B8A", "_16.tif"),
    ("B01_B09_B10", "_32.tif"),
]

for window_name in tqdm.tqdm(os.listdir(in_dir)):
    parts = window_name.split("_")
    epsg_id = parts[0].split(":")[1]
    out_name = f"{epsg_id}_{parts[1]}_{parts[2]}"

    options = glob.glob(os.path.join(in_dir, window_name, f"layers/*/{bands[0][0]}/geotiff.tif"))

    for band, suffix in bands:
        profile = None
        images = []
        for option in options:
            fname = option.replace(bands[0][0], band)
            with rasterio.open(fname) as src:
                images.append(src.read())
                if profile is None:
                    profile = src.profile

        out_fname = os.path.join(out_dir, out_name + suffix)
        images = np.concatenate(images, axis=0)
        profile["count"] = images.shape[0]
        with rasterio.open(out_fname, "w", **profile) as dst:
            dst.write(images)
