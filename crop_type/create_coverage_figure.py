"""
This script is for creating coverage figure showing where the crop type datasets are.
It uses OpenStreetMap for the background.
"""
import math
import numpy as np
import os
import subprocess

from PIL import Image
import rasterio
import skimage.io
import tqdm

zoom = 3
dataset_colors = {
    "agrifieldnet_2021": [240, 128, 128],
    "cdl_2023": [255, 20, 147],
    "eurocrops": [255, 165, 0],
    "nccm_2019": [240, 230, 140],
    "sact_2017": [255, 0, 255],
    "sas_2021": [0, 255, 127],
}
dataset_path_template = "/data/yichiac/sentinel2_subsample/sentinel2_{dataset_name}_subsampled"
padding = 1

# Compute m/pixel of the image.
web_mercator_m = 2 * math.pi * 6378137
num_tiles = 2**zoom
total_pixels = num_tiles*256
meters_per_pixel = web_mercator_m / total_pixels

# Download OpenStreetMap data.
url_template = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
cache_dir = "./openstreetmap/"
print("getting OpenStreetMap image")
im = np.zeros((total_pixels, total_pixels, 3), dtype=np.uint8)
for x in range(num_tiles):
    for y in range(num_tiles):
        print(f"... {x}, {y}")
        url = url_template.format(z=zoom, x=x, y=y)
        local_fname = os.path.join(cache_dir, f"{zoom}_{x}_{y}.png")
        if not os.path.exists(local_fname):
            # Using wget here since kept getting blocked with urllib.request.
            subprocess.check_call(["wget", url, "-O", local_fname + ".tmp"])
            os.rename(local_fname + ".tmp", local_fname)
        # Use skimage.io instead of PIL to handle color map.
        cur_im = skimage.io.imread(local_fname)
        im[y*256:(y+1)*256, x*256:(x+1)*256, :] = cur_im

# Add each dataset.
# All the images should be in WebMercator already.
# So we just need to convert that to the m/pixel of the output image.
for dataset_name, color in dataset_colors.items():
    print(f"adding {dataset_name}")
    dataset_path = dataset_path_template.format(dataset_name=dataset_name)
    for fname in tqdm.tqdm(os.listdir(dataset_path)):
        with rasterio.open(os.path.join(dataset_path, fname)) as src:
            bounds = src.bounds
            center = (
                (bounds[0] + bounds[2]) / 2,
                (bounds[1] + bounds[3]) / 2,
            )
        # Transform to pixels.
        center = (
            int(center[0] / meters_per_pixel) + total_pixels//2,
            int(center[1] / -meters_per_pixel) + total_pixels//2,
        )
        im[center[1]-padding:center[1]+padding+1, center[0]-padding:center[0]+padding+1, :] = color

Image.fromarray(im).save("out.png")
