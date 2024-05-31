import json
import os
import sys

import numpy as np
from PIL import Image
import rasterio
import rasterio.features

label = sys.argv[1]
root_dir = sys.argv[2]
out_dir = sys.argv[3]

# Landsat.
with rasterio.open(os.path.join(root_dir, "landsat", f"{label}_8.tif")) as src:
    array = src.read()
for idx in range(16):
    img = np.clip((array[idx, :, :].astype(np.float64) - 5000) / 20, 0, 255).astype(np.uint8)
    img = img.repeat(axis=0, repeats=8).repeat(axis=1, repeats=8)
    Image.fromarray(img).save(os.path.join(out_dir, f"{label}_landsat{idx}.png"))

# NAIP.
array = np.array(Image.open(os.path.join(root_dir, "naip", f"{label}.png")))
Image.fromarray(array[:, :, 0:3]).save(os.path.join(out_dir, f"{label}_naip.png"))

# Old NAIP.
array = np.array(Image.open(os.path.join(root_dir, "oldnaip", f"{label}.png")))
Image.fromarray(array[:, :, 0:3]).save(os.path.join(out_dir, f"{label}_oldnaip.png"))

# OpenStreetMap.
with open(os.path.join(root_dir, "openstreetmap", f"{label}.geojson")) as f:
    data = json.load(f)
category_colors = {
    "river": [0, 0, 255],
    "road": [255, 255, 255],
    "building": [255, 255, 0],
    "parking": [255, 0, 0],
    "leisure_park": [144, 238, 144],
    "solar": [128, 128, 128],
}
category_selectors = {
    "leisure_park": lambda feat: feat["properties"]["category"] == "leisure" and feat["properties"].get("leisure") == "park",
    "solar": lambda feat: feat["properties"]["category"] == "power_plant" and feat["properties"].get("plant:source") == "solar",
}
img = np.zeros((512, 512, 3), dtype=np.uint8)
for category, color in category_colors.items():
    selector = category_selectors.get(category, lambda feat: feat["properties"]["category"] == category)
    geometries = [feat["geometry"] for feat in data["features"] if selector(feat)]
    if len(geometries) == 0:
        continue
    mask = rasterio.features.rasterize(geometries, out_shape=(512, 512))
    img[mask > 0] = color
Image.fromarray(img).save(os.path.join(out_dir, f"{label}_openstreetmap.png"))

# Sentinel-1.
with rasterio.open(os.path.join(root_dir, "sentinel1", f"{label}.tif")) as src:
    array = src.read()
img = np.clip((array[0, :, :] + 20) * 10, 0, 255).astype(np.uint8)
img = img.repeat(axis=0, repeats=8).repeat(axis=1, repeats=8)
Image.fromarray(img).save(os.path.join(out_dir, f"{label}_sentinel1.png"))

# Sentinel-2.
with rasterio.open(os.path.join(root_dir, "sentinel2", f"{label}_8.tif")) as src:
    array = src.read()
for idx in range(8):
    img = np.clip(array[(idx*4+2, idx*4+1, idx*4+0), :, :].transpose(1, 2, 0) / 10, 0, 255).astype(np.uint8)
    img = img.repeat(axis=0, repeats=8).repeat(axis=1, repeats=8)
    Image.fromarray(img).save(os.path.join(out_dir, f"{label}_sentinel2_{idx}.png"))

# WorldCover.
array = np.array(Image.open(os.path.join(root_dir, "worldcover", f"{label}.png")))
img = np.zeros((512, 512, 3), dtype=np.uint8)
category_colors = {
    10: [0, 100, 0],
    20: [255, 187, 34],
    30: [255, 255, 76],
    40: [240, 150, 255],
    50: [250, 0, 0],
    60: [180, 180, 180],
    70: [240, 240, 240],
    80: [0, 100, 200],
    90: [0, 150, 160],
    95: [0, 207, 117],
    100: [250, 230, 160],
}
for category, color in category_colors.items():
    mask = (array == category).repeat(axis=0, repeats=8).repeat(axis=1, repeats=8)
    img[mask] = color
Image.fromarray(img).save(os.path.join(out_dir, f"{label}_worldcover.png"))
