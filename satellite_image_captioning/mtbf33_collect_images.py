"""
After rslearn dataset is materialized, use this script to put all the images in one
folder while also including JSON specifying the longitude, latitude, and Google Maps
URL of the center of the image.
"""

import glob
import json
import os
import shutil

import numpy as np
from PIL import Image
import rasterio.features
import shapely.geometry

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry

in_dir = "/data/favyenb/datasets_for_image_caption/rslearn_mtbf33/windows/"
out_dir = "/data/favyenb/datasets_for_image_caption/rslearn_mtbf33/images/"

for group in os.listdir(in_dir):
    for window_name in os.listdir(os.path.join(in_dir, group)):
        out_label = "_".join(window_name.split("_")[0:4])
        window_root = os.path.join(in_dir, group, window_name)
        window = Window.load(window_root)
        src_geom = window.get_geometry()
        dst_geom = src_geom.to_projection(WGS84_PROJECTION)
        lon = dst_geom.shp.centroid.x
        lat = dst_geom.shp.centroid.y

        image_fnames = glob.glob(os.path.join(window_root, "layers/*/R_G_B/image.png"))
        for image_fname in image_fnames:
            year = image_fname.split("/")[-3]
            shutil.copyfile(image_fname, os.path.join(out_dir, f"{out_label}_{year}.png"))

        with open(os.path.join(out_dir, f"{out_label}.json"), "w") as f:
            json.dump({
                "longitude": lon,
                "latitude": lat,
                "google_url": f"https://www.google.com/maps/search/?api=1&query={lat},{lon}",
            }, f)

        mask = np.zeros((window.bounds[3] - window.bounds[1], window.bounds[2] - window.bounds[0]), dtype=np.uint8)
        if os.path.exists(os.path.join(window_root, "layers/vector_mask/data.geojson")):
            with open(os.path.join(window_root, "layers/vector_mask/data.geojson")) as f:
                fc = json.load(f)
            shapes = []
            for feat in fc["features"]:
                shp = shapely.geometry.shape(feat["geometry"])
                offset = np.array([window.bounds[0], window.bounds[1]])
                shp = shapely.transform(shp, lambda array: array - offset)
                if not shp.is_valid or shp.is_empty:
                    continue
                shapes.append((shp, 255))

            if len(shapes) > 0:
                mask = rasterio.features.rasterize(
                    shapes=shapes,
                    out_shape=(window.bounds[3] - window.bounds[1], window.bounds[2] - window.bounds[0]),
                    dtype=np.uint8,
                )

        Image.fromarray(mask).save(os.path.join(out_dir, f"{out_label}_mask.png"))
