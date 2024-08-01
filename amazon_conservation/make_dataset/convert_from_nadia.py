"""This is based on 2024_03_27_recompute_masks in multisat amazon_conservation code.

This script will do a similar recompute process but instead of updating the mask,
the output is rslearn windows with the info.json and mask.png similar to
create_unlabeled_dataset.json.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

import fiona
import rasterio
import rasterio.features
import shapely.geometry
import tqdm
from PIL import Image
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry

tasks = {
    "nadia2": {
        "shp_fname": "/multisat/datasets/amazon_conservation/2023-12-06-nadia/polygon/2_Training-data_2021-2022_GLAD-S2_Polygons_WGS84.shp",
    },
    "nadia3": {
        "shp_fname": "/multisat/datasets/amazon_conservation/2024-02-01-nadia/3_Training-data_2021-2022_GLAD-S2_Polygons_WGS84_v2.shp",
    },
}
group = "nadia2"
task = tasks[group]
shp_fname = task["shp_fname"]

crop_size = 128
out_dir = "/data/favyenb/rslearn_amazon_conservation_closetime/"

web_mercator_crs = CRS.from_epsg(3857)
web_mercator_m = 2 * math.pi * 6378137
web_mercator_total_pixels = 2**13 * 512
pixel_size = web_mercator_m / web_mercator_total_pixels
web_mercator_projection = Projection(web_mercator_crs, pixel_size, -pixel_size)

categories = [
    "unknown",
    "mining",
    "agriculture-small",
    "agriculture-mennonite",
    "agriculture-rice",
    "coca",
    "airstrip",
    "road",
    "logging",
    "burned",
    "landslide",
    "hurricane",
    "river",
]

with fiona.open(shp_fname) as src:
    for feat_idx, feat in enumerate(tqdm.tqdm(src)):
        date_parts = feat.properties["Date"].split("-")
        ts = datetime(
            int(date_parts[0]),
            int(date_parts[1]),
            int(date_parts[2]),
            tzinfo=timezone.utc,
        )
        category = categories[feat.properties["Level_2"]]

        shp = shapely.geometry.shape(feat.geometry)
        geom = STGeometry(WGS84_PROJECTION, shp, None)
        geom = geom.to_projection(web_mercator_projection)
        center = geom.shp.centroid
        rslearn_center = (int(center.x), int(center.y))
        multisat_center = (
            rslearn_center[0] + web_mercator_total_pixels // 2,
            rslearn_center[1] + web_mercator_total_pixels // 2,
        )

        window_name = f"feat_{feat_idx}_{multisat_center[0]}_{multisat_center[1]}"

        bounds = [
            rslearn_center[0] - crop_size // 2,
            rslearn_center[1] - crop_size // 2,
            rslearn_center[0] + crop_size // 2,
            rslearn_center[1] + crop_size // 2,
        ]
        time_range = (
            ts,
            ts + timedelta(days=30),
        )
        window = Window(
            window_root=os.path.join(out_dir, "windows", group, window_name),
            group=group,
            name=window_name,
            projection=web_mercator_projection,
            bounds=bounds,
            time_range=time_range,
        )
        window.save()

        # Create mask.png.
        def to_out_pixel(points):
            points[:, 0] -= bounds[0]
            points[:, 1] -= bounds[1]
            return points

        pixel_shp = shapely.transform(geom.shp, to_out_pixel)
        mask_im = rasterio.features.rasterize(
            [(pixel_shp, 255)],
            out_shape=(crop_size, crop_size),
        )
        Image.fromarray(mask_im).save(os.path.join(window.window_root, "mask.png"))

        # Create label.json.
        with open(os.path.join(window.window_root, "label.json"), "w") as f:
            json.dump(
                {
                    "old_label": category,
                    "new_label": category,
                },
                f,
            )
