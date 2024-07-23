"""
Create rslearn data from the GCS dataset retrieved using retrieve_dataset.py.
Two groups of windows are created:
- images: windows of the full original GeoTIFF images
- crops: windows corresponding to image patches that are actually annotated
"""

import argparse
from datetime import datetime, timedelta
import json
import os
import shutil

import numpy as np
import rasterio
import rasterio.features
import shapely

from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import LocalFileAPI, Projection, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat

CATEGORIES = [
    "unknown",
    "F_2_2_SMALL_PERMANENT_FRESHWATER_LAKES",
    "MT_1_3_SANDY_SHORELINES",
    "MT_2_1_COASTAL_SHRUBLANDS_AND_GRASSLANDS",
    "T_7_1_ANNUAL_CROPLANDS",
    "T_7_4_URBAN_AND_INDUSTRIAL_ECOSYSTEMS",
]

COLORS = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [0, 255, 255],
    [255, 0, 255],
    [0, 128, 0],
    [255, 160, 122],
    [139, 69, 19],
    [128, 128, 128],
    [255, 255, 255],
    [143, 188, 143],
    [95, 158, 160],
    [255, 200, 0],
    [128, 0, 0],
]

class ProcessJob:
    def __init__(self, prefix, out_dir):
        self.prefix = prefix
        self.out_dir = out_dir

def process(job):
    label = os.path.basename(job.prefix)
    raster = rasterio.open(job.prefix + ".tif")
    projection = Projection(raster.crs, raster.transform.a, raster.transform.e)
    start_col = round(raster.transform.c / raster.transform.a)
    start_row = round(raster.transform.f / raster.transform.e)
    raster_bounds = [
        start_col,
        start_row,
        start_col + raster.width,
        start_row + raster.height,
    ]

    # Extract datetime.
    parts = job.prefix.split("_")[-1].split("-")
    assert len(parts) == 5
    ts = datetime(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))

    # First create window for the entire GeoTIFF.
    window_root = os.path.join(job.out_dir, "windows", "images", label)
    os.makedirs(window_root, exist_ok=True)
    window = Window(
        file_api=LocalFileAPI(window_root),
        group="images",
        name=label,
        projection=projection,
        bounds=raster_bounds,
        time_range=(ts - timedelta(minutes=1), ts + timedelta(minutes=1)),
    )
    window.save()
    dst_geotiff_fname = os.path.join(window_root, "layers", "maxar", "R_G_B", "geotiff.tif")
    os.makedirs(os.path.dirname(dst_geotiff_fname), exist_ok=True)
    shutil.copyfile(job.prefix + ".tif", dst_geotiff_fname)
    with open(os.path.join(window_root, "layers", "maxar", "completed"), "w") as f:
        pass

    # Second create windows for the annotated patches.
    # We start by extracting the bounds of each patch.
    # To do so, we cluster the annotations agglomeratively.
    clusters = {}
    with open(job.prefix + "_labels.json") as f:
        for idx, annot in enumerate(json.load(f)["annotations"]):
            assert len(annot["categories"]) == 1
            category_id = CATEGORIES.index(annot["categories"][0]["name"])
            bounding_poly = annot["boundingPoly"]
            exterior = [(vertex["x"], vertex["y"]) for vertex in bounding_poly[0]["normalizedVertices"]]
            interiors = []
            for poly in bounding_poly[1:]:
                interior = [(vertex["x"], vertex["y"]) for vertex in poly["normalizedVertices"]]
                interiors.append(interior)
            shp = shapely.Polygon(exterior, interiors)
            src_geom = STGeometry(WGS84_PROJECTION, shp, None)
            dst_geom = src_geom.to_projection(projection)
            clusters[idx] = (
                dst_geom.shp,
                [(dst_geom.shp, category_id)],
            )
    distance_threshold = 5
    while True:
        # Merge up to one pair of clusters per iteration.
        # If there's nothing to merge then that means we're done and can quit.
        did_merge = False
        cluster_keys = list(clusters.keys())
        for idx1, k1 in enumerate(cluster_keys):
            for k2 in cluster_keys[idx1+1:]:
                c1 = clusters[k1]
                c2 = clusters[k2]
                if c1[0].distance(c2[0]) > distance_threshold:
                    continue
                # Merge these clusters.
                # We use k1 for the new cluster key and remove k2.
                clusters[k1] = (
                    c1[0].union(c2[0]),
                    c1[1] + c2[1],
                )
                del clusters[k2]
                did_merge = True
                break

            if did_merge:
                break

        if not did_merge:
            break

    # Now we can create separate windows for each patch.
    array = raster.read()
    for bounding_shp, shps in clusters.values():
        proj_bounds = [int(x) for x in bounding_shp.bounds]
        pixel_bounds = [
            proj_bounds[0] - raster_bounds[0],
            proj_bounds[1] - raster_bounds[1],
            proj_bounds[2] - raster_bounds[0],
            proj_bounds[3] - raster_bounds[1],
        ]

        # Create window.
        window_name = f"{label}_{pixel_bounds[0]}_{pixel_bounds[1]}"
        window_root = os.path.join(job.out_dir, "windows", "crops", window_name)
        os.makedirs(window_root, exist_ok=True)
        window = Window(
            file_api=LocalFileAPI(window_root),
            group="crops",
            name=window_name,
            projection=projection,
            bounds=proj_bounds,
            time_range=(ts - timedelta(minutes=1), ts + timedelta(minutes=1)),
        )
        window.save()

        # Write the GeoTIFF.
        crop = array[:, pixel_bounds[1]:pixel_bounds[3], pixel_bounds[0]:pixel_bounds[2]]
        file_api = window.file_api.get_folder("layers", "maxar", "R_G_B")
        GeotiffRasterFormat().encode_raster(file_api, projection, proj_bounds, crop)

        # Render the GeoJSON labels and write that too.
        shapes = []
        for shp, category_id in shps:
            shp = shapely.transform(shp, lambda coords: coords - [proj_bounds[0], proj_bounds[1]])
            shapes.append((shp, category_id))
        mask = rasterio.features.rasterize(shapes, out_shape=(proj_bounds[3] - proj_bounds[1], proj_bounds[2] - proj_bounds[0]))
        file_api = window.file_api.get_folder("layers", "label", "label")
        GeotiffRasterFormat().encode_raster(file_api, projection, proj_bounds, mask[None, :, :])

        # Along with a visualization image.
        label_vis = np.zeros((mask.shape[0], mask.shape[1], 3))
        for category_id in range(len(CATEGORIES)):
            color = COLORS[category_id % len(COLORS)]
            label_vis[mask == category_id] = color
        file_api = window.file_api.get_folder("layers", "label", "vis")
        GeotiffRasterFormat().encode_raster(file_api, projection, proj_bounds, label_vis.transpose(2, 0, 1))

        with open(os.path.join(window_root, "layers", "maxar", "completed"), "w") as f:
            pass
        with open(os.path.join(window_root, "layers", "label", "completed"), "w") as f:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", help="Input directory containing retrieved images and labels")
    parser.add_argument("--out_dir", help="Output directory")
    args = parser.parse_args()

    jobs = []
    for fname in os.listdir(args.in_dir):
        if not fname.endswith(".tif"):
            continue
        prefix = os.path.join(args.in_dir, fname.split(".tif")[0])
        job = ProcessJob(prefix, args.out_dir)
        jobs.append(job)
    for job in jobs:
        process(job)
