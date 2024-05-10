"""
Create tar files corresponding to 40 m/pixel tiles combining all the 10 m/pixel, 20
m/pixel, and 40 m/pixel Sentinel-2 images into the tar file.
The Sentinel-2 images are split into 1.25 m/pixel files that contain multiple images
(up to 32).
Then upload the tar files to Hugging Face.
"""
import csv
from datetime import date, timedelta
import glob
import io
import multiprocessing
import os
import random
import tarfile

import affine
import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
import tqdm

input_dirs = [
    "/mnt/sentinel1_1/tiles/sentinel1",
    "/mnt/sentinel1_2/tiles/sentinel1",
]
num_workers = 64
naip_csv_fname = "/mnt/hfupload_sentinel1/naip.csv"
tar_dir = "/mnt/hfupload_sentinel1/sentinel1_tar/"
min_images = 2
max_images = 8

def get_yearmo_offset(yearmo, offset):
    d = date(int(yearmo[0:4]), int(yearmo[4:6]), 15)
    sign = 1
    if offset < 0:
        offset = -offset
        sign = -1
    for _ in range(offset):
        d += sign * timedelta(days=30)
        d = d.replace(day=15)
    return d.strftime("%Y%m")

# Figure out which 1.25 m/pixel tiles, and which year/month for each tile, are needed.
# Group these by big tile (40 m/pixel).
needed_tiles = {}
with open(naip_csv_fname) as f:
    reader = csv.DictReader(f)
    for row in tqdm.tqdm(reader, desc="Reading CSV"):
        small_tile = (row["projection"], int(row["col"]), int(row["row"]))
        big_tile = (small_tile[0], small_tile[1]//32, small_tile[2]//32)
        parts = row["naip_scene"].split("_")
        # Should be parts[5] but we messed up in 3_sentinel2_windows
        # (using processing time instead of sense time).
        assert len(parts[-1]) == 8
        yearmo = parts[-1][0:6]
        if big_tile not in needed_tiles:
            needed_tiles[big_tile] = []
        needed_tiles[big_tile].append((small_tile, yearmo))

# Identify available Sentinel-1 images.
def get_fnames(image_dir):
    fnames = glob.glob(os.path.join(image_dir, "VV_VH/*/*.tif"))
    cur_images = {}
    for fname in fnames:
        parts = fname.split("/")
        projection = parts[-2].split("_")[0].split(":")[1]
        tile_parts = parts[-1].split(".")[0].split("_")
        col = int(tile_parts[0])
        row = int(tile_parts[1])
        tile = (projection, col, row)

        yearmo = parts[-4].split("_")[4][0:6]

        if (tile, yearmo) not in cur_images:
            cur_images[(tile, yearmo)] = []
        cur_images[(tile, yearmo)].append(fname)
    return cur_images

jobs = []
for input_dir in input_dirs:
    for image_name in os.listdir(input_dir):
        jobs.append(os.path.join(input_dir, image_name))
sentinel1_images = {}
p = multiprocessing.Pool(num_workers)
random.shuffle(jobs)
outputs = p.imap_unordered(get_fnames, jobs)
for cur_images in tqdm.tqdm(outputs, total=len(jobs), desc="Get Sentinel-1 filenames"):
    for k, v in cur_images.items():
        if k not in sentinel1_images:
            sentinel1_images[k] = []
        sentinel1_images[k].extend(v)
p.close()

def process(job):
    big_tile, small_tiles = job

    tar_fname = os.path.join(tar_dir, f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.tar")
    csv_fname = os.path.join(tar_dir, f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.csv")

    if os.path.exists(tar_fname) and os.path.exists(csv_fname):
        return

    metadata = []

    with tarfile.open(tar_fname+".tmp", "w") as tar_file:
        # Maintain an image cache from fname -> image array.
        # Since we'll be reusing large 10 m/pixel images for lots of 1.25 m/pixel tiles.
        image_cache = {}

        def get_image(fname):
            def load_image(fname):
                if not os.path.exists(fname):
                    return None
                try:
                    with rasterio.open(fname) as src:
                        data = src.read()
                        return data.astype(np.float32)
                except Exception as e:
                    print(f"warning: got error reading {fname}: {e}")
                    return None
            if fname not in image_cache:
                image_cache[fname] = load_image(fname)
            return image_cache[fname]

        for small_tile, yearmo in small_tiles:
            # Determine which Sentinel-2 scenes to use at this 1.25 m/pixel tile.
            # We only use images with no missing pixels. And then sample randomly if it exceeds max_images.
            band_res = 8
            band_tile = (small_tile[0], small_tile[1]//band_res, small_tile[2]//band_res)
            crop_size = 512 // band_res
            crop_start = (
                small_tile[1] - band_tile[1]*band_res,
                small_tile[2] - band_tile[2]*band_res,
            )

            candidate_images = []
            for offset in [-2, -1, 0, 1, 2]:
                cur_yearmo = get_yearmo_offset(yearmo, offset)
                for fname in sentinel1_images.get((band_tile, cur_yearmo), []):
                    whole_image = get_image(fname)
                    cur_image = whole_image[:, crop_start[1]*crop_size:(crop_start[1]+1)*crop_size, crop_start[0]*crop_size:(crop_start[0]+1)*crop_size]
                    missing_pixels = np.count_nonzero(np.isnan(cur_image[0, :, :]))
                    if missing_pixels > 0:
                        continue
                    potentially_missing_pixels = np.count_nonzero(cur_image[0, :, :] == 0)
                    if potentially_missing_pixels > crop_size:
                        continue
                    candidate_images.append((cur_image, fname))

            if len(candidate_images) < min_images:
                continue
            random.shuffle(candidate_images)
            candidate_images = candidate_images[0:max_images]

            for index, (image, fname) in enumerate(candidate_images):
                sentinel1_scene = fname.split("/")[-4]
                metadata.append({
                    "projection": small_tile[0],
                    "col": small_tile[1],
                    "row": small_tile[2],
                    "index": index,
                    "scene": sentinel1_scene,
                })

            data = np.concatenate([t[0] for t in candidate_images], axis=0)

            crs = CRS.from_epsg(int(small_tile[0]))
            transform = affine.Affine(
                1.25 * band_res,
                0,
                small_tile[1] * crop_size * 1.25 * band_res,
                0,
                -1.25 * band_res,
                small_tile[2] * crop_size * -1.25 * band_res,
            )
            profile = {
                "driver": "GTiff",
                "compress": "lzw",
                "width": data.shape[2],
                "height": data.shape[1],
                "count": data.shape[0],
                "dtype": data.dtype.name,
                "crs": crs,
                "transform": transform,
            }
            buf = io.BytesIO()
            with rasterio.open(buf, "w", **profile) as dst:
                dst.write(data)
            out_entry = tarfile.TarInfo(name=f"sentinel1/{small_tile[0]}_{small_tile[1]}_{small_tile[2]}.tif")
            out_entry.size = buf.getbuffer().nbytes
            out_entry.mode = 0o644
            buf.seek(0)
            tar_file.addfile(out_entry, fileobj=buf)

    with open(csv_fname+".tmp", "w") as f:
        writer = csv.DictWriter(f, ["projection", "col", "row", "index", "scene"])
        writer.writeheader()
        writer.writerows(metadata)

    os.rename(tar_fname+".tmp", tar_fname)
    os.rename(csv_fname+".tmp", csv_fname)

jobs = list(needed_tiles.items())
random.shuffle(jobs)
p = multiprocessing.Pool(num_workers)
outputs = p.imap_unordered(process, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Process"):
    pass
p.close()
