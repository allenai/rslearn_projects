"""Create tar files corresponding to 40 m/pixel tiles combining all the 10 m/pixel and 20
m/pixel Landsat images into the tar file.
The Landsat images are split into 1.25 m/pixel files that contain multiple images
(up to 32).
Then upload the tar files to Hugging Face.
"""

import csv
import glob
import io
import multiprocessing
import os
import random
import tarfile
from datetime import date, timedelta

import affine
import numpy as np
import rasterio
import tqdm
from PIL import Image
from rasterio.crs import CRS

input_dirs = [
    "/mnt/landsat_1/tiles/landsat",
]
num_workers = 64
naip_csv_fname = "/home/ubuntu/naip.csv"
tar_dir = "/mnt/hfupload_landsat/landsat_tar/"
bands = {
    8: ["B8"],
    16: ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"],
}
resolution_strs = {
    8: "10_-10",
    16: "20.0_-20.0",
}
min_images = 4
max_images = 32


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
        big_tile = (small_tile[0], small_tile[1] // 32, small_tile[2] // 32)
        parts = row["naip_scene"].split("_")
        # Should be parts[5] but we messed up in 3_sentinel2_windows
        # (using processing time instead of sense time).
        assert len(parts[-1]) == 8
        yearmo = parts[-1][0:6]
        if big_tile not in needed_tiles:
            needed_tiles[big_tile] = []
        needed_tiles[big_tile].append((small_tile, yearmo))


# Identify available Sentinel-2 images.
def get_fnames(image_dir):
    fnames = glob.glob(os.path.join(image_dir, "*/*/*.tif"))
    cur_images = {}
    for fname in fnames:
        parts = fname.split("/")
        projection = parts[-2].split("_")[0].split(":")[1]
        tile_parts = parts[-1].split(".")[0].split("_")
        col = int(tile_parts[0])
        row = int(tile_parts[1])
        tile = (projection, col, row)

        band = parts[-3]
        yearmo = parts[-4].split("_")[3][0:6]

        if (band, tile, yearmo) not in cur_images:
            cur_images[(band, tile, yearmo)] = []
        cur_images[(band, tile, yearmo)].append(fname)
    return cur_images


jobs = []
for input_dir in input_dirs:
    for image_name in os.listdir(input_dir):
        jobs.append(os.path.join(input_dir, image_name))
landsat_images = {}
p = multiprocessing.Pool(num_workers)
random.shuffle(jobs)
outputs = p.imap_unordered(get_fnames, jobs)
for cur_images in tqdm.tqdm(outputs, total=len(jobs), desc="Get Landsat filenames"):
    for k, v in cur_images.items():
        if k not in landsat_images:
            landsat_images[k] = []
        landsat_images[k].extend(v)
p.close()


def process(job):
    big_tile, small_tiles = job

    tar_fname = os.path.join(tar_dir, f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.tar")
    csv_fname = os.path.join(tar_dir, f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.csv")

    if os.path.exists(tar_fname) and os.path.exists(csv_fname):
        return

    metadata = []

    with tarfile.open(tar_fname + ".tmp", "w") as tar_file:
        # Maintain an image cache from fname -> image array.
        # Since we'll be reusing large 10+ m/pixel images for lots of 1.25 m/pixel tiles.
        image_cache = {}

        def get_image(fname):
            def load_image(fname):
                if not os.path.exists(fname):
                    return None
                try:
                    return np.array(Image.open(fname))
                except Exception:
                    return None

            if fname not in image_cache:
                image_cache[fname] = load_image(fname)
            return image_cache[fname]

        for small_tile, yearmo in small_tiles:
            band_res = 16
            band_tile = (
                small_tile[0],
                small_tile[1] // band_res,
                small_tile[2] // band_res,
            )
            crop_size = 512 // band_res
            crop_start = (
                small_tile[1] - band_tile[1] * band_res,
                small_tile[2] - band_tile[2] * band_res,
            )

            candidate_fnames = []
            for offset in [-2, -1, 0, 1, 2]:
                cur_yearmo = get_yearmo_offset(yearmo, offset)
                for fname in landsat_images.get(("B1", band_tile, cur_yearmo), []):
                    failed = False
                    for band in bands[band_res]:
                        cur_fname = fname.replace("B1", f"/{band}/")
                        cur_image = get_image(cur_fname)
                        if cur_image is None:
                            failed = True
                            break
                        crop = cur_image[
                            crop_start[1] * crop_size : (crop_start[1] + 1) * crop_size,
                            crop_start[0] * crop_size : (crop_start[0] + 1) * crop_size,
                        ]
                        if np.count_nonzero(crop == 0) > crop_size:
                            failed = True
                            break
                    if failed:
                        continue
                    candidate_fnames.append(fname)

            if len(candidate_fnames) < min_images:
                continue
            random.shuffle(candidate_fnames)
            fnames = candidate_fnames[0:max_images]

            for index, fname in enumerate(fnames):
                landsat_scene = fname.split("/")[-4]
                metadata.append(
                    {
                        "projection": small_tile[0],
                        "col": small_tile[1],
                        "row": small_tile[2],
                        "index": index,
                        "scene": landsat_scene,
                    }
                )

            # Now use those selected image filenames to create output tif at each resolution.
            for band_res, band_names in bands.items():
                band_tile = (
                    small_tile[0],
                    small_tile[1] // band_res,
                    small_tile[2] // band_res,
                )
                crop_size = 512 // band_res
                crop_start = (
                    small_tile[1] - band_tile[1] * band_res,
                    small_tile[2] - band_tile[2] * band_res,
                )

                data = []
                for fname in fnames:
                    parts = fname.split("/")
                    # Replace resolution with current band_res.
                    parts[-2] = parts[-2].replace(
                        "20.0_-20.0", resolution_strs[band_res]
                    )
                    # Replace the tile part too.
                    parts[-1] = f"{band_tile[1]}_{band_tile[2]}.tif"
                    fname = "/".join(parts)

                    for band in band_names:
                        cur_fname = fname.replace("B1", f"/{band}/")
                        cur_image = get_image(cur_fname)
                        if cur_image is None:
                            data.append(
                                np.zeros((crop_size, crop_size), dtype=np.uint16)
                            )
                            continue
                        crop = cur_image[
                            crop_start[1] * crop_size : (crop_start[1] + 1) * crop_size,
                            crop_start[0] * crop_size : (crop_start[0] + 1) * crop_size,
                        ]
                        data.append(crop)
                data = np.stack(data, axis=0)

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
                out_entry = tarfile.TarInfo(
                    name=f"landsat/{small_tile[0]}_{small_tile[1]}_{small_tile[2]}_{band_res}.tif"
                )
                out_entry.size = buf.getbuffer().nbytes
                out_entry.mode = 0o644
                buf.seek(0)
                tar_file.addfile(out_entry, fileobj=buf)

    with open(csv_fname + ".tmp", "w") as f:
        writer = csv.DictWriter(f, ["projection", "col", "row", "index", "scene"])
        writer.writeheader()
        writer.writerows(metadata)

    os.rename(tar_fname + ".tmp", tar_fname)
    os.rename(csv_fname + ".tmp", csv_fname)


jobs = list(needed_tiles.items())
random.shuffle(jobs)
p = multiprocessing.Pool(num_workers)
outputs = p.imap_unordered(process, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Process"):
    pass
p.close()
