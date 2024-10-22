"""Create tar files corresponding to 40 m/pixel tiles combining all the 10 m/pixel, 20
m/pixel, and 40 m/pixel Sentinel-2 images into the tar file.
The Sentinel-2 images are split into 1.25 m/pixel files that contain multiple images
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
    "/mnt/sentinel2_2019_1/tiles/sentinel2",
    "/mnt/sentinel2_2019_2/tiles/sentinel2",
    "/mnt/sentinel2_2020_1/tiles/sentinel2",
    "/mnt/sentinel2_2020_2/tiles/sentinel2",
    "/mnt/sentinel2_2021_1/tiles/sentinel2",
    "/mnt/sentinel2_2021_2/tiles/sentinel2",
]
num_workers = 64
naip_csv_fname = "/mnt/hfupload_naip/naip.csv"
tar_dir = "/mnt/hfupload_sentinel2/sentinel2_tar/"
bands = {
    8: ["B02", "B03", "B04", "B08"],
    16: ["B05", "B06", "B07", "B11", "B12", "B8A"],
    32: ["B01", "B09", "B10"],
}
resolution_strs = {
    8: "10_-10",
    16: "20.0_-20.0",
    32: "40.0_-40.0",
}
min_images = 16
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
        yearmo = parts[-4].split("_")[2][0:6]

        if (band, tile, yearmo) not in cur_images:
            cur_images[(band, tile, yearmo)] = []
        cur_images[(band, tile, yearmo)].append(fname)
    return cur_images


jobs = []
for input_dir in input_dirs:
    for image_name in os.listdir(input_dir):
        jobs.append(os.path.join(input_dir, image_name))
sentinel2_images = {}
p = multiprocessing.Pool(num_workers)
random.shuffle(jobs)
outputs = p.imap_unordered(get_fnames, jobs)
for cur_images in tqdm.tqdm(outputs, total=len(jobs), desc="Get Sentinel-2 filenames"):
    for k, v in cur_images.items():
        if k not in sentinel2_images:
            sentinel2_images[k] = []
        sentinel2_images[k].extend(v)
p.close()


def process(job):
    big_tile, small_tiles = job

    tar_fname = os.path.join(tar_dir, f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.tar")
    csv_fname = os.path.join(tar_dir, f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.csv")
    tar_fname0 = os.path.join(
        "/mnt/data/sentinel2_tar/", f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.tar"
    )
    csv_fname0 = os.path.join(
        "/mnt/data/sentinel2_tar/", f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.csv"
    )

    if os.path.exists(tar_fname) and os.path.exists(csv_fname):
        return
    if os.path.exists(tar_fname0) and os.path.exists(csv_fname0):
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
            # Use the RGB bands to determine which Sentinel-2 scenes to use at this 1.25 m/pixel tile.
            # We only use images with no missing pixels, and then pick images with minimal estimated cloud cover.
            band_res = 8
            rgb_bands = ["B02", "B03", "B04"]
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
                for fname in sentinel2_images.get(
                    (rgb_bands[0], band_tile, cur_yearmo), []
                ):
                    image = []
                    failed = False
                    for band in rgb_bands:
                        cur_fname = fname.replace(f"/{rgb_bands[0]}/", f"/{band}/")
                        cur_image = get_image(cur_fname)
                        if cur_image is None:
                            failed = True
                            break
                        crop = cur_image[
                            crop_start[1] * crop_size : (crop_start[1] + 1) * crop_size,
                            crop_start[0] * crop_size : (crop_start[0] + 1) * crop_size,
                        ]
                        image.append(crop)
                    if failed:
                        continue
                    image = np.stack(image, axis=2)
                    missing_pixels = np.count_nonzero(image.max(axis=2) == 0)
                    cloudy_pixels = np.count_nonzero(image.min(axis=2) >= 1500)

                    if missing_pixels > crop_size:
                        continue
                    candidate_fnames.append((fname, missing_pixels * 5 + cloudy_pixels))

            if len(candidate_fnames) < min_images:
                continue
            candidate_fnames.sort(key=lambda t: t[1])
            fnames = [fname for fname, _ in candidate_fnames[0:max_images]]

            for index, fname in enumerate(fnames):
                sentinel2_scene = fname.split("/")[-4]
                metadata.append(
                    {
                        "projection": small_tile[0],
                        "col": small_tile[1],
                        "row": small_tile[2],
                        "index": index,
                        "scene": sentinel2_scene,
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
                    parts[-2] = parts[-2].replace("10_-10", resolution_strs[band_res])
                    # Replace the tile part too.
                    parts[-1] = f"{band_tile[1]}_{band_tile[2]}.tif"
                    fname = "/".join(parts)

                    for band in band_names:
                        cur_fname = fname.replace(f"/{rgb_bands[0]}/", f"/{band}/")
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
                    name=f"sentinel2/{small_tile[0]}_{small_tile[1]}_{small_tile[2]}_{band_res}.tif"
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
