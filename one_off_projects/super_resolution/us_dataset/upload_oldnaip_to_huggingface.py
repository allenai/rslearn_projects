"""Create tar files corresponding to 40 m/pixel tiles combining all the 1.25 m/pixel NAIP
images into the tar file.
Then upload the tar files to Hugging Face.
"""

import csv
import glob
import multiprocessing
import os
import random
import tarfile

import numpy as np
import tqdm
from PIL import Image

input_dirs = [
    "/mnt/oldnaip_1/tiles/naip",
    "/mnt/oldnaip_2/tiles/naip",
]
num_workers = 64
tar_dir = "/mnt/data/oldnaip_tar/"


# Pick best fname for each 1.25 m/pixel tile.
def get_fnames(image_dir):
    return glob.glob(os.path.join(image_dir, "*/*.png"))


jobs = []
for input_dir in input_dirs:
    for image_name in os.listdir(input_dir):
        jobs.append(os.path.join(input_dir, image_name))
image_fnames = []
p = multiprocessing.Pool(num_workers)
random.shuffle(jobs)
outputs = p.imap_unordered(get_fnames, jobs)
for cur_fnames in tqdm.tqdm(outputs, total=len(jobs), desc="Getting NAIP filenames"):
    image_fnames.extend(cur_fnames)
p.close()
print(f"got {len(image_fnames)} files total")

print("grouping files by big tile")
info_by_big_tile = {}
for fname in image_fnames:
    projection = fname.split("/")[-2].split("_")[0].split(":")[1]
    parts = fname.split("/")[-1].split(".")[0].split("_")
    col = int(parts[0])
    row = int(parts[1])
    tile = (projection, col, row)
    big_tile = (projection, col // 32, row // 32)
    if big_tile not in info_by_big_tile:
        info_by_big_tile[big_tile] = []
    info_by_big_tile[big_tile].append((tile, fname))
print(f"got {len(info_by_big_tile)} big tiles total")


# Tar and upload the files corresponding to each 40 m/pixel tile.
def process(job):
    big_tile, infos = job

    tar_fname = os.path.join(tar_dir, f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.tar")
    csv_fname = os.path.join(tar_dir, f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.csv")

    if os.path.exists(tar_fname) and os.path.exists(csv_fname):
        return

    # Group infos by tile.
    fnames_by_tile = {}
    for tile, fname in infos:
        if tile not in fnames_by_tile:
            fnames_by_tile[tile] = []
        fnames_by_tile[tile].append(fname)

    # Identify the best file for each tile.
    best_by_tile = {}
    for tile, fnames in fnames_by_tile.items():
        best_fname = None
        best_invalid = None
        for fname in fnames:
            im = np.array(Image.open(fname))
            num_invalid = np.count_nonzero(im.max(axis=2) <= 1)
            if best_invalid is None or num_invalid < best_invalid:
                best_fname = fname
                best_invalid = num_invalid
        best_by_tile[tile] = best_fname

    with tarfile.open(tar_fname + ".tmp", "w") as tar_file:
        for tile, src_fname in best_by_tile.items():
            dst_fname = f"oldnaip/{tile[0]}_{tile[1]}_{tile[2]}.png"
            tar_file.add(src_fname, arcname=dst_fname)
    os.rename(tar_fname + ".tmp", tar_fname)

    # Write the best fnames to CSV so we can use it when doing Sentinel-2 stuff.
    with open(csv_fname + ".tmp", "w") as f:
        writer = csv.DictWriter(f, ["projection", "col", "row", "naip_scene"])
        writer.writeheader()
        for tile, fname in best_by_tile.items():
            naip_scene = fname.split("/")[-3]
            writer.writerow(
                {
                    "projection": tile[0],
                    "col": tile[1],
                    "row": tile[2],
                    "naip_scene": naip_scene,
                }
            )
    os.rename(csv_fname + ".tmp", csv_fname)


jobs = list(info_by_big_tile.items())
random.shuffle(jobs)
p = multiprocessing.Pool(num_workers)
outputs = p.imap_unordered(process, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Process"):
    pass
p.close()
