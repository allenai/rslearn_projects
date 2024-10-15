"""Create WorldCover tar files."""

import glob
import io
import multiprocessing
import os
import random
import tarfile

import numpy as np
import tqdm
from PIL import Image

input_dir = "/mnt/data/worldcover/tiles/worldcover"
num_workers = 64
tar_dir = "/mnt/data/hfupload_worldcover/"


# List WorldCover 10 m/pixel files.
# We keep all the files so later we can choose the ones with fewest missing pixels.
def get_fnames(image_dir):
    return glob.glob(os.path.join(image_dir, "B1/*/*.png"))


jobs = []
for image_name in os.listdir(input_dir):
    jobs.append(os.path.join(input_dir, image_name))
image_fnames = []
p = multiprocessing.Pool(num_workers)
random.shuffle(jobs)
outputs = p.imap_unordered(get_fnames, jobs)
for cur_fnames in tqdm.tqdm(
    outputs, total=len(jobs), desc="Getting WorldCover filenames"
):
    image_fnames.extend(cur_fnames)
p.close()
print(f"got {len(image_fnames)} files total")

# Group the 10 m/pixel filenames by 40 m/pixel tiles.
info_by_big_tile = {}
for fname in image_fnames:
    projection = fname.split("/")[-2].split("_")[0].split(":")[1]
    parts = fname.split("/")[-1].split(".")[0].split("_")
    col = int(parts[0])
    row = int(parts[1])
    tile = (projection, col, row)
    big_tile = (projection, col // 4, row // 4)
    if big_tile not in info_by_big_tile:
        info_by_big_tile[big_tile] = []
    info_by_big_tile[big_tile].append((tile, fname))
print(f"got {len(info_by_big_tile)} big tiles total")


def process(job):
    big_tile, infos = job

    worldcover_fnames = {}
    for tile, fname in infos:
        if tile not in worldcover_fnames:
            worldcover_fnames[tile] = []
        worldcover_fnames[tile].append(fname)

    tar_fname = os.path.join(tar_dir, f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.tar")

    if os.path.exists(tar_fname):
        return

    with tarfile.open(tar_fname + ".tmp", "w") as tar_file:
        # Maintain an image cache from fname -> image array.
        # Since we'll be reusing large 10 m/pixel images for lots of 1.25 m/pixel tiles.
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

        # Iterate over each 1.25 m/pixel tile.
        # Find crops of corresponding 10 m/pixel WorldCover files and pick the one with
        # fewest missing pixels.
        for small_col in range(big_tile[1] * 32, (big_tile[1] + 1) * 32):
            for small_row in range(big_tile[2] * 32, (big_tile[2] + 1) * 32):
                factor = 8
                worldcover_tile = (
                    big_tile[0],
                    small_col // factor,
                    small_row // factor,
                )
                crop_size = 512 // factor
                crop_start = (
                    small_col - worldcover_tile[1] * factor,
                    small_row - worldcover_tile[2] * factor,
                )

                best_image = None
                best_missing = None
                for fname in worldcover_fnames.get(worldcover_tile, []):
                    image = get_image(fname)
                    crop = image[
                        crop_start[1] * crop_size : (crop_start[1] + 1) * crop_size,
                        crop_start[0] * crop_size : (crop_start[0] + 1) * crop_size,
                    ]
                    cur_missing = np.count_nonzero(crop == 0)
                    if best_image is None or cur_missing < best_missing:
                        best_image = crop
                        best_missing = cur_missing

                if best_image is None:
                    continue

                buf = io.BytesIO()
                Image.fromarray(best_image).save(buf, format="PNG")
                out_entry = tarfile.TarInfo(
                    name=f"worldcover/{big_tile[0]}_{small_col}_{small_row}.png"
                )
                out_entry.size = buf.getbuffer().nbytes
                out_entry.mode = 0o644
                buf.seek(0)
                tar_file.addfile(out_entry, fileobj=buf)

    os.rename(tar_fname + ".tmp", tar_fname)


jobs = list(info_by_big_tile.items())
random.shuffle(jobs)
p = multiprocessing.Pool(num_workers)
outputs = p.imap_unordered(process, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Process"):
    pass
p.close()
