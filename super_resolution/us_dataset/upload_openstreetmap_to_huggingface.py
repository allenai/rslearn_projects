"""
Create tar files corresponding to 40 m/pixel tiles combining all the 1.25 m/pixel NAIP
images into the tar file.
Then upload the tar files to Hugging Face.
"""
import glob
import json
import multiprocessing
import os
import tarfile

import random
import tqdm

input_dir = "/mnt/data/osm/tiles/openstreetmap/"
num_workers = 32
tar_dir = "/mnt/data/openstreetmap_tar/"

# List the geojson fnames.
def get_fnames(pbf_dir):
    return glob.glob(os.path.join(pbf_dir, "*/*.geojson"))
jobs = []
for pbf_name in os.listdir(input_dir):
    jobs.append(os.path.join(input_dir, pbf_name))
vector_fnames = []
p = multiprocessing.Pool(num_workers)
random.shuffle(jobs)
outputs = p.imap_unordered(get_fnames, jobs)
for cur_fnames in tqdm.tqdm(outputs, total=len(jobs), desc="Getting OSM filenames"):
    vector_fnames.extend(cur_fnames)
p.close()
print(f"got {len(vector_fnames)} files total")

print("grouping files by big tile")
info_by_big_tile = {}
for fname in vector_fnames:
    projection = fname.split("/")[-2].split("_")[0].split(":")[1]
    parts = fname.split("/")[-1].split(".")[0].split("_")
    col = int(parts[0])
    row = int(parts[1])
    tile = (projection, col, row)
    big_tile = (projection, col//32, row//32)
    if big_tile not in info_by_big_tile:
        info_by_big_tile[big_tile] = []
    info_by_big_tile[big_tile].append((tile, fname))
print(f"got {len(info_by_big_tile)} big tiles total")

# Tar and upload the files corresponding to each 40 m/pixel tile.
def process(job):
    big_tile, infos = job

    tar_fname = os.path.join(tar_dir, f"{big_tile[0]}_{big_tile[1]}_{big_tile[2]}.tar")
    if os.path.exists(tar_fname):
        return

    # Group infos by tile.
    fnames_by_tile = {}
    for tile, fname in infos:
        if tile not in fnames_by_tile:
            fnames_by_tile[tile] = []
        fnames_by_tile[tile].append(fname)

    # Add the GeoJSONs to the tarfile.
    # For tiles covered by multiple GeoJSONs, we concatenate the features.
    with tarfile.open(tar_fname+".tmp", "w") as tar_file:
        for tile, src_fnames in fnames_by_tile.items():
            if len(src_fnames) == 1:
                src_fname = src_fnames[0]
            else:
                fc = None
                for fname in src_fnames:
                    with open(fname) as f:
                        cur_fc = json.load(f)
                    if not fc:
                        cur_fc = fc
                    else:
                        cur_fc["features"] += fc["features"]
                    src_fname = f"/tmp/{os.getpid()}.geojson"
                    with open(src_fname, "w") as f:
                        json.dump(cur_fc, f)

            dst_fname = f"openstreetmap/{tile[0]}_{tile[1]}_{tile[2]}.geojson"
            tar_file.add(src_fname, arcname=dst_fname)
    os.rename(tar_fname+".tmp", tar_fname)

jobs = list(info_by_big_tile.items())
random.shuffle(jobs)
p = multiprocessing.Pool(num_workers)
outputs = p.imap_unordered(process, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Process"):
    pass
p.close()
