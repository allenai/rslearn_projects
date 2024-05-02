"""
    prior-elanding-54: [cdl_2023: materialized] [cdl_2019: materializing]
    prior-elanding-56: [nccm: materializing] [cdl_2021: materialized]
    prior-elanding-52: [eurocrops: materialized] [cdl_2020: materialized] [cdl_2017: materializing]
    prior-elandign-75: [cdl_2022: materialized] [cdl_2018: materializing]
"""

import glob
import multiprocessing
import os

import boto3
import tqdm

root_dir = "/data/favyenb/rslearn_crop_type/windows/"
group = "agrifieldnet"

s3 = boto3.resource('s3',
    endpoint_url = os.environ['CLOUDFLARE_R2_ENDPOINT'],
    aws_access_key_id = os.environ['CLOUDFLARE_R2_ACCESS_KEY_ID'],
    aws_secret_access_key = os.environ['CLOUDFLARE_R2_SECRET_ACCESS_KEY'],
)
bucket = s3.Bucket('satlas-explorer-data')

def upload(job):
    local_fname, remote_fname = job
    print(f"{local_fname} -> {remote_fname}")
    bucket.upload_file(local_fname, remote_fname)

jobs = []
for window_name in os.listdir(os.path.join(root_dir, group)):
    window_dir = os.path.join(root_dir, group, window_name)
    year = int(window_name.split("_")[2])
    fnames = glob.glob(os.path.join(window_dir, "layers", "sentinel2", "*/geotiff.tif"))
    for fname in fnames:
        band = fname.split("/")[-2]
        remote_fname = f"crop_type_mapping_sentinel2_20240328/{group}/{window_name}/T00AAA_{year}0701T000000_{band}.tif"
        jobs.append((fname, remote_fname))

with open(f"index_{group}.txt", "w") as f:
    for _, remote_fname in jobs:
        f.write("https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/" + remote_fname + "\n")

p = multiprocessing.Pool(32)
outputs = p.imap_unordered(upload, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs)):
    pass
p.close()
