"""
Uploads smaller 256x256 patches to R2.
So we tar random groups of 1000 windows together.
"""

import multiprocessing
import os
import tarfile

import boto3
import tqdm

BATCH_SIZE = 1000
root_dir = "/data/favyenb/rslearn_crop_type/windows/"
group = "eurocrops"

s3 = boto3.resource('s3',
    endpoint_url = os.environ['CLOUDFLARE_R2_ENDPOINT'],
    aws_access_key_id = os.environ['CLOUDFLARE_R2_ACCESS_KEY_ID'],
    aws_secret_access_key = os.environ['CLOUDFLARE_R2_SECRET_ACCESS_KEY'],
)
bucket = s3.Bucket('satlas-explorer-data')

def upload(job):
    batch_idx, batch = job
    tar_fname = f"/tmp/{batch_idx}.tar"
    remote_fname = f"crop_type_mapping_sentinel2_20240330/{group}/{batch_idx}.tar"

    with tarfile.open(tar_fname, "w") as tar_file:
        for window_name in batch:
            parts = window_name.split("_")
            col = int(parts[0])
            row = int(parts[1])
            year = int(parts[2])
            archive_name = f"{year}_{col}_{row}/T00AAA_{year}0701T000000_combined.tif"
            geotiff_fname = os.path.join(root_dir, group, window_name, "layers/sentinel2/B01_B02_B03_B04_B05_B06_B07_B08_B09_B10_B11_B12_B8A/geotiff.tif")
            if not os.path.exists(geotiff_fname):
                print(f"warning: missing {geotiff_fname}")
                continue
            tar_file.add(geotiff_fname, arcname=archive_name)

    bucket.upload_file(tar_fname, remote_fname)
    os.remove(tar_fname)

window_names = os.listdir(os.path.join(root_dir, group))
jobs = []
for i in range(0, len(window_names), BATCH_SIZE):
    batch_idx = i // BATCH_SIZE
    jobs.append((batch_idx, window_names[i:i+BATCH_SIZE]))

with open(f"index_{group}.txt", "w") as f:
    for batch_idx, _ in jobs:
        f.write(f"https://pub-956f3eb0f5974f37b9228e0a62f449bf.r2.dev/crop_type_mapping_sentinel2_20240330/{group}/{batch_idx}.tar\n")

p = multiprocessing.Pool(128)
outputs = p.imap_unordered(upload, jobs)
for _ in tqdm.tqdm(outputs, total=len(jobs)):
    pass
p.close()
