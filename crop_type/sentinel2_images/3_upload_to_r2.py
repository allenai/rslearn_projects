import os
import boto3

s3 = boto3.resource('s3',
    endpoint_url = os.environ['CLOUDFLARE_R2_ENDPOINT'],
    aws_access_key_id = os.environ['CLOUDFLARE_R2_ACCESS_KEY_ID'],
    aws_secret_access_key = os.environ['CLOUDFLARE_R2_SECRET_ACCESS_KEY'],
)
bucket = s3.Bucket('satlas-explorer-data')

for window_name in os.listdir('.'):
    year = int(window_name.split("_")[2])
    sentinel2_dir = os.path.join(window_name, "layers", "sentinel2")
    if not os.path.exists(sentinel2_dir):
        continue
    for fname in os.listdir(sentinel2_dir):
        if not fname.endswith('.tif'):
            continue
        local_fname = os.path.join(sentinel2_dir, fname)
        remote_fname = f"crop_type_mapping_sentinel2/{window_name}/T00AAA_{year}0701T000000_{fname}"
        print(f"{local_fname} -> {remote_fname}")
        bucket.upload_file(local_fname, remote_fname)
