"""
Download the GeoTIFF and JSON file pairs from GCS.
"""

import argparse
import multiprocessing
import os

from google.cloud import storage
import tqdm
from unidecode import unidecode

src_bucket = None
dst_bucket = None


def worker_init():
    global src_bucket, dst_bucket
    if src_bucket is not None and dst_bucket is not None:
        return
    storage_client = storage.Client()
    src_bucket = storage_client.bucket(args.src_bucket)
    os.sched_setaffinity(0, set(range(multiprocessing.cpu_count())))


def download_file(src_fname, dst_fname):
    if os.path.exists(dst_fname):
        return
    blob = src_bucket.blob(src_fname)
    blob.download_to_filename(dst_fname + ".tmp")
    os.rename(dst_fname + ".tmp", dst_fname)


def retrieve_image(job):
    """
    job is tuple (prefix, out_dir).
    prefix is a path on src_bucket like maxar/mgp/x/y/abc.
    So then abc_labels.json and abc.tif should both exist.
    """
    prefix, out_dir = job
    worker_init()
    label = unidecode(prefix.replace("/", "_"))
    fnames = [
        "_labels.json",
        ".tif",
    ]
    for fname in fnames:
        download_file(f"{prefix}{fname}", os.path.join(out_dir, f"{label}{fname}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_bucket", help="Name of source bucket", default="earthsystem-a1")
    parser.add_argument("--src_prefix", help="Source prefix", default="maxar/")
    parser.add_argument("--out_dir", help="Output directory")
    parser.add_argument("--workers", help="Number of worker threads", type=int, default=32)
    args = parser.parse_args()

    p = multiprocessing.Pool(args.workers)

    worker_init()
    jobs = []
    for blob in src_bucket.list_blobs(prefix=args.src_prefix, match_glob="**/*_labels.json"):
        prefix = blob.name.split("_labels.json")[0]
        jobs.append((prefix, args.out_dir))
    outputs = p.imap_unordered(retrieve_image, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        continue
