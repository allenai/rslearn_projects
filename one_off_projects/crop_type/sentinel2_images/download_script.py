import multiprocessing
import os
import subprocess

import tqdm


def handle(url):
    local_fname = url.split("/")[-1]
    if os.path.exists(local_fname):
        return
    subprocess.check_call(["wget", url, "-O", local_fname])
    subprocess.check_call(["tar", "xvf", local_fname])


with open("index_eurocrops.txt") as f:
    urls = [line.strip() for line in f.readlines() if line.strip()]

p = multiprocessing.Pool(64)
outputs = p.imap_unordered(handle, urls)
for _ in tqdm.tqdm(outputs, total=len(urls)):
    pass
p.close()
