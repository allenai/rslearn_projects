"""Assign properties matching the old splits from multisat."""

import json
import multiprocessing
import sys

import tqdm
from rslearn.dataset import Dataset
from upath import UPath

in_fnames = {
    "train": "/multisat/mosaic/splits/turbine_naip_supervision/train.json",
    "val": "/multisat/mosaic/splits/turbine_naip_supervision/val.json",
}


def process(job):
    window, split = job
    if "old_split" in window.options and window.options["old_split"] == split:
        return
    window.options["old_split"] = split
    window.save()


def assign_split(ds_root: str, workers: int = 32):
    ds_path = UPath(ds_root)
    dataset = Dataset(ds_path)
    windows = dataset.load_windows(show_progress=True, workers=workers)
    windows_by_name = {window.name: window for window in windows}

    jobs = []
    for split, fname in in_fnames.items():
        with open(fname) as f:
            for col, row in json.load(f):
                expected_window_name = f"{col*512}_{row*512}"
                if expected_window_name not in windows_by_name:
                    continue
                jobs.append((windows_by_name[expected_window_name], split))

    p = multiprocessing.Pool(workers)
    outputs = p.imap_unordered(process, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Assign old split"):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    assign_split(ds_root=sys.argv[1])
