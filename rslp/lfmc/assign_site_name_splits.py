"""Assign split based on site name."""

import hashlib
import multiprocessing
import sys

import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath


def assign_split(window: Window) -> None:
    """Assign split of the window to train or val."""
    site_name = window.options["site_name"]
    is_val = hashlib.sha256(site_name.encode()).hexdigest()[0] in ["0", "1"]
    if is_val:
        split = "val"
    else:
        split = "train"
    window.options["split"] = split
    window.save()


if __name__ == "__main__":
    ds_path = UPath(sys.argv[1])
    multiprocessing.set_start_method("forkserver")
    windows = Dataset(ds_path).load_windows(workers=128, show_progress=True)
    p = multiprocessing.Pool(128)
    outputs = p.imap_unordered(assign_split, windows)
    for _ in tqdm.tqdm(outputs, total=len(windows)):
        pass
    p.close()
