import hashlib
import multiprocessing
import sys

import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath


def handle_window(window: Window):
    if hashlib.sha256(window.name.encode()).hexdigest()[0] in ["0", "1"]:
        split = "val"
    else:
        split = "train"

    if "split" in window.options and window.options["split"] == split:
        return
    window.options["split"] = split
    window.save()


def assign_split(ds_root: str, workers: int = 32):
    ds_path = UPath(ds_root)
    dataset = Dataset(ds_path)
    windows = dataset.load_windows(show_progress=True, workers=workers)
    p = multiprocessing.Pool(workers)
    outputs = p.imap_unordered(handle_window, windows)
    for _ in tqdm.tqdm(outputs, total=len(windows), desc="Assign split"):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    assign_split(ds_root=sys.argv[1])
