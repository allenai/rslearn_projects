import json
import multiprocessing
import sys

import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath


def handle_window(window: Window):
    if "has_objects" in window.options:
        return
    try:
        if (window.path / "gt.json").exists():
            with open(window.path / "gt.json") as f:
                window.options["has_objects"] = bool(json.load(f))
        else:
            with open(window.path / "layers" / "label" / "data.geojson") as f:
                geodata = json.load(f)
                window.options["has_objects"] = len(geodata["features"]) > 0
        window.save()
    except FileNotFoundError:
        pass


def assign_split(ds_root: str, workers: int = 32):
    ds_path = UPath(ds_root)
    dataset = Dataset(ds_path)
    windows = dataset.load_windows(show_progress=True, workers=workers)
    p = multiprocessing.Pool(workers)
    outputs = p.imap_unordered(handle_window, windows)
    for _ in tqdm.tqdm(outputs, total=len(windows), desc="Object detection tags"):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    assign_split(ds_root=sys.argv[1])
