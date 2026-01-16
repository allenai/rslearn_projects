import json
import multiprocessing
import sys

import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath


def handle_window(window: Window):
    image_fnames = window.path.glob("layers/*/*/image.png")
    for image_fname in image_fnames:
        metadata_fname = image_fname.parent / "metadata.json"
        if metadata_fname.exists():
            continue
        with metadata_fname.open("w") as f:
            json.dump({"bounds": window.bounds}, f)


def set_single_image_metadata(ds_root: str, workers: int = 32):
    ds_path = UPath(ds_root)
    dataset = Dataset(ds_path)
    windows = dataset.load_windows(show_progress=True, workers=workers)
    p = multiprocessing.Pool(workers)
    outputs = p.imap_unordered(handle_window, windows)
    for _ in tqdm.tqdm(outputs, total=len(windows), desc="Set single image metadata"):
        pass
    p.close()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    set_single_image_metadata(ds_root=sys.argv[1])
