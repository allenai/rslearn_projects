"""Just print the unique categories."""

import json
import multiprocessing
import sys

import tqdm
from upath import UPath


def get_category(window_dir: UPath) -> str:
    """Get the category for this window."""
    fname = window_dir / "layers" / "label" / "data.geojson"
    with fname.open() as f:
        fc = json.load(f)
    return fc["features"][0]["properties"]["label"]


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    ds_path = UPath(sys.argv[1])

    window_dirs = list((ds_path / "windows" / "train").iterdir())
    p = multiprocessing.Pool(64)
    outputs = p.imap_unordered(get_category, window_dirs)
    categories = set(tqdm.tqdm(outputs, total=len(window_dirs)))
    for category in categories:
        print(f"- {category}")
