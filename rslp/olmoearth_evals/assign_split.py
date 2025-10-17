"""Assign train/val/test split."""

import hashlib
import multiprocessing
import sys

import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath


def assign_split(window: Window) -> None:
    """Assign split of the window to train, val, or test.

    We split based on a 1024x1024 pixel grid so the train windows are not too close to
    the validation and test windows. We assign 1/4 val, 1/4 test, and 1/2 train.
    """
    tile = (window.bounds[0] // 256, window.bounds[1] // 256)
    grid_cell_id = f"{window.projection.crs}_{tile[0]}_{tile[1]}"
    # first_hex_char_in_hash = hashlib.sha256(grid_cell_id.encode()).hexdigest()[0]
    # if first_hex_char_in_hash in ["0", "1", "2", "3"]:
    #     split = "val"
    # # elif first_hex_char_in_hash in ["4", "5", "6", "7"]:
    # #     split = "test"
    # else:
    #     split = "train"
    # window.options["split_256"] = split

    # Create deterministic hash from input string
    sha_hash = hashlib.sha256(grid_cell_id.encode()).hexdigest()

    # Convert full hash to integer and normalize to [0, 1)
    hash_int = int(sha_hash, 16)
    normalized_hash = hash_int / (2**256)

    # Assign split based on cumulative proportions
    if normalized_hash < 0.75:
        split = "train"
    elif normalized_hash < 1.0:
        split = "val"

    window.options["split_256"] = split
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
