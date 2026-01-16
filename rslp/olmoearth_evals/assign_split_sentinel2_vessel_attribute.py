"""Assign train/val/test split."""

import hashlib
import multiprocessing
import sys

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath


def assign_split(window: Window) -> None:
    """Assign split of the window to train, val, or test.

    This is specialized for vessel attribute prediction to only set the split for
    windows where length and type are both known.
    """
    features = GeojsonVectorFormat().decode_vector(
        window.get_layer_dir("info"), window.projection, window.bounds
    )
    assert len(features) == 1
    feat = features[0]

    tile = (window.bounds[0] // 1024, window.bounds[1] // 1024)
    grid_cell_id = f"{window.projection.crs}_{tile[0]}_{tile[1]}"
    first_hex_char_in_hash = hashlib.sha256(grid_cell_id.encode()).hexdigest()[0]
    if "length" not in feat.properties or "type" not in feat.properties:
        split = "unused"
    if first_hex_char_in_hash in ["0", "1", "2", "3"]:
        split = "val"
    elif first_hex_char_in_hash in ["4", "5", "6", "7"]:
        split = "test"
    else:
        split = "train"
    window.options["olmoearth_evals_split"] = split
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
