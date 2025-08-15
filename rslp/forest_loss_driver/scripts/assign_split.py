"""Assign split to only the windows that have non-empty label."""

import hashlib
import multiprocessing
import sys

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath


def assign_split(window: Window) -> None:
    """Assign split of the window to train or val."""
    layer_dir = window.get_layer_dir("label")
    features = GeojsonVectorFormat().decode_vector(
        layer_dir, window.projection, window.bounds
    )
    category = features[0].properties["new_label"]
    if category in ["unknown", "unlabeled", "human", "natural"]:
        return

    if window.group in ["20250428_brazil_phase1", "20250428_colombia_phase1"]:
        is_val = hashlib.sha256(window.name.encode()).hexdigest()[0] in [
            "0",
            "1",
            "2",
            "3",
        ]
        if is_val:
            split = "val"
        else:
            split = "train"
    elif window.group in [
        "peru3",
        "peru3_flagged_in_peru",
        "peru_interesting",
        "nadia2",
        "nadia3",
        "brazil_interesting",
    ]:
        split = "train"
    else:
        return

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
