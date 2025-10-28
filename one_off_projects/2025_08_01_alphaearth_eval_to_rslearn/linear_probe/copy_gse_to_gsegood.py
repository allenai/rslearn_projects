"""Pick which GSE layer is good and copy it to a "gsegood" layer.

gsegood is referenced in the fine-tuning model config.
"""

import multiprocessing
import shutil
import sys

import numpy as np
import rasterio
import tqdm
from upath import UPath

GSE_BAND_DIR = "610a2ee7942f0f42b7a9bddea505048ac5b6d68739b0d848c0b30e36b201d01a"


def process_window(window_dir: UPath) -> None:
    good_fname: UPath | None = None
    for layer_name in ["gse", "gse.1", "gse.2", "gse.3"]:
        fname = window_dir / "layers" / layer_name / GSE_BAND_DIR / "geotiff.tif"
        if not fname.exists():
            continue
        array = rasterio.open(fname).read()
        if np.count_nonzero(array.max(axis=0) == 0) > 0:
            continue
        good_fname = fname

    if good_fname is None:
        return

    dst_fname = window_dir / "layers" / "gsegood" / GSE_BAND_DIR / "geotiff.tif"
    dst_fname.parent.mkdir(parents=True, exist_ok=True)
    with good_fname.open("rb") as src:
        with dst_fname.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    (window_dir / "layers" / "gsegood" / "completed").touch()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    ds_path = UPath(sys.argv[1])
    window_dirs = list(ds_path.glob("windows/*/*"))

    p = multiprocessing.Pool(64)
    outputs = p.imap_unordered(process_window, window_dirs)
    for _ in tqdm.tqdm(outputs, total=len(window_dirs)):
        pass
    p.close()
