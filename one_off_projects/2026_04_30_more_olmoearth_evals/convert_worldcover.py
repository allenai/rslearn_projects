"""Sample worldcover windows for OlmoEarth evals.

For each window in the source dataset, read its label_raster to get the set of
class labels present. Then for the train split, sample 200 windows per class
(skipping the nodata class 0); for the val split, take all windows. Finally,
copy the selected windows into the destination split directories.
"""

import json
import multiprocessing as mp
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import tqdm

SRC = Path("/weka/dfive-default/rslearn-eai/datasets/worldcover/windows/20260109")
SRC_CONFIG = Path("/weka/dfive-default/rslearn-eai/datasets/worldcover/config.json")
DST = Path("/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/worldcover_200_per_class")
NUM_PROC = 128
PER_CLASS = 200
RNG_SEED = 0


def load_window(window_dir: Path) -> tuple[str, str, list[int]] | None:
    """Return (split, window_name, unique_nonzero_labels) for a window dir."""
    metadata_path = window_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open() as f:
        metadata = json.load(f)
    split = metadata.get("options", {}).get("split")
    if split is None:
        return None

    label_path = window_dir / "layers" / "label_raster" / "label" / "geotiff.tif"
    if not label_path.exists():
        return None
    with rasterio.open(label_path) as src:
        arr = src.read(1)
    unique_labels = [int(x) for x in np.unique(arr).tolist() if int(x) != 0]
    return split, window_dir.name, unique_labels


def copy_window(args: tuple[Path, Path]) -> None:
    src, dst = args
    if dst.exists():
        return
    shutil.copytree(src, dst)


def main() -> None:
    window_dirs = sorted(p for p in SRC.iterdir() if p.is_dir())
    print(f"Found {len(window_dirs)} windows in {SRC}")

    with mp.Pool(NUM_PROC) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(load_window, window_dirs, chunksize=4),
                total=len(window_dirs),
                desc="Loading labels",
            )
        )

    train_class_to_windows: dict[int, list[str]] = defaultdict(list)
    val_windows: list[str] = []
    skipped = 0
    for r in results:
        if r is None:
            skipped += 1
            continue
        split, name, labels = r
        if split == "train":
            for c in labels:
                train_class_to_windows[c].append(name)
        elif split == "val":
            val_windows.append(name)
        else:
            skipped += 1

    print(f"Skipped (no metadata/split/label): {skipped}")
    print("Train class -> num windows containing class:")
    for c in sorted(train_class_to_windows.keys()):
        print(f"  class {c}: {len(train_class_to_windows[c])} windows")
    print(f"Val windows: {len(val_windows)}")

    rng = np.random.default_rng(RNG_SEED)
    train_to_copy: set[str] = set()
    for c in sorted(train_class_to_windows.keys()):
        names = train_class_to_windows[c]
        if len(names) <= PER_CLASS:
            chosen = names
        else:
            chosen = rng.choice(names, size=PER_CLASS, replace=False).tolist()
        train_to_copy.update(chosen)
    print(f"Total unique train windows to copy: {len(train_to_copy)}")

    DST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC_CONFIG, DST / "config.json")

    train_dst = DST / "windows" / "train"
    val_dst = DST / "windows" / "val"
    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    copy_args: list[tuple[Path, Path]] = []
    for name in sorted(train_to_copy):
        copy_args.append((SRC / name, train_dst / name))
    for name in sorted(val_windows):
        copy_args.append((SRC / name, val_dst / name))

    with mp.Pool(NUM_PROC) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(copy_window, copy_args, chunksize=1),
            total=len(copy_args),
            desc="Copying windows",
        ):
            pass


if __name__ == "__main__":
    main()
