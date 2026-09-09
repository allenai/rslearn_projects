"""Subset EuroSAT for small OlmoEarth evals.

From the 27K windows in the default group, sample 200 per class for train,
200 per class for val, and 200 per class for test (10 classes, 6000 total).
Windows are already 64x64 so no cropping is needed.
"""

import json
import multiprocessing as mp
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import tqdm

SRC = Path(
    "/weka/dfive-default/rslearn-eai/datasets/"
    "eurosat/rslearn_dataset/windows/default"
)
SRC_CONFIG = Path(
    "/weka/dfive-default/rslearn-eai/datasets/"
    "eurosat/rslearn_dataset/config.json"
)
DST = Path(
    "/weka/dfive-default/rslearn-eai/datasets/"
    "olmoearth_evals/small_eurosat"
)

NUM_PROC = 128
PER_CLASS_PER_SPLIT = 200
RNG_SEED = 0


def copy_window(args: tuple[Path, Path]) -> None:
    src, dst = args
    if dst.exists():
        return
    shutil.copytree(src, dst)


def main() -> None:
    window_dirs = sorted(p for p in SRC.iterdir() if p.is_dir())
    print(f"Found {len(window_dirs)} windows in {SRC}")

    # Group by class (class is the prefix before the last underscore+number).
    class_to_names: dict[str, list[str]] = defaultdict(list)
    for d in window_dirs:
        # Window names are like "AnnualCrop_1", "HerbaceousVegetation_100"
        # Class is everything before the last "_".
        name = d.name
        category = name.rsplit("_", 1)[0]
        class_to_names[category].append(name)

    print("Windows per class:")
    for c in sorted(class_to_names.keys()):
        print(f"  {c}: {len(class_to_names[c])}")

    # For each class, shuffle and split into train/val/test.
    rng = np.random.default_rng(RNG_SEED)
    train_names: list[str] = []
    val_names: list[str] = []
    test_names: list[str] = []

    for c in sorted(class_to_names.keys()):
        names = np.array(class_to_names[c])
        rng.shuffle(names)
        n_train = PER_CLASS_PER_SPLIT
        n_val = PER_CLASS_PER_SPLIT
        n_test = PER_CLASS_PER_SPLIT
        train_names.extend(names[:n_train].tolist())
        val_names.extend(names[n_train : n_train + n_val].tolist())
        test_names.extend(names[n_train + n_val : n_train + n_val + n_test].tolist())

    print(f"Train: {len(train_names)}, Val: {len(val_names)}, Test: {len(test_names)}")

    # Prepare destination.
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC_CONFIG, DST / "config.json")
    for split_name in ("train", "val", "test"):
        (DST / "windows" / split_name).mkdir(parents=True, exist_ok=True)

    # Build copy jobs.
    copy_args: list[tuple[Path, Path]] = []
    for name in sorted(train_names):
        copy_args.append((SRC / name, DST / "windows" / "train" / name))
    for name in sorted(val_names):
        copy_args.append((SRC / name, DST / "windows" / "val" / name))
    for name in sorted(test_names):
        copy_args.append((SRC / name, DST / "windows" / "test" / name))

    print(f"Total windows to copy: {len(copy_args)}")

    with mp.Pool(NUM_PROC) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(copy_window, copy_args, chunksize=4),
            total=len(copy_args),
            desc="Copying windows",
        ):
            pass


if __name__ == "__main__":
    main()
