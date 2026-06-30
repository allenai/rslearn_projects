"""Subset mangrove classification for small OlmoEarth evals.

From the 100K sample_100K group, sample 2000 Mangrove and 2000 non-Mangrove
windows (ignoring original split). The sampled set is then divided into
train/val/test (50/25/25). Windows are already 32x32 so no cropping is needed.
"""

import json
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
import tqdm

SRC = Path(
    "/weka/dfive-default/rslearn-eai/datasets/"
    "mangrove/classification/20250626/windows/sample_100K"
)
SRC_CONFIG = Path(
    "/weka/dfive-default/rslearn-eai/datasets/"
    "mangrove/classification/20250626/config.json"
)
DST = Path(
    "/weka/dfive-default/rslearn-eai/datasets/"
    "olmoearth_evals/small_mangrove"
)

NUM_PROC = 128
PER_CLASS = 2000
RNG_SEED = 0

# 50% train, 25% val, 25% test
TRAIN_FRAC = 0.5
VAL_FRAC = 0.25


def load_window(window_dir: Path) -> tuple[str, str] | None:
    """Return (window_name, label) or None."""
    metadata_path = window_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open() as f:
        metadata = json.load(f)

    label = metadata.get("options", {}).get("label")
    if label is None:
        return None

    return window_dir.name, label


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
                pool.imap_unordered(load_window, window_dirs, chunksize=64),
                total=len(window_dirs),
                desc="Scanning windows",
            )
        )

    # Partition into mangrove vs non-mangrove.
    mangrove: list[str] = []
    non_mangrove: list[str] = []
    skipped = 0
    for r in results:
        if r is None:
            skipped += 1
            continue
        name, label = r
        if label == "Mangrove":
            mangrove.append(name)
        else:
            non_mangrove.append(name)

    print(f"Skipped: {skipped}")
    print(f"Mangrove: {len(mangrove)}")
    print(f"Non-Mangrove: {len(non_mangrove)}")

    # Sample from each class.
    rng = np.random.default_rng(RNG_SEED)
    selected: list[str] = []

    if len(mangrove) <= PER_CLASS:
        selected.extend(mangrove)
    else:
        chosen = rng.choice(mangrove, size=PER_CLASS, replace=False).tolist()
        selected.extend(chosen)

    if len(non_mangrove) <= PER_CLASS:
        selected.extend(non_mangrove)
    else:
        chosen = rng.choice(non_mangrove, size=PER_CLASS, replace=False).tolist()
        selected.extend(chosen)

    print(f"Total selected: {len(selected)}")

    # Shuffle and split into train/val/test.
    selected.sort()
    rng.shuffle(selected)
    n = len(selected)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_names = selected[:n_train]
    val_names = selected[n_train : n_train + n_val]
    test_names = selected[n_train + n_val :]
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
