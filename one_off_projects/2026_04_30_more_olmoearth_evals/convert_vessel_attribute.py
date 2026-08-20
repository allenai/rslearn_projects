"""Subset and crop sentinel2_vessel_attribute for small OlmoEarth evals.

Scans all windows across both groups in the source dataset and keeps only those
that have both ``type`` and ``length`` properties.

Two sampling strategies are combined (from ALL eligible windows, ignoring the
original split):
  1. Up to 500 windows per vessel type (9 classes).
  2. Up to 500 windows per length bucket:
     [0,50), [50,75), [75,100), [100,125), [125,150), [150,175),
     [175,200), [200,225), [225,250), [250,+inf)  — 10 buckets total.
The union of both samples is then split into train/val/test (50/25/25).

All sentinel2 GeoTIFFs are center-cropped from 128x128 to 64x64. The info
vector layer is copied as-is.
"""

import json
import multiprocessing as mp
import shutil
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import tqdm
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.geometry import Projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

SRC = Path(
    "/weka/dfive-default/rslearn-eai/datasets/"
    "sentinel2_vessel_attribute/dataset_v1/20250205"
)
SRC_CONFIG = SRC / "config.json"
DST = Path(
    "/weka/dfive-default/rslearn-eai/datasets/"
    "olmoearth_evals/small_sentinel2_vessel_attribute"
)

NUM_PROC = 128
PER_SAMPLE = 500
RNG_SEED = 0
CROP_SIZE = 64

# 50% train, 25% val, 25% test
TRAIN_FRAC = 0.5
VAL_FRAC = 0.25

LENGTH_BUCKET_EDGES = [50, 75, 100, 125, 150, 175, 200, 225, 250]

GROUPS = ["detections_bigtable", "detections_jan_470k"]

GEOTIFF_FORMAT = GeotiffRasterFormat()


def _length_bucket(length: float) -> int:
    """Return bucket index for a given vessel length."""
    return bisect_right(LENGTH_BUCKET_EDGES, length)


def load_window(window_dir: Path) -> tuple[str, str, str, float] | None:
    """Return (group, window_name, vessel_type, length) or None."""
    metadata_path = window_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open() as f:
        metadata = json.load(f)

    geojson_path = window_dir / "layers" / "info" / "data.geojson"
    if not geojson_path.exists():
        return None
    with geojson_path.open() as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    if len(features) != 1:
        return None
    props = features[0].get("properties", {})

    vessel_type = props.get("type")
    length = props.get("length")
    if vessel_type is None or length is None:
        return None

    group = window_dir.parent.name
    return group, window_dir.name, vessel_type, float(length)


def copy_and_crop_window(args: tuple[Path, Path]) -> None:
    """Copy one window, center-cropping sentinel2 rasters to CROP_SIZE."""
    src_window_dir, dst_window_dir = args

    if (dst_window_dir / "metadata.json").exists():
        return

    src_metadata_path = src_window_dir / "metadata.json"
    if not src_metadata_path.exists():
        return
    with src_metadata_path.open() as f:
        metadata = json.load(f)

    bounds = metadata["bounds"]
    cx = (bounds[0] + bounds[2]) // 2
    cy = (bounds[1] + bounds[3]) // 2
    new_bounds = (
        cx - CROP_SIZE // 2,
        cy - CROP_SIZE // 2,
        cx + CROP_SIZE // 2,
        cy + CROP_SIZE // 2,
    )
    projection = Projection.deserialize(metadata["projection"])

    dst_window_dir.mkdir(parents=True, exist_ok=True)

    src_layers_dir = src_window_dir / "layers"
    if src_layers_dir.exists():
        dst_layers_dir = dst_window_dir / "layers"
        for layer_dir in src_layers_dir.iterdir():
            if not layer_dir.is_dir():
                continue

            if layer_dir.name == "info":
                dst_layer_dir = dst_layers_dir / layer_dir.name
                if dst_layer_dir.exists():
                    shutil.rmtree(dst_layer_dir)
                shutil.copytree(layer_dir, dst_layer_dir)
                continue

            if layer_dir.name != "sentinel2":
                continue

            dst_layer_dir = dst_layers_dir / layer_dir.name
            dst_layer_dir.mkdir(parents=True, exist_ok=True)
            for child in layer_dir.iterdir():
                if child.is_file():
                    shutil.copy2(child, dst_layer_dir / child.name)
                    continue
                src_band_dir = UPath(str(child))
                dst_band_dir = UPath(str(dst_layer_dir / child.name))
                raster = GEOTIFF_FORMAT.decode_raster(
                    src_band_dir, projection, new_bounds
                )
                GEOTIFF_FORMAT.encode_raster(
                    dst_band_dir, projection, new_bounds, raster
                )

    src_items_path = src_window_dir / "items.json"
    if src_items_path.exists():
        shutil.copy2(src_items_path, dst_window_dir / "items.json")

    new_metadata = dict(metadata)
    new_metadata["bounds"] = list(new_bounds)
    with open_atomic(UPath(str(dst_window_dir / "metadata.json")), "w") as f:
        json.dump(new_metadata, f)


def main() -> None:
    # Collect all window dirs across groups.
    all_window_dirs: list[Path] = []
    for group in GROUPS:
        group_dir = SRC / "windows" / group
        if not group_dir.exists():
            print(f"Group dir not found: {group_dir}")
            continue
        all_window_dirs.extend(group_dir.iterdir())
    print(f"Found {len(all_window_dirs)} total windows across {len(GROUPS)} groups")

    # Parallel scan.
    with mp.Pool(NUM_PROC) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap_unordered(load_window, all_window_dirs, chunksize=64),
                total=len(all_window_dirs),
                desc="Scanning windows",
            )
        )

    # Group eligible windows by type and length bucket.
    type_to_keys: dict[str, list[tuple[str, str]]] = defaultdict(list)
    bucket_to_keys: dict[int, list[tuple[str, str]]] = defaultdict(list)
    skipped = 0
    for r in results:
        if r is None:
            skipped += 1
            continue
        group, name, vessel_type, length = r
        key = (group, name)
        type_to_keys[vessel_type].append(key)
        bucket_to_keys[_length_bucket(length)].append(key)

    print(f"Skipped: {skipped}")
    print("Windows per vessel type:")
    for t in sorted(type_to_keys.keys()):
        print(f"  {t}: {len(type_to_keys[t])}")
    print("Windows per length bucket:")
    for b in sorted(bucket_to_keys.keys()):
        lo = ([0] + LENGTH_BUCKET_EDGES)[b]
        hi = (LENGTH_BUCKET_EDGES + [None])[b]
        label = f"[{lo}, {hi})" if hi is not None else f"[{lo}, +inf)"
        print(f"  {label}: {len(bucket_to_keys[b])}")

    # Sample: 500 per type + 500 per bucket, then union.
    rng = np.random.default_rng(RNG_SEED)
    selected: set[tuple[str, str]] = set()

    for t in sorted(type_to_keys.keys()):
        keys = type_to_keys[t]
        if len(keys) <= PER_SAMPLE:
            selected.update(keys)
        else:
            idx = rng.choice(len(keys), size=PER_SAMPLE, replace=False)
            selected.update(keys[i] for i in idx)

    for b in sorted(bucket_to_keys.keys()):
        keys = bucket_to_keys[b]
        if len(keys) <= PER_SAMPLE:
            selected.update(keys)
        else:
            idx = rng.choice(len(keys), size=PER_SAMPLE, replace=False)
            selected.update(keys[i] for i in idx)

    print(f"Total selected windows: {len(selected)}")

    # Shuffle and split into train/val/test.
    selected_list = sorted(selected)
    rng.shuffle(selected_list)
    n = len(selected_list)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_keys = selected_list[:n_train]
    val_keys = selected_list[n_train : n_train + n_val]
    test_keys = selected_list[n_train + n_val :]
    print(f"Train: {len(train_keys)}, Val: {len(val_keys)}, Test: {len(test_keys)}")

    # Prepare destination.
    DST.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC_CONFIG, DST / "config.json")
    for split_name in ("train", "val", "test"):
        (DST / "windows" / split_name).mkdir(parents=True, exist_ok=True)

    # Build copy jobs.
    copy_args: list[tuple[Path, Path]] = []
    for group, name in train_keys:
        copy_args.append((
            SRC / "windows" / group / name,
            DST / "windows" / "train" / name,
        ))
    for group, name in val_keys:
        copy_args.append((
            SRC / "windows" / group / name,
            DST / "windows" / "val" / name,
        ))
    for group, name in test_keys:
        copy_args.append((
            SRC / "windows" / group / name,
            DST / "windows" / "test" / name,
        ))

    print(f"Total windows to copy: {len(copy_args)}")

    with mp.Pool(NUM_PROC) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(copy_and_crop_window, copy_args, chunksize=4),
            total=len(copy_args),
            desc="Copying & cropping",
        ):
            pass


if __name__ == "__main__":
    main()
