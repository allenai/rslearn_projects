"""Remove all landsat state from the ds1020 dataset windows.

The first launch (2026-08-31) ran with no wrs_row filter, so the CLOUD_COVER-
ascending sort filled landsat mosaic groups with nighttime ascending scenes
(L1GT tier-2, rows ~208-212 over California): optical bands flat at the DN-5000
zero-reflectance offset, only thermal carrying signal. The poison lives in two
places per window -- the materialized layers/landsat* rasters AND the landsat
item groups recorded in items.json (materialize reuses saved item groups, it
does not re-query) -- so both must go before re-running prepare/materialize
with the fixed query (wrs_row lte 122 in config_90d.json).

Sentinel-1/Sentinel-2 layers and item groups are untouched (S2 has no night
acquisitions; S1 is radar), as is all window metadata.

    python scrub_landsat.py --root $DS [--dry_run]
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import tqdm


def scrub_window(wdir: str, dry_run: bool) -> tuple[int, int]:
    """Returns (n_layer_dirs_removed, items_json_rewritten)."""
    removed = 0
    layers_dir = os.path.join(wdir, "layers")
    if os.path.isdir(layers_dir):
        for entry in os.listdir(layers_dir):
            if entry == "landsat" or entry.startswith("landsat."):
                if not dry_run:
                    shutil.rmtree(os.path.join(layers_dir, entry))
                removed += 1

    rewrote = 0
    items_path = os.path.join(wdir, "items.json")
    if os.path.exists(items_path):
        with open(items_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "landsat" in data:
                del data["landsat"]
                rewrote = 1
        elif isinstance(data, list):
            kept = [d for d in data if d.get("layer_name") != "landsat"]
            if len(kept) != len(data):
                data = kept
                rewrote = 1
        if rewrote and not dry_run:
            tmp = items_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, items_path)
    return removed, rewrote


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    windows = []
    groups_root = os.path.join(args.root, "windows")
    for group in sorted(os.listdir(groups_root)):
        gdir = os.path.join(groups_root, group)
        windows += [e.path for e in os.scandir(gdir) if e.is_dir()]
    print(f"scanning {len(windows)} windows")

    total_removed = total_rewrote = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for removed, rewrote in tqdm.tqdm(
            pool.map(lambda w: scrub_window(w, args.dry_run), windows),
            total=len(windows),
        ):
            total_removed += removed
            total_rewrote += rewrote
    print(
        f"{'DRY RUN: would remove' if args.dry_run else 'removed'} "
        f"{total_removed} landsat layer dirs; "
        f"{'would rewrite' if args.dry_run else 'rewrote'} "
        f"{total_rewrote} items.json files"
    )


if __name__ == "__main__":
    main()
