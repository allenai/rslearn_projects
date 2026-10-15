"""Assign spatial-block, territory-stratified train/val/test splits to windows.

Splits are keyed on a window's ~BLOCK_DEG (default 0.1 deg ~= 11 km) lon/lat
block, so whole neighbourhoods stay in one split (no spatial leakage). Within
each territory the blocks are ordered by a stable hash and assigned cumulatively
to hit ~70/15/15 by window count -- so every territory (hence every temperate/
tropical class) is represented in all three splits with balanced ratios.

The result is written to each window's ``options["split"]`` (train/val/test) in
metadata.json -- where rslearn's studio_ingest / eval builder read it -- and a
sidecar ``splits_map.json`` mapping ``"<territory>|<bx>|<by>" -> split`` is
saved next to the dataset root. Pass that map to ``--apply-map`` on a second
dataset (e.g. the 12-monthly-mosaic twin built from the SAME cells) to copy the
identical split onto every co-located window by lookup -- guaranteeing parity.

Run:
  # source dataset: compute + write splits, emit the map
  python assign_splits.py --dataset data/national_ds --group rpg_2019_dec31 \
      --write-map splits_map.json
  # twin dataset: apply the same map (identical splits by construction)
  python assign_splits.py --dataset data/national_ds_monthly --group rpg_2019_dec31 \
      --apply-map splits_map.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyproj

BLOCK_DEG = 0.1  # ~11 km spatial block (whole block -> one split)
RATIOS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def territory_of(lon: float, lat: float) -> str:
    """Coarse territory label from a lon/lat centroid (DROMs + metropole)."""
    if -6 < lon < 10 and 41 < lat < 52:
        return "metropole"
    if -62 < lon < -60.9 and 15.7 < lat < 16.6:
        return "guadeloupe"
    if -61.3 < lon < -60.7 and 14.3 < lat < 15.0:
        return "martinique"
    if -55 < lon < -51 and 2 < lat < 6:
        return "guyane"
    if 55.0 < lon < 56.0 and -21.5 < lat < -20.7:
        return "reunion"
    if 44.9 < lon < 45.4 and -13.1 < lat < -12.5:
        return "mayotte"
    return "unknown"


_TRANSFORMERS: dict[str, pyproj.Transformer] = {}


def _centroid_lonlat(meta: dict) -> tuple[float, float]:
    """Window centroid in EPSG:4326 from its projection + pixel bounds."""
    crs = meta["projection"]["crs"]
    xr, yr = meta["projection"]["x_resolution"], meta["projection"]["y_resolution"]
    minx, miny, maxx, maxy = meta["bounds"]
    cx, cy = (minx + maxx) / 2 * xr, (miny + maxy) / 2 * yr
    if crs not in _TRANSFORMERS:
        _TRANSFORMERS[crs] = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = _TRANSFORMERS[crs].transform(cx, cy)
    return lon, lat


def block_key(lon: float, lat: float) -> str:
    """Location-only key '<territory>|<bx>|<by>' shared across datasets."""
    terr = territory_of(lon, lat)
    return f"{terr}|{int(lon // BLOCK_DEG)}|{int(lat // BLOCK_DEG)}"


def _block_hash(key: str) -> float:
    """Stable [0,1) hash of a block key (for deterministic block ordering)."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) / float(1 << 256)


def build_split_map(windows: list[Path]) -> dict[str, str]:
    """Compute a block_key -> split map with per-territory cumulative 70/15/15."""
    # Count windows per block, and remember each block's territory.
    block_count: Counter[str] = Counter()
    terr_blocks: dict[str, set[str]] = defaultdict(set)
    for mp in windows:
        lon, lat = _centroid_lonlat(json.loads(mp.read_text()))
        key = block_key(lon, lat)
        block_count[key] += 1
        terr_blocks[key.split("|", 1)[0]].add(key)

    split_map: dict[str, str] = {}
    for _terr, blocks in terr_blocks.items():
        ordered = sorted(blocks, key=_block_hash)  # stable, location-only order
        total = sum(block_count[b] for b in ordered)
        seen = 0
        for b in ordered:
            frac = seen / max(total, 1)
            cum = 0.0
            chosen = RATIOS[-1][0]
            for name, r in RATIOS:
                cum += r
                if frac < cum:
                    chosen = name
                    break
            split_map[b] = chosen
            seen += block_count[b]
    return split_map


def _report(windows: list[Path], split_of: dict[str, str]) -> None:
    counts: Counter[str] = Counter()
    by_terr: dict[str, Counter[str]] = {}
    for mp in windows:
        lon, lat = _centroid_lonlat(json.loads(mp.read_text()))
        key = block_key(lon, lat)
        s = split_of[key]
        counts[s] += 1
        by_terr.setdefault(territory_of(lon, lat), Counter())[s] += 1
    n = sum(counts.values())
    for s in ("train", "val", "test"):
        print(f"  {s:5} {counts[s]:5} ({100 * counts[s] / max(n, 1):.1f}%)")
    print("per-territory:")
    for terr in sorted(by_terr):
        c = by_terr[terr]
        tot = sum(c.values())
        print(
            f"  {terr:11} n={tot:5}  "
            f"train={c['train']:4} val={c['val']:4} test={c['test']:4}"
        )


def main() -> None:
    """Compute (or apply) splits and write options['split'] into each window."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="rslearn dataset root")
    ap.add_argument("--group", required=True, help="window group")
    ap.add_argument("--write-map", help="compute splits and save block->split map here")
    ap.add_argument("--apply-map", help="load an existing block->split map and apply it")
    ap.add_argument("--dry-run", action="store_true", help="report without writing")
    args = ap.parse_args()

    windows = sorted(
        (Path(args.dataset) / "windows" / args.group).glob("*/metadata.json")
    )
    print(f"{len(windows)} windows in {args.dataset}/{args.group}")

    if args.apply_map:
        split_map = json.loads(Path(args.apply_map).read_text())
        # Any block absent from the map (should not happen for same-cell twins)
        # falls back to a fresh per-block bucket so nothing is left unlabeled.
        missing = 0
        for mp in windows:
            lon, lat = _centroid_lonlat(json.loads(mp.read_text()))
            if block_key(lon, lat) not in split_map:
                missing += 1
        if missing:
            print(f"WARNING: {missing} windows have no block in the map (fallback)")
    else:
        split_map = build_split_map(windows)
        if args.write_map:
            Path(args.write_map).write_text(json.dumps(split_map, indent=1, sort_keys=True))
            print(f"wrote {len(split_map)} block assignments -> {args.write_map}")

    def split_of(lon: float, lat: float) -> str:
        key = block_key(lon, lat)
        if key in split_map:
            return split_map[key]
        frac = _block_hash(key)
        cum = 0.0
        for name, r in RATIOS:
            cum += r
            if frac < cum:
                return name
        return RATIOS[-1][0]

    resolved: dict[str, str] = {}
    for mp in windows:
        meta = json.loads(mp.read_text())
        lon, lat = _centroid_lonlat(meta)
        s = split_of(lon, lat)
        resolved[block_key(lon, lat)] = s
        if not args.dry_run:
            meta.setdefault("options", {})["split"] = s
            mp.write_text(json.dumps(meta))

    print(f"{'wrote' if not args.dry_run else 'would write'} splits:")
    _report(windows, resolved)


if __name__ == "__main__":
    main()
