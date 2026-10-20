"""Create a class-balanced v2 annotation JSON from LCC prediction tiles.

Input is a prediction output directory written with ``write_summary_raster`` (per
tile: ``<tile>_summary.tif`` and the ``<tile>_meta.json`` sidecar). This script:

1. For each tile and each change class the model predicts there, picks one random
   pixel of that class within a connected component of at least ``min_pixels``
   pixels (the polygonization threshold). Classes are the categories of the
   merged pre+same head (pre-change and same-change categories) and of the
   post-change head; "none" is not a class.
2. Samples up to ``per_class`` of those pixels per class (at most one per tile
   for each class) and writes one v2 annotation entry per pixel, shuffled.

Each entry is a 128x128 window centered on the pixel with one positive point
holding the model's predicted categories and predicted pre-change month (it lacks
first_date_change_noticeable/post_change, so prepare skips it until labeled). The
entry also sets:

- ``anchor_date = T - 6 days`` where T is the tile's prediction reference
  timestamp, so if the entry is relabeled as negative-only, the earliest training
  option's frequent block is exactly [T - 90d, T] like the prediction's.
- ``prediction_items``: the Sentinel-2 scenes the prediction used for this tile,
  which prepare prefers on cloud-cover ties so the option's mosaics use the same
  scenes as prediction did.
- ``pred``: provenance (tile, T, sampled class, scores at the pixel, region).

Usage::

    python -m rslp.olmoearth_lcc.annotation_scripts.phase21_class_balanced.create_class_balanced_annotations \
        --input_dir /weka/.../lcc_model_outputs_20261020_af_sa/ \
        --output annotations_phase21_af_sa_balanced.json \
        --group phase21
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import rasterio
import scipy.ndimage
import tqdm
from rslearn.utils.geometry import Projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_raster_projection_and_bounds
from upath import UPath

from rslp.olmoearth_lcc.annotation_scripts.phase12_mining_predictions.create_mining_annotations import (
    _change_category_value,
    _pixel_to_lonlat,
    _shift_years,
)
from rslp.olmoearth_lcc.lcc_model.postprocess import (
    LC_CLASS_NAMES,
    MERGED_SAME_CATEGORY_START,
    NONE_CATEGORY_IDX,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
    SUMMARY_BINARY_CHANGE_BAND,
    SUMMARY_DST_CLASS_BAND,
    SUMMARY_POST_CLASS_BAND,
    SUMMARY_POST_SCORE_BAND,
    SUMMARY_PRE_CLASS_BAND,
    SUMMARY_PRE_SCORE_BAND,
    SUMMARY_SRC_CLASS_BAND,
    SUMMARY_TS_PRE_MONTH_BAND,
    _month_value_to_date,
)
from rslp.olmoearth_lcc.lcc_model.prepare import OPTION_MARGIN

SUMMARY_SUFFIX = "_summary.tif"
META_SUFFIX = "_meta.json"
DEFAULT_WINDOW_SIZE = 128
DEFAULT_PER_CLASS = 50
DEFAULT_MIN_PIXELS = 10
# Years on either side of the predicted change date for the entry time_range (as in
# the mining annotation script).
TIME_RANGE_YEARS = 3


def class_keys() -> list[tuple[str, str, int]]:
    """All sampled classes as (class key, summary head, class index)."""
    keys = []
    for idx, name in enumerate(PRE_CHANGE_CATEGORY_NAMES):
        if idx <= NONE_CATEGORY_IDX:
            continue
        kind = "same" if idx >= MERGED_SAME_CATEGORY_START else "pre"
        keys.append((f"{kind}:{name}", "pre", idx))
    for idx, name in enumerate(POST_CHANGE_CATEGORY_NAMES):
        if idx <= NONE_CATEGORY_IDX:
            continue
        keys.append((f"post:{name}", "post", idx))
    return keys


def _prediction_items(meta: dict[str, Any], px: int, py: int) -> list[str]:
    """Names of all scenes used by the prediction window containing the pixel."""
    for window in meta["windows"]:
        b = window["bounds"]
        if b[0] <= px < b[2] and b[1] <= py < b[3]:
            return sorted(
                {
                    item["name"]
                    for layer in window["layers"].values()
                    for group in layer["groups"]
                    for item in group["items"]
                }
            )
    raise ValueError(f"no prediction window contains pixel ({px}, {py})")


def process_tile(summary_path: str, min_pixels: int, seed: int) -> list[dict[str, Any]]:
    """Pick one pixel per predicted class in a tile; return candidate dicts."""
    summary_upath = UPath(summary_path)
    tile = summary_upath.name.removesuffix(SUMMARY_SUFFIX)
    meta_upath = summary_upath.parent / f"{tile}{META_SUFFIX}"
    if not meta_upath.exists():
        return []
    with meta_upath.open() as f:
        meta = json.load(f)
    with summary_upath.open("rb") as f:
        with rasterio.open(f) as src:
            arr = src.read()
            projection, bounds = get_raster_projection_and_bounds(src)

    head_bands = {"pre": SUMMARY_PRE_CLASS_BAND, "post": SUMMARY_POST_CLASS_BAND}
    candidates = []
    for key, head, idx in class_keys():
        mask = arr[head_bands[head]] == idx
        if mask.sum() < min_pixels:
            continue
        labels, num = scipy.ndimage.label(mask)
        sizes = np.bincount(labels.ravel(), minlength=num + 1)
        big = np.nonzero(sizes >= min_pixels)[0]
        big = big[big > 0]
        if len(big) == 0:
            continue
        ys, xs = np.nonzero(np.isin(labels, big))
        rng = random.Random(f"{tile}:{key}:{seed}")
        i = rng.randrange(len(ys))
        row, col = int(ys[i]), int(xs[i])
        candidates.append(
            dict(
                class_key=key,
                tile=tile,
                projection=projection.serialize(),
                px=bounds[0] + col,
                py=bounds[1] + row,
                pixel=[int(v) for v in arr[:, row, col]],
                time=meta["time_range"][0],
                prediction_items=_prediction_items(
                    meta, bounds[0] + col, bounds[1] + row
                ),
            )
        )
    return candidates


def candidate_to_entry(
    cand: dict[str, Any], group: str, window_size: int, region: str | None
) -> dict[str, Any]:
    """Build the v2 annotation entry for a sampled pixel."""
    projection = Projection.deserialize(cand["projection"])
    px, py = cand["px"], cand["py"]
    pixel = cand["pixel"]
    lon, lat = _pixel_to_lonlat(px, py, projection)

    pre_idx = pixel[SUMMARY_PRE_CLASS_BAND]
    post_idx = pixel[SUMMARY_POST_CLASS_BAND]
    # Split the merged pre+same head back into the pre/same annotation fields.
    if pre_idx >= MERGED_SAME_CATEGORY_START:
        pre_value = ""
        same_value = PRE_CHANGE_CATEGORY_NAMES[pre_idx]
    else:
        pre_value = _change_category_value(PRE_CHANGE_CATEGORY_NAMES, pre_idx)
        same_value = ""
    pre_change_date = _month_value_to_date(pixel[SUMMARY_TS_PRE_MONTH_BAND])

    point = {
        "lon": lon,
        "lat": lat,
        "pre_category": LC_CLASS_NAMES[pixel[SUMMARY_SRC_CLASS_BAND]],
        "post_category": LC_CLASS_NAMES[pixel[SUMMARY_DST_CLASS_BAND]],
        "pre_change_category": pre_value,
        "post_change_category": _change_category_value(
            POST_CHANGE_CATEGORY_NAMES, post_idx
        ),
        "same_change_category": same_value,
        "pre_change": pre_change_date.date().isoformat(),
    }

    t = datetime.fromisoformat(cand["time"])
    half = window_size // 2
    return {
        "projection": cand["projection"],
        "bounds": [
            px - half,
            py - half,
            px - half + window_size,
            py - half + window_size,
        ],
        "window_name": f"{projection.crs}_{px}_{py}",
        "group": group,
        "time_range": [
            _shift_years(pre_change_date, -TIME_RANGE_YEARS).date().isoformat(),
            _shift_years(pre_change_date, TIME_RANGE_YEARS).date().isoformat(),
        ],
        "anchor_date": (t - OPTION_MARGIN).date().isoformat(),
        "description": (
            f"sampled for {cand['class_key']}; prediction at {t.date().isoformat()}"
        ),
        "positive_points": [point],
        "negative_points": [],
        "pred": {
            "tile": cand["tile"],
            "time": cand["time"],
            "class": cand["class_key"],
            "region": region,
            "binary_change_score": pixel[SUMMARY_BINARY_CHANGE_BAND],
            "pre_score": pixel[SUMMARY_PRE_SCORE_BAND],
            "post_score": pixel[SUMMARY_POST_SCORE_BAND],
        },
        "prediction_items": cand["prediction_items"],
    }


def create_annotations(
    input_dir: str,
    output: str,
    group: str,
    per_class: int = DEFAULT_PER_CLASS,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    window_size: int = DEFAULT_WINDOW_SIZE,
    seed: int = 0,
    workers: int = 32,
) -> None:
    """Scan prediction tiles and write the class-balanced v2 annotation JSON."""
    in_path = UPath(input_dir)
    summary_paths = sorted(str(p) for p in in_path.glob(f"*{SUMMARY_SUFFIX}"))
    print(f"Found {len(summary_paths)} summary rasters in {input_dir}")

    # Region per tile from the job writer's manifest, if present.
    regions: dict[str, str] = {}
    manifest = in_path / "tasks.jsonl"
    if manifest.exists():
        with manifest.open() as f:
            for line in f:
                task = json.loads(line)
                regions[task["output"].removesuffix(".geojson")] = task.get("region")

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kwargs_list = [
        dict(summary_path=p, min_pixels=min_pixels, seed=seed) for p in summary_paths
    ]
    with multiprocessing.Pool(workers) as pool:
        results = star_imap_unordered(pool, process_tile, kwargs_list)
        for candidates in tqdm.tqdm(results, total=len(kwargs_list)):
            for cand in candidates:
                by_class[cand["class_key"]].append(cand)

    rng = random.Random(seed)
    entries = []
    seen: set[str] = set()
    for key, _, _ in class_keys():
        cands = sorted(by_class.get(key, []), key=lambda c: c["tile"])
        rng.shuffle(cands)
        num = 0
        for cand in cands:
            if num >= per_class:
                break
            entry = candidate_to_entry(
                cand, group, window_size, regions.get(cand["tile"])
            )
            if entry["window_name"] in seen:
                continue
            seen.add(entry["window_name"])
            entries.append(entry)
            num += 1
        print(f"{key}: {num} sampled from {len(cands)} tiles")

    rng.shuffle(entries)
    with UPath(output).open("w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} entries to {output}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--per_class", type=int, default=DEFAULT_PER_CLASS)
    parser.add_argument("--min_pixels", type=int, default=DEFAULT_MIN_PIXELS)
    parser.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    create_annotations(
        input_dir=args.input_dir,
        output=args.output,
        group=args.group,
        per_class=args.per_class,
        min_pixels=args.min_pixels,
        window_size=args.window_size,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
