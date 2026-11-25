"""Create a class-balanced v2 annotation JSON from LCC prediction tiles.

Input is a prediction output directory (per tile: the ``<tile>_meta.json`` sidecar)
and the GeoZarr store the predictions were written to. This script:

1. For each tile and each change class the model predicts there, picks one random
   pixel of that class within a connected component of at least ``min_pixels``
   pixels. Components are found per prediction window. Classes are the categories
   of the pre-change and post-change heads; "none" is not a class.
2. Samples up to ``per_class`` of those pixels per class (at most one per tile
   for each class) and writes one v2 annotation entry per pixel, shuffled.

Each entry is a 128x128 window centered on the pixel with one positive point
holding the model's predicted categories and predicted pre-change date (it lacks
first_date_change_noticeable/post_change, so prepare skips it until labeled). The
entry also sets:

- ``anchor_date = T - 6 days`` where T is the tile's prediction reference
  timestamp, so if the entry is relabeled as negative-only, the earliest training
  option's frequent block is exactly [T - 90d, T] like the prediction's.
- ``prediction_items``: the Sentinel-2 scenes the prediction used for this tile,
  which prepare prefers on cloud-cover ties so the option's mosaics use the same
  scenes as prediction did.
- ``pred``: provenance (tile, T, sampled class, binary change score at the pixel).

Usage::

    python -m rslp.olmoearth_lcc.annotation_scripts.phase21_class_balanced.create_class_balanced_annotations \
        --input_dir /weka/.../lcc_model_outputs_20261020_af_sa/ \
        --zarr_store_path /weka/.../lcc_20261020_af_sa.zarr \
        --output annotations_phase21_af_sa_balanced.json \
        --group phase21
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import scipy.ndimage
import shapely.geometry
import tqdm
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from rslp.olmoearth_lcc.lcc_model import geozarr
from rslp.olmoearth_lcc.lcc_model.output_bands import (
    LC_CLASS_NAMES,
    NONE_CATEGORY_IDX,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
)
from rslp.olmoearth_lcc.lcc_model.prepare import OPTION_MARGIN
from rslp.olmoearth_lcc.lcc_model.timestamp_encoding import UNIX_EPOCH

META_SUFFIX = "_meta.json"
DEFAULT_WINDOW_SIZE = 128
DEFAULT_PER_CLASS = 50
DEFAULT_MIN_PIXELS = 10
# Years on either side of the predicted change date for the entry time_range (as in
# the mining annotation script).
TIME_RANGE_YEARS = 3


def _shift_years(dt: datetime, years: int) -> datetime:
    """Shift a datetime by a whole number of years, handling Feb 29."""
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        # Feb 29 -> Feb 28 in a non-leap target year.
        return dt.replace(year=dt.year + years, day=28)


def _pixel_to_lonlat(px: int, py: int, projection: Projection) -> tuple[float, float]:
    """Convert absolute pixel coordinates (column, row) to lon/lat."""
    pt = shapely.geometry.Point(px + 0.5, py + 0.5)
    wgs84_pt = STGeometry(projection, pt, None).to_projection(WGS84_PROJECTION).shp
    return float(wgs84_pt.x), float(wgs84_pt.y)


def _change_category_value(names: list[str], idx: int) -> str:
    """Map a predicted change-category class to its annotation JSON value.

    The "none" class maps to an unset category (the empty string).
    """
    name = names[idx]
    return "" if name == "none" else name


def class_keys() -> list[tuple[str, str, int]]:
    """All sampled classes as (class key, head, class index)."""
    keys = []
    for idx, name in enumerate(PRE_CHANGE_CATEGORY_NAMES):
        if idx <= NONE_CATEGORY_IDX:
            continue
        keys.append((f"pre:{name}", "pre", idx))
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


def _pixel_values(arrays: dict[str, np.ndarray], row: int, col: int) -> dict[str, int]:
    """Get one pixel's values from the store arrays of a window, by band name."""
    values = {
        band: int(arrays[geozarr.CHANGE_ARRAY][i, row, col])
        for i, band in enumerate(geozarr.CHANGE_BANDS)
    }
    values.update(
        {
            band: int(arrays[geozarr.CHANGE_DATES_ARRAY][i, row, col])
            for i, band in enumerate(geozarr.CHANGE_DATES_BANDS)
        }
    )
    return values


def process_tile(
    meta_path: str, store_path: str, min_pixels: int, seed: int
) -> list[dict[str, Any]]:
    """Pick one pixel per predicted class in a tile; return candidate dicts."""
    meta_upath = UPath(meta_path)
    tile = meta_upath.name.removesuffix(META_SUFFIX)
    with meta_upath.open() as f:
        meta = json.load(f)
    projection = Projection.deserialize(meta["projection"])

    head_bands = {
        "pre": geozarr.CHANGE_BANDS.index("pre_change_category"),
        "post": geozarr.CHANGE_BANDS.index("post_change_category"),
    }
    rngs = {key: random.Random(f"{tile}:{key}:{seed}") for key, _, _ in class_keys()}
    # Per class: (number of eligible pixels seen so far, chosen candidate). The
    # candidate is reservoir-sampled across windows, so it is uniform over the
    # tile's eligible pixels.
    chosen: dict[str, tuple[int, dict[str, Any]]] = {}
    for window_meta in meta["windows"]:
        b = window_meta["bounds"]
        bounds = (b[0], b[1], b[2], b[3])
        arrays = geozarr.read_window(store_path, projection, bounds)
        change = arrays[geozarr.CHANGE_ARRAY]
        for key, head, idx in class_keys():
            mask = change[head_bands[head]] == idx
            if mask.sum() < min_pixels:
                continue
            labels, num = scipy.ndimage.label(mask)
            sizes = np.bincount(labels.ravel(), minlength=num + 1)
            big = np.nonzero(sizes >= min_pixels)[0]
            big = big[big > 0]
            if len(big) == 0:
                continue
            ys, xs = np.nonzero(np.isin(labels, big))
            rng = rngs[key]
            seen = chosen[key][0] if key in chosen else 0
            total = seen + len(ys)
            if rng.randrange(total) >= len(ys):
                chosen[key] = (total, chosen[key][1])
                continue
            i = rng.randrange(len(ys))
            row, col = int(ys[i]), int(xs[i])
            chosen[key] = (
                total,
                dict(
                    class_key=key,
                    tile=tile,
                    projection=projection.serialize(),
                    px=bounds[0] + col,
                    py=bounds[1] + row,
                    pixel=_pixel_values(arrays, row, col),
                    time=meta["time_range"][0],
                    prediction_items=_prediction_items(
                        meta, bounds[0] + col, bounds[1] + row
                    ),
                ),
            )
    return [candidate for _, candidate in chosen.values()]


def candidate_to_entry(
    cand: dict[str, Any], group: str, window_size: int
) -> dict[str, Any]:
    """Build the v2 annotation entry for a sampled pixel."""
    projection = Projection.deserialize(cand["projection"])
    px, py = cand["px"], cand["py"]
    pixel = cand["pixel"]
    lon, lat = _pixel_to_lonlat(px, py, projection)

    pre_idx = pixel["pre_change_category"]
    post_idx = pixel["post_change_category"]
    pre_change_date = UNIX_EPOCH + timedelta(days=pixel["change_start"])

    point = {
        "lon": lon,
        "lat": lat,
        "pre_category": LC_CLASS_NAMES[pixel["pre_lc"]],
        "post_category": LC_CLASS_NAMES[pixel["post_lc"]],
        "pre_change_category": _change_category_value(
            PRE_CHANGE_CATEGORY_NAMES, pre_idx
        ),
        "post_change_category": _change_category_value(
            POST_CHANGE_CATEGORY_NAMES, post_idx
        ),
        "pre_change": pre_change_date.date().isoformat(),
    }

    t = datetime.fromisoformat(cand["time"])
    pred: dict[str, Any] = {
        "tile": cand["tile"],
        "time": cand["time"],
        "class": cand["class_key"],
        "binary_change_score": pixel["binary_change"],
    }
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
        "pred": pred,
        "prediction_items": cand["prediction_items"],
    }


def create_annotations(
    input_dir: str,
    zarr_store_path: str,
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
    meta_paths = sorted(str(p) for p in in_path.glob(f"*{META_SUFFIX}"))
    print(f"Found {len(meta_paths)} tiles in {input_dir}")

    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    kwargs_list = [
        dict(meta_path=p, store_path=zarr_store_path, min_pixels=min_pixels, seed=seed)
        for p in meta_paths
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
            entry = candidate_to_entry(cand, group, window_size)
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
    parser.add_argument(
        "--input_dir", required=True, help="Directory with the tiles' _meta.json"
    )
    parser.add_argument("--zarr_store_path", required=True)
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
        zarr_store_path=args.zarr_store_path,
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
