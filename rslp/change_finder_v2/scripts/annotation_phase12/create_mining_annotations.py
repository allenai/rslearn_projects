"""Create a v2 annotation JSON of predicted mining pixels from write_raster outputs.

The random-2048 prediction runs (``write_jobs_random_2048``) produce, for each
2048x2048 tile, a 57-band ``output_change`` GeoTIFF (when ``write_raster`` is set)
plus a sibling GeoJSON with the same basename. This script:

1. Scans every ``.tif`` in the input directory.
2. Selects pixels predicted as mining by the post-change-category head:
   - If ``--threshold`` is provided, pixels where the ``post_change_mining``
     probability band (softmax * 255) is >= the threshold.
   - Otherwise, pixels where the argmax over the post-change classes (skipping
     class 0 = nodata) is mining.
   One qualifying pixel per tile is randomly (but deterministically) selected.
3. Reads the per-pixel argmax source/destination land cover category, the
   predicted change categories, and the predicted pre-change date (decoded from
   the day-encoded timestamp band) at that pixel. The pre head is the merged
   pre+same head, so its prediction is split back into the separate
   pre/same annotation fields.
4. Emits a v2 annotation entry (a 128x128 window centered on the pixel, with one
   positive point) suitable for
   ``rslp.change_finder_v2.annotation_app.create_windows``.

Usage::

    python -m rslp.change_finder_v2.scripts.annotation_phase12.create_mining_annotations \
        --input_dir /path/to/write_raster_outputs/ \
        --output mining_annotations.json \
        --group phase12 \
        --threshold 128
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import random
from datetime import datetime

import numpy as np
import rasterio
import shapely
import shapely.geometry
import tqdm
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_raster_projection_and_bounds
from upath import UPath

from rslp.change_finder_v2.lcc_model.postprocess import (
    DST_BAND_OFFSET,
    LC_CLASS_NAMES,
    MERGED_SAME_CATEGORY_START,
    NUM_LC_CLASSES,
    POST_CHANGE_BAND_OFFSET,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_BAND_OFFSET,
    PRE_CHANGE_CATEGORY_NAMES,
    SRC_BAND_OFFSET,
    TS_PRE_DAYS_BAND,
)
from rslp.change_finder_v2.lcc_model.timestamp_encoding import days_to_date

# Class index of "mining" within the post-change category classes.
MINING_CLASS_IDX = POST_CHANGE_CATEGORY_NAMES.index("mining")
# Side length (pixels) of the annotation window centered on the chosen pixel.
DEFAULT_WINDOW_SIZE = 128
# Number of years on either side of the change date for the entry time_range.
TIME_RANGE_YEARS = 3


def _shift_years(dt: datetime, years: int) -> datetime:
    """Shift a datetime by a whole number of years, handling Feb 29."""
    try:
        return dt.replace(year=dt.year + years)
    except ValueError:
        # Feb 29 -> Feb 28 in a non-leap target year.
        return dt.replace(year=dt.year + years, day=28)


def _pixel_to_lonlat(px: int, py: int, projection: object) -> tuple[float, float]:
    """Convert absolute pixel coordinates (column, row) to lon/lat."""
    pt = shapely.geometry.Point(px + 0.5, py + 0.5)
    wgs84_pt = STGeometry(projection, pt, None).to_projection(WGS84_PROJECTION).shp
    return float(wgs84_pt.x), float(wgs84_pt.y)


def _change_category_value(names: list[str], idx: int) -> str:
    """Map a predicted change-category class to its annotation JSON value.

    The annotation app represents an unset change category as the empty string
    (the em-dash dropdown option), so the "none" class maps to "" rather than
    the literal string "none" (which would show up as an extra "none (legacy)"
    option in the app).
    """
    name = names[idx]
    return "" if name == "none" else name


def process_tile(
    tif_path_str: str,
    group: str,
    threshold: int | None,
    window_size: int,
    seed: int,
) -> dict | None:
    """Process one prediction raster and return a v2 annotation entry, or None."""
    tif_path = UPath(tif_path_str)

    with tif_path.open("rb") as f:
        with rasterio.open(f) as src:
            arr = src.read()
            projection, bounds = get_raster_projection_and_bounds(src)

    post_cat_bands = arr[
        POST_CHANGE_BAND_OFFSET : POST_CHANGE_BAND_OFFSET
        + len(POST_CHANGE_CATEGORY_NAMES)
    ]
    if threshold is not None:
        # Threshold the mining probability band (softmax * 255).
        mining_mask = post_cat_bands[MINING_CLASS_IDX] >= threshold
    else:
        # Argmax over post-change classes, skipping class 0 (nodata).
        post_cat_argmax = post_cat_bands[1:].argmax(axis=0) + 1
        mining_mask = post_cat_argmax == MINING_CLASS_IDX
    ys, xs = np.where(mining_mask)
    if len(ys) == 0:
        return None

    # Deterministically pick one mining pixel for this tile.
    rng = random.Random(f"{tif_path.name}:{seed}")
    i = rng.randrange(len(ys))
    row, col = int(ys[i]), int(xs[i])

    src_probs = arr[SRC_BAND_OFFSET : SRC_BAND_OFFSET + NUM_LC_CLASSES, row, col]
    dst_probs = arr[DST_BAND_OFFSET : DST_BAND_OFFSET + NUM_LC_CLASSES, row, col]

    # Skip class 0 (nodata) by taking argmax over classes 1..12 and adding 1.
    src_idx = int(src_probs[1:].argmax()) + 1
    dst_idx = int(dst_probs[1:].argmax()) + 1

    # Predicted change categories at the pixel (skip class 0 = nodata; class 1 =
    # "none" is a valid prediction).
    pre_cat_probs = arr[
        PRE_CHANGE_BAND_OFFSET : PRE_CHANGE_BAND_OFFSET
        + len(PRE_CHANGE_CATEGORY_NAMES),
        row,
        col,
    ]
    post_cat_probs = post_cat_bands[:, row, col]
    pre_cat_idx = int(pre_cat_probs[1:].argmax()) + 1
    post_cat_idx = int(post_cat_probs[1:].argmax()) + 1

    # The pre head is the merged pre+same head; split its prediction back into
    # the separate pre/same annotation fields (the former same categories are
    # appended after the pre categories).
    if pre_cat_idx >= MERGED_SAME_CATEGORY_START:
        pre_change_value = ""
        same_change_value = PRE_CHANGE_CATEGORY_NAMES[pre_cat_idx]
    else:
        pre_change_value = _change_category_value(
            PRE_CHANGE_CATEGORY_NAMES, pre_cat_idx
        )
        same_change_value = ""

    # Predicted pre-change date, decoded from the day-encoded timestamp band.
    pre_change_date = days_to_date(int(arr[TS_PRE_DAYS_BAND, row, col]))

    px = bounds[0] + col
    py = bounds[1] + row
    lon, lat = _pixel_to_lonlat(px, py, projection)

    point: dict = {
        "lon": lon,
        "lat": lat,
        "pre_category": LC_CLASS_NAMES[src_idx],
        "post_category": LC_CLASS_NAMES[dst_idx],
        "pre_change_category": pre_change_value,
        "post_change_category": _change_category_value(
            POST_CHANGE_CATEGORY_NAMES, post_cat_idx
        ),
        "same_change_category": same_change_value,
    }

    time_range: list[str] | None = None
    point["pre_change"] = pre_change_date.date().isoformat()
    time_range = [
        _shift_years(pre_change_date, -TIME_RANGE_YEARS).date().isoformat(),
        _shift_years(pre_change_date, TIME_RANGE_YEARS).date().isoformat(),
    ]

    half = window_size // 2
    entry_bounds = [
        px - half,
        py - half,
        px - half + window_size,
        py - half + window_size,
    ]

    entry: dict = {
        "projection": projection.serialize(),
        "bounds": entry_bounds,
        "window_name": f"{projection.crs}_{px}_{py}",
        "group": group,
        "positive_points": [point],
        "negative_points": [],
    }
    if time_range is not None:
        entry["time_range"] = time_range

    return entry


def create_annotations(
    input_dir: str,
    output: str,
    group: str,
    threshold: int | None = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    seed: int = 0,
    workers: int = 32,
) -> None:
    """Scan write_raster outputs and write a v2 annotation JSON."""
    in_path = UPath(input_dir)
    tif_paths = sorted(str(p) for p in in_path.glob("*.tif"))
    print(f"Found {len(tif_paths)} prediction rasters in {input_dir}")

    kwargs_list = [
        dict(
            tif_path_str=p,
            group=group,
            threshold=threshold,
            window_size=window_size,
            seed=seed,
        )
        for p in tif_paths
    ]

    entries: list[dict] = []
    with multiprocessing.Pool(workers) as pool:
        results = star_imap_unordered(pool, process_tile, kwargs_list)
        for entry in tqdm.tqdm(results, total=len(kwargs_list), desc="Processing"):
            if entry is not None:
                entries.append(entry)

    random.shuffle(entries)

    out_path = UPath(output)
    with out_path.open("w") as f:
        json.dump(entries, f, indent=2)
    print(f"Wrote {len(entries)} annotation entries to {output}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a v2 annotation JSON of predicted mining pixels from "
            "change_finder_v2 write_raster outputs."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing the per-tile .tif and sibling .geojson outputs.",
    )
    parser.add_argument(
        "--output", required=True, help="Output v2 annotation JSON path."
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help=(
            "Mining probability threshold (0-255). If omitted, pixels where the "
            "argmax over post-change classes is mining are selected instead."
        ),
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="Side length (pixels) of each annotation window. Default 128.",
    )
    parser.add_argument(
        "--group",
        required=True,
        help="Window group name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for per-tile pixel selection. Default 0.",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    create_annotations(
        input_dir=args.input_dir,
        output=args.output,
        group=args.group,
        threshold=args.threshold,
        window_size=args.window_size,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
