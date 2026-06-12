"""Find locations and years with the most land cover change.

Consumes the per-year land cover probability GeoTIFFs produced by
``rslp/change_finder/compute_land_cover_change.py`` and, for each window,
picks the combination of pivot year and (source class, destination class)
that yields the largest number of confidently-changed pixels.

For a given pivot year Y, the early window is the ``num_early`` years
immediately before Y (Y excluded) and the late window is the ``num_late``
years immediately after Y.  A pixel qualifies for a (src, dst) pair when the
src class probability exceeds ``threshold`` at every early timestep and the
dst class probability exceeds ``threshold`` at every late timestep.
Qualifying pixels are then filtered by a ``min_pixels`` connected-component
minimum area before being counted.

The script iterates over every valid pivot year and every (src, dst) pair
(excluding nodata and src == dst) and keeps the one with the highest pixel
count.  Windows where no combination produces at least one qualifying CC are
skipped entirely.

Output: a single JSON file containing a list of dicts with keys
``window_name``, ``window_group``, ``pivot_year``, ``src_class_id``,
``src_class_name``, ``dst_class_id``, ``dst_class_name``, and ``num_pixels``.
"""

import argparse
import json
import multiprocessing
import multiprocessing.pool
from collections.abc import Iterable

import numpy as np
import tqdm
from rslearn.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from scipy import ndimage
from upath import UPath

RASTER_FORMAT = GeotiffRasterFormat()

BASE_YEAR = 2016
NUM_YEARS = 10
NUM_CLASSES = 13
# Class 0 is nodata; transitions in/out of nodata are data-availability
# artifacts rather than real land cover change, so we skip that class as
# both src and dst when searching for the best combination.
NODATA_CLASS = 0
CLASS_NAMES = [
    "nodata",
    "bare",
    "burnt",
    "crops",
    "fallow/shifting cultivation",
    "grassland",
    "Lichen and moss",
    "shrub",
    "snow and ice",
    "tree",
    "urban/built-up",
    "water",
    "wetland (herbaceous)",
]


def _filtered_pixel_count(mask: np.ndarray, min_pixels: int) -> int:
    """Return the number of pixels in mask that belong to a CC of size >= min_pixels."""
    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        return 0
    sizes = ndimage.sum(mask, labels, range(1, num_features + 1))
    good_labels = np.where(sizes >= min_pixels)[0] + 1
    if len(good_labels) == 0:
        return 0
    filtered = np.isin(labels, good_labels)
    return int(filtered.sum())


def _process_window(
    window: Window,
    threshold: float,
    num_early: int,
    num_late: int,
    min_pixels: int,
    probs_filename: str,
) -> dict | None:
    """Process one window.

    Returns a summary dict for the (pivot_year, src, dst) combination with the
    most confidently-changed pixels (after ``min_pixels`` CC filtering), or
    ``None`` if no combination has at least one qualifying CC.
    """
    window_root = window.storage.get_window_root(window.group, window.name)
    if not (window_root / probs_filename).exists():
        return None

    min_pivot = num_early
    max_pivot = NUM_YEARS - 1 - num_late
    if max_pivot < min_pivot:
        return None

    probs_raster: RasterArray = RASTER_FORMAT.decode_raster(
        window_root, window.projection, window.bounds, fname=probs_filename
    )
    probs_data = probs_raster.get_chw_array()  # (NUM_YEARS * NUM_CLASSES, H, W) uint8
    _, h, w = probs_data.shape

    # Per-year per-class confidence mask.  Matches the float comparison used
    # in create_land_cover_change_geojson.py.
    probs_float = probs_data.astype(np.float32) / 255.0
    probs_float = probs_float.reshape(NUM_YEARS, NUM_CLASSES, h, w)
    confident = probs_float > threshold  # (NUM_YEARS, NUM_CLASSES, H, W) bool

    best: dict | None = None
    best_count = 0

    for pivot_year_idx in range(min_pivot, max_pivot + 1):
        early_slice = slice(pivot_year_idx - num_early, pivot_year_idx)
        late_slice = slice(pivot_year_idx + 1, pivot_year_idx + 1 + num_late)
        # AND across years => (NUM_CLASSES, H, W)
        early_confident = np.all(confident[early_slice], axis=0)
        late_confident = np.all(confident[late_slice], axis=0)

        for src_id in range(NUM_CLASSES):
            if src_id == NODATA_CLASS:
                continue
            src_mask = early_confident[src_id]
            # An intersection with any dst mask cannot exceed src_mask.sum(),
            # so skip this src entirely if even its raw count can't beat the
            # current best (or doesn't reach min_pixels).
            if int(src_mask.sum()) < max(min_pixels, best_count + 1):
                continue
            for dst_id in range(NUM_CLASSES):
                if dst_id == NODATA_CLASS or dst_id == src_id:
                    continue
                mask = src_mask & late_confident[dst_id]
                raw = int(mask.sum())
                if raw < max(min_pixels, best_count + 1):
                    continue
                count = _filtered_pixel_count(mask, min_pixels)
                if count <= best_count:
                    continue
                best_count = count
                best = {
                    "window_name": window.name,
                    "window_group": window.group,
                    "pivot_year": BASE_YEAR + pivot_year_idx,
                    "src_class_id": src_id,
                    "src_class_name": CLASS_NAMES[src_id],
                    "dst_class_id": dst_id,
                    "dst_class_name": CLASS_NAMES[dst_id],
                    "num_pixels": count,
                }

    return best


def _process_window_star(kwargs: dict) -> dict | None:
    return _process_window(**kwargs)


def find_change(
    ds_path: str,
    threshold: float = 0.75,
    num_early: int = 3,
    num_late: int = 3,
    min_pixels: int = 10,
    out_path: str = "land_cover_change_summary.json",
    probs_filename: str = "land_cover_probs.tif",
    workers: int = 32,
) -> None:
    """Scan all windows and write a JSON list of best change events."""
    dataset = Dataset(UPath(ds_path))
    windows = dataset.load_windows(workers=128, show_progress=True)

    kwargs_list = [
        dict(
            window=window,
            threshold=threshold,
            num_early=num_early,
            num_late=num_late,
            min_pixels=min_pixels,
            probs_filename=probs_filename,
        )
        for window in windows
    ]

    results: Iterable[dict | None]
    pool: multiprocessing.pool.Pool | None = None
    if workers <= 0:
        results = map(_process_window_star, kwargs_list)
    else:
        pool = multiprocessing.Pool(workers)
        results = pool.imap_unordered(_process_window_star, kwargs_list)

    summaries: list[dict] = []
    try:
        for summary in tqdm.tqdm(
            results, total=len(kwargs_list), desc="Scanning windows"
        ):
            if summary is not None:
                summaries.append(summary)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    with open(out_path, "w") as f:
        json.dump(summaries, f)

    print(f"Wrote {len(summaries)} window summaries to {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Find locations and years with the most land cover change"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Per-timestep class probability threshold for 'confident' pixels",
    )
    parser.add_argument("--num_early", type=int, default=3)
    parser.add_argument("--num_late", type=int, default=3)
    parser.add_argument("--min_pixels", type=int, default=10)
    parser.add_argument(
        "--out_path",
        default="land_cover_change_summary.json",
        help="Output JSON path (list of per-window summary dicts)",
    )
    parser.add_argument(
        "--probs_filename",
        default="land_cover_probs.tif",
        help="Per-year probabilities GeoTIFF filename per window",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    find_change(
        ds_path=args.ds_path,
        threshold=args.threshold,
        num_early=args.num_early,
        num_late=args.num_late,
        min_pixels=args.min_pixels,
        out_path=args.out_path,
        probs_filename=args.probs_filename,
        workers=args.workers,
    )
