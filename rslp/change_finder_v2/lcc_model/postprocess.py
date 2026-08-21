"""Convert v2 LCC prediction rasters to GeoJSON change polygons.

Reads the ``output_change`` layer from each prediction window. For each window:

1. Threshold the binary change probability band.
2. Compute per-pixel argmax source and destination land cover classes.
3. For each unique (src, dst) class pair, find connected components and
   vectorize them to WGS-84 polygons.
4. Estimate the change timestamp per polygon via majority vote of per-pixel
   argmax over the 20 timestamp probability bands, then map to actual dates
   read from the dataset's layer metadata.

Usage::

    python -m rslp.change_finder_v2.lcc_model.postprocess \
        --dataset_path /path/to/predict_dataset \
        --output geojson_out.geojson \
        --threshold 128
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import multiprocessing.pool
from collections.abc import Iterable

import numpy as np
import rasterio
import rasterio.features
import shapely
import shapely.affinity
import shapely.geometry
import shapely.ops
import tqdm
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from rslearn.utils.raster_format import (
    get_bandset_dirname,
    get_raster_projection_and_bounds,
)
from scipy import ndimage
from upath import UPath

from .timestamp_encoding import TIMESTAMP_EPOCH, days_to_date

BINARY_CHANGE_BAND = 2
SRC_BAND_OFFSET = 3
DST_BAND_OFFSET = 16
# The timestamp section is two bands holding the predicted pre/post change dates
# encoded as integer days since TIMESTAMP_EPOCH (see timestamp_encoding.py).
TS_PRE_DAYS_BAND = 29
TS_POST_DAYS_BAND = 30
NUM_LC_CLASSES = 13

LC_CLASS_NAMES = [
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

# Change-category output bands, in the same order as the model output and the
# output_change layer in config.json (class layout [nodata, none, <options...>]).
# The pre_change head is the merged pre+same head: the former same_change
# categories are appended after the pre categories (see
# SinglePassMultiTask merge_same_into_pre).
PRE_CHANGE_BANDS = [
    "pre_change_nodata",
    "pre_change_none",
    "pre_change_deforestation",
    "pre_change_urban_erosion",
    "pre_change_wetland_loss",
    "pre_change_water_contract",
    "pre_change_removed_crop_structure",
    "pre_change_agricultural_activity",
    "pre_change_wildfire",
    "pre_change_ice_motion",
    "pre_change_flooding",
]
POST_CHANGE_BANDS = [
    "post_change_nodata",
    "post_change_none",
    "post_change_vegetation_growth",
    "post_change_new_building",
    "post_change_new_road",
    "post_change_new_infrastructure",
    "post_change_new_crop_field",
    "post_change_new_aquafarm",
    "post_change_site_clearing",
    "post_change_water_expand",
    "post_change_mining",
    "post_change_new_crop_structure",
    "post_change_selective_logging",
    "post_change_landslide",
    "post_change_settlement",
]

# Band offsets of the two change-category sections within the output raster.
PRE_CHANGE_BAND_OFFSET = TS_POST_DAYS_BAND + 1
POST_CHANGE_BAND_OFFSET = PRE_CHANGE_BAND_OFFSET + len(PRE_CHANGE_BANDS)

# Plain category names (class layout [nodata, none, <options...>]).
PRE_CHANGE_CATEGORY_NAMES = [b.removeprefix("pre_change_") for b in PRE_CHANGE_BANDS]
POST_CHANGE_CATEGORY_NAMES = [b.removeprefix("post_change_") for b in POST_CHANGE_BANDS]

# Class index within the merged pre+same head where the former same_change
# categories start (they are appended after the original pre categories).
MERGED_SAME_CATEGORY_START = PRE_CHANGE_CATEGORY_NAMES.index("agricultural_activity")

OUTPUT_LAYER = "output_change"
OUTPUT_BANDS = [
    "binary_nodata",
    "binary_no_change",
    "binary_change",
    *(
        f"src_{LC_CLASS_NAMES[i].split('/')[0].split(' ')[0].lower()}"
        for i in range(NUM_LC_CLASSES)
    ),
    *(
        f"dst_{LC_CLASS_NAMES[i].split('/')[0].split(' ')[0].lower()}"
        for i in range(NUM_LC_CLASSES)
    ),
    "ts_pre_days",
    "ts_post_days",
    *PRE_CHANGE_BANDS,
    *POST_CHANGE_BANDS,
]

# Compact uint8 summary raster layout (see summary_window_array). The class
# bands hold the per-head argmax class index (0 = no prediction; argmax skips
# the nodata class), the score bands hold the probability (0-255) of the
# argmax class, and the month bands hold 0 for no prediction or 1 + whole
# calendar months since TIMESTAMP_EPOCH (Jan 2015; 255 reaches March 2036).
# This layout matches the olmoearth_lcc_viewer COGs exactly, so its
# make_cogs.py converts summary rasters without any band changes.
SUMMARY_BANDS = [
    "binary_change",
    "pre_class",
    "post_class",
    "src_class",
    "dst_class",
    "pre_score",
    "post_score",
    "ts_pre_month",
    "ts_post_month",
]


def _days_to_month_values(days: np.ndarray) -> np.ndarray:
    """Convert day-since-epoch values to the summary month encoding.

    The result is 1 + whole calendar months between TIMESTAMP_EPOCH and the
    day's date, clipped to the uint8 range.
    """
    epoch_day = np.datetime64(TIMESTAMP_EPOCH.date().isoformat())
    dates = epoch_day + days.astype("timedelta64[D]")
    months = (dates.astype("datetime64[M]") - epoch_day.astype("datetime64[M]")).astype(
        np.int64
    ) + 1
    return np.clip(months, 1, 255).astype(np.uint8)


def summary_window_array(arr: np.ndarray) -> np.ndarray:
    """Derive the uint8 summary bands from a full output window array.

    Args:
        arr: (len(OUTPUT_BANDS), H, W) array with probabilities scaled 0-255.

    Returns:
        (len(SUMMARY_BANDS), H, W) uint8 array.
    """
    # Pixels with no prediction were left all-zero; use the binary head to
    # detect them.
    valid = arr[0 : BINARY_CHANGE_BAND + 1].max(axis=0) > 0

    head_probs = {
        "pre": arr[
            PRE_CHANGE_BAND_OFFSET : PRE_CHANGE_BAND_OFFSET + len(PRE_CHANGE_BANDS)
        ],
        "post": arr[
            POST_CHANGE_BAND_OFFSET : POST_CHANGE_BAND_OFFSET + len(POST_CHANGE_BANDS)
        ],
        "src": arr[SRC_BAND_OFFSET : SRC_BAND_OFFSET + NUM_LC_CLASSES],
        "dst": arr[DST_BAND_OFFSET : DST_BAND_OFFSET + NUM_LC_CLASSES],
    }
    classes: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for head, probs in head_probs.items():
        # Argmax excluding the nodata class (index 0); 0 is reserved for
        # pixels with no prediction.
        argmax = probs[1:].argmax(axis=0) + 1
        scores[head] = np.clip(
            np.take_along_axis(probs, argmax[None], axis=0)[0], 0, 255
        ).astype(np.uint8)
        classes[head] = np.where(valid, argmax, 0).astype(np.uint8)

    months = {}
    for head, band in [("pre", TS_PRE_DAYS_BAND), ("post", TS_POST_DAYS_BAND)]:
        months[head] = np.where(valid, _days_to_month_values(arr[band]), 0).astype(
            np.uint8
        )

    return np.stack(
        [
            np.clip(arr[BINARY_CHANGE_BAND], 0, 255).astype(np.uint8),
            classes["pre"],
            classes["post"],
            classes["src"],
            classes["dst"],
            scores["pre"],
            scores["post"],
            months["pre"],
            months["post"],
        ]
    )


def _get_geotiff_path(window_dir: UPath) -> UPath | None:
    """Find the output_change geotiff under a window directory."""
    bandset_dir = get_bandset_dirname(OUTPUT_BANDS)
    tif = window_dir / "layers" / OUTPUT_LAYER / bandset_dir / "geotiff.tif"
    if tif.exists():
        return tif
    return None


def _majority_class(class_map: np.ndarray, mask: np.ndarray) -> int:
    """Return the most common class value within the mask."""
    return int(np.bincount(class_map[mask]).argmax())


def _component_to_feature(
    comp_mask: np.ndarray,
    change_score: np.ndarray,
    pre_days: np.ndarray,
    post_days: np.ndarray,
    pre_cat_class: np.ndarray,
    post_cat_class: np.ndarray,
    src_id: int,
    dst_id: int,
    projection: object,
    bounds: tuple[int, int, int, int],
) -> dict | None:
    """Vectorize a single connected component and build a GeoJSON feature dict."""
    num_pixels = int(comp_mask.sum())
    avg_score = float(change_score[comp_mask].mean())
    col0, row0 = bounds[0], bounds[1]

    # Change dates: median over the component of the per-pixel pre/post day bands
    # (days since TIMESTAMP_EPOCH), converted back to real dates.
    pre_day = int(np.median(pre_days[comp_mask]))
    post_day = int(np.median(post_days[comp_mask]))

    shapes = list(
        rasterio.features.shapes(
            comp_mask.astype(np.uint8),
            mask=comp_mask,
            connectivity=8,
        )
    )
    if not shapes:
        return None

    polys = []
    for geom, _ in shapes:
        shp = shapely.geometry.shape(geom)
        shp = shapely.affinity.translate(shp, xoff=col0, yoff=row0)
        polys.append(shp)

    merged = shapely.ops.unary_union(polys)
    if merged.is_empty:
        return None

    geom_wgs84 = STGeometry(projection, merged, None).to_projection(WGS84_PROJECTION)

    props: dict = {
        "num_pixels": num_pixels,
        "avg_change_score": round(avg_score, 2),
        "src_class": LC_CLASS_NAMES[src_id],
        "src_class_idx": src_id,
        "dst_class": LC_CLASS_NAMES[dst_id],
        "dst_class_idx": dst_id,
        "pre_change_days": pre_day,
        "post_change_days": post_day,
        "pre_change_date": days_to_date(pre_day).isoformat(),
        "post_change_date": days_to_date(post_day).isoformat(),
        # Predicted change categories: majority vote over the component of the
        # per-pixel argmax class ("none" is a valid prediction). The pre head is
        # the merged pre+same head, so pre_change_category may also be one of
        # the former same_change categories.
        "pre_change_category": PRE_CHANGE_CATEGORY_NAMES[
            _majority_class(pre_cat_class, comp_mask)
        ],
        "post_change_category": POST_CHANGE_CATEGORY_NAMES[
            _majority_class(post_cat_class, comp_mask)
        ],
    }

    return {
        "type": "Feature",
        "geometry": shapely.geometry.mapping(geom_wgs84.shp),
        "properties": props,
    }


def process_window(
    window_dir: UPath,
    threshold: int,
    min_pixels: int,
) -> list[dict]:
    """Process one prediction window and return GeoJSON-ready feature dicts."""
    tif_path = _get_geotiff_path(window_dir)
    if tif_path is None:
        return []

    with rasterio.open(tif_path) as src:
        arr = src.read()
        projection, bounds = get_raster_projection_and_bounds(src)

    change_score = arr[BINARY_CHANGE_BAND]
    change_mask = change_score >= threshold
    if not change_mask.any():
        return []

    src_probs = arr[SRC_BAND_OFFSET : SRC_BAND_OFFSET + NUM_LC_CLASSES]
    dst_probs = arr[DST_BAND_OFFSET : DST_BAND_OFFSET + NUM_LC_CLASSES]
    pre_days = arr[TS_PRE_DAYS_BAND]
    post_days = arr[TS_POST_DAYS_BAND]

    # Per-pixel argmax class (skip class 0 = nodata by taking argmax over 1..12
    # and adding 1).
    src_class = src_probs[1:].argmax(axis=0) + 1  # (H, W)
    dst_class = dst_probs[1:].argmax(axis=0) + 1  # (H, W)

    # Per-pixel argmax change category (skip class 0 = nodata; class 1 = "none"
    # is a valid prediction).
    pre_cat_probs = arr[
        PRE_CHANGE_BAND_OFFSET : PRE_CHANGE_BAND_OFFSET + len(PRE_CHANGE_BANDS)
    ]
    post_cat_probs = arr[
        POST_CHANGE_BAND_OFFSET : POST_CHANGE_BAND_OFFSET + len(POST_CHANGE_BANDS)
    ]
    pre_cat_class = pre_cat_probs[1:].argmax(axis=0) + 1  # (H, W)
    post_cat_class = post_cat_probs[1:].argmax(axis=0) + 1  # (H, W)

    features: list[dict] = []

    # Build a combined label image for joint (src, dst) segmentation.
    # Encode as src_id * NUM_LC_CLASSES + dst_id so each unique pair gets a
    # unique integer, then iterate over unique pairs.
    pair_labels = src_class.astype(np.int32) * NUM_LC_CLASSES + dst_class.astype(
        np.int32
    )
    pair_labels[~change_mask] = -1

    for pair_val in np.unique(pair_labels):
        if pair_val < 0:
            continue
        s_id = int(pair_val // NUM_LC_CLASSES)
        d_id = int(pair_val % NUM_LC_CLASSES)
        if s_id == d_id:
            continue

        pair_mask = pair_labels == pair_val
        labels, num_components = ndimage.label(pair_mask)

        for comp_id in range(1, num_components + 1):
            comp_mask = labels == comp_id
            if comp_mask.sum() < min_pixels:
                continue

            feat = _component_to_feature(
                comp_mask,
                change_score,
                pre_days,
                post_days,
                pre_cat_class,
                post_cat_class,
                s_id,
                d_id,
                projection,
                bounds,
            )
            if feat is not None:
                features.append(feat)

    return features


def _process_window_star(kwargs: dict) -> list[dict]:
    return process_window(**kwargs)


def collect_features(
    dataset_path: str,
    threshold: int = 128,
    min_pixels: int = 10,
    workers: int = 32,
) -> list[dict]:
    """Scan all predict windows and return GeoJSON feature dicts (WGS84).

    Returns an empty list if the predict group has no windows.
    """
    ds_root = UPath(dataset_path)
    predict_dir = ds_root / "windows" / "predict"

    if not predict_dir.exists():
        return []

    kwargs_list = [
        dict(window_dir=window_dir, threshold=threshold, min_pixels=min_pixels)
        for window_dir in sorted(predict_dir.iterdir())
        if window_dir.is_dir()
    ]

    all_features: list[dict] = []

    pool: multiprocessing.pool.Pool | None = None
    results: Iterable[list[dict]]
    if workers <= 0:
        results = map(_process_window_star, kwargs_list)
    else:
        pool = multiprocessing.Pool(workers)
        results = pool.imap_unordered(_process_window_star, kwargs_list)

    try:
        for features in tqdm.tqdm(results, total=len(kwargs_list), desc="Processing"):
            all_features.extend(features)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return all_features


def create_geojson(
    dataset_path: str,
    output: str,
    threshold: int = 128,
    min_pixels: int = 10,
    workers: int = 32,
) -> None:
    """Scan all predict windows and write a GeoJSON FeatureCollection."""
    all_features = collect_features(
        dataset_path=dataset_path,
        threshold=threshold,
        min_pixels=min_pixels,
        workers=workers,
    )

    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    out_path = UPath(output)
    with out_path.open("w") as f:
        json.dump(geojson, f)

    print(f"Wrote {len(all_features)} features to {output}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Convert v2 LCC prediction rasters to GeoJSON change polygons."
    )
    parser.add_argument(
        "--dataset_path", required=True, help="Root of the prediction dataset."
    )
    parser.add_argument("--output", required=True, help="Output GeoJSON file path.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Binary change probability threshold (0-255).",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=10,
        help="Minimum pixels for a connected component to be included.",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    create_geojson(
        dataset_path=args.dataset_path,
        output=args.output,
        threshold=args.threshold,
        min_pixels=args.min_pixels,
        workers=args.workers,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
