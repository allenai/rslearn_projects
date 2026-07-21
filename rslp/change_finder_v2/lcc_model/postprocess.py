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

from .timestamp_encoding import days_to_date

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
PRE_CHANGE_BANDS = [
    "pre_change_nodata",
    "pre_change_none",
    "pre_change_deforestation",
    "pre_change_urban_erosion",
    "pre_change_wetland_loss",
    "pre_change_water_contract",
    "pre_change_removed_crop_structure",
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
]
SAME_CHANGE_BANDS = [
    "same_change_nodata",
    "same_change_none",
    "same_change_agricultural_activity",
    "same_change_wildfire",
    "same_change_ice_motion",
    "same_change_flooding",
]

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
    *SAME_CHANGE_BANDS,
]


def _get_geotiff_path(window_dir: UPath) -> UPath | None:
    """Find the output_change geotiff under a window directory."""
    bandset_dir = get_bandset_dirname(OUTPUT_BANDS)
    tif = window_dir / "layers" / OUTPUT_LAYER / bandset_dir / "geotiff.tif"
    if tif.exists():
        return tif
    return None


def _component_to_feature(
    comp_mask: np.ndarray,
    change_score: np.ndarray,
    pre_days: np.ndarray,
    post_days: np.ndarray,
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
