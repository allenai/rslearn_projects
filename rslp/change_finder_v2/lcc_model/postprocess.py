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
from datetime import datetime

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
    GeotiffRasterFormat,
    get_bandset_dirname,
    get_raster_projection_and_bounds,
)
from scipy import ndimage
from upath import UPath

BINARY_CHANGE_BAND = 2
SRC_BAND_OFFSET = 3
DST_BAND_OFFSET = 16
TS_BAND_OFFSET = 29
NUM_LC_CLASSES = 13
NUM_TIMESTAMPS = 20

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
    *(f"ts_{i}" for i in range(NUM_TIMESTAMPS)),
]

SENTINEL2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]
SENTINEL2_BANDSET_DIR = get_bandset_dirname(SENTINEL2_BANDS)

NUM_QUARTERLY = 16
NUM_FREQUENT = 4


def _get_geotiff_path(window_dir: UPath) -> UPath | None:
    """Find the output_change geotiff under a window directory."""
    bandset_dir = get_bandset_dirname(OUTPUT_BANDS)
    tif = window_dir / "layers" / OUTPUT_LAYER / bandset_dir / "geotiff.tif"
    if tif.exists():
        return tif
    return None


def _load_timestamps(window_dir: UPath) -> list[tuple[datetime, datetime]] | None:
    """Load the 20 timestamps (16 quarterly + 4 frequent) for a window.

    Reads metadata.json sidecars from the per-item-group layer directories,
    replicating the ordering that PredictPassBuilder uses: last 16 quarterly
    timestamps followed by 4 frequent timestamps.

    Returns None if timestamps cannot be determined.
    """
    quarterly_ts: list[tuple[datetime, datetime]] = []
    group_idx = 0
    while True:
        folder = (
            "sentinel2_quarterly"
            if group_idx == 0
            else f"sentinel2_quarterly.{group_idx}"
        )
        meta_dir = window_dir / "layers" / folder / SENTINEL2_BANDSET_DIR
        meta = GeotiffRasterFormat.decode_metadata(meta_dir)
        if meta is None:
            break
        if meta.timestamps:
            quarterly_ts.extend(meta.timestamps)
        group_idx += 1

    frequent_ts: list[tuple[datetime, datetime]] = []
    group_idx = 0
    while True:
        folder = (
            "sentinel2_frequent_0"
            if group_idx == 0
            else f"sentinel2_frequent_0.{group_idx}"
        )
        meta_dir = window_dir / "layers" / folder / SENTINEL2_BANDSET_DIR
        meta = GeotiffRasterFormat.decode_metadata(meta_dir)
        if meta is None:
            break
        if meta.timestamps:
            frequent_ts.extend(meta.timestamps)
        group_idx += 1

    if not quarterly_ts and not frequent_ts:
        return None

    # Take last NUM_QUARTERLY quarterlies (pad if needed), then all frequent.
    if len(quarterly_ts) > NUM_QUARTERLY:
        quarterly_ts = quarterly_ts[-NUM_QUARTERLY:]
    elif len(quarterly_ts) < NUM_QUARTERLY:
        pad = quarterly_ts[0] if quarterly_ts else (datetime.min, datetime.min)
        quarterly_ts = [pad] * (NUM_QUARTERLY - len(quarterly_ts)) + quarterly_ts

    return quarterly_ts + frequent_ts


def _component_to_feature(
    comp_mask: np.ndarray,
    change_score: np.ndarray,
    ts_probs: np.ndarray,
    src_id: int,
    dst_id: int,
    projection: object,
    bounds: tuple[int, int, int, int],
    timestamps: list[tuple[datetime, datetime]] | None,
) -> dict | None:
    """Vectorize a single connected component and build a GeoJSON feature dict."""
    num_pixels = int(comp_mask.sum())
    avg_score = float(change_score[comp_mask].mean())
    col0, row0 = bounds[0], bounds[1]

    # Timestamp: per-pixel argmax then majority vote.
    pixel_ts_indices = ts_probs[:, comp_mask].argmax(axis=0)  # (num_pixels,)
    counts = np.bincount(pixel_ts_indices, minlength=NUM_TIMESTAMPS)
    ts_idx = int(counts.argmax())

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
        "timestamp_idx": ts_idx,
    }

    if timestamps and ts_idx < len(timestamps):
        ts_start, ts_end = timestamps[ts_idx]
        props["timestamp_start"] = ts_start.isoformat()
        props["timestamp_end"] = ts_end.isoformat()

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
    ts_probs = arr[TS_BAND_OFFSET : TS_BAND_OFFSET + NUM_TIMESTAMPS]

    # Per-pixel argmax class (skip class 0 = nodata by taking argmax over 1..12
    # and adding 1).
    src_class = src_probs[1:].argmax(axis=0) + 1  # (H, W)
    dst_class = dst_probs[1:].argmax(axis=0) + 1  # (H, W)

    timestamps = _load_timestamps(window_dir)

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
                ts_probs,
                s_id,
                d_id,
                projection,
                bounds,
                timestamps,
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
