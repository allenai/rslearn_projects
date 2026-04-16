"""Create GeoJSON of land cover change events from precomputed change GeoTIFFs.

For each window, iterates over all pairs of (source, destination) classes.  A
pixel qualifies for a pair when the source class probability exceeds the
threshold at every early timestep and the destination class probability
exceeds the threshold at every late timestep.  All connected components with
at least ``min_pixels`` pixels are combined into a single MultiPolygon and
emitted as one GeoJSON feature per (window, src, dst) with
``feature_type`` = ``"change"``.

In addition, for each window, a ``"no_change"`` feature is emitted summarizing
pixels where the model is confident in the *same* class across every early
and late timestep.  These features are intended for visualization/label
augmentation only (they don't carry src/dst properties).
"""

import argparse
import json
import multiprocessing
import multiprocessing.pool
from collections.abc import Iterable

import numpy as np
import rasterio.features
import shapely
import shapely.geometry
import tqdm
from rslearn.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from scipy import ndimage
from upath import UPath

RASTER_FORMAT = GeotiffRasterFormat()

NUM_CLASSES = 13
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


def _class_probs(
    change_data: np.ndarray,
    class_id: int,
    num_early: int,
    num_late: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return (early_probs, late_probs) for the given class as float32 arrays."""
    header_bands = 3
    early_probs = []
    for i in range(num_early):
        band_idx = header_bands + i * NUM_CLASSES + class_id
        early_probs.append(change_data[band_idx].astype(np.float32) / 255.0)

    late_probs = []
    for i in range(num_late):
        band_idx = header_bands + num_early * NUM_CLASSES + i * NUM_CLASSES + class_id
        late_probs.append(change_data[band_idx].astype(np.float32) / 255.0)

    return early_probs, late_probs


def _mask_to_wgs84_multipolygon(
    mask: np.ndarray,
    projection: Projection,
    bounds: tuple[int, int, int, int],
    min_pixels: int,
) -> tuple[dict | None, int]:
    """Return (geojson geometry, total pixel count) for CCs >= ``min_pixels``.

    Filters the mask to only connected components that meet ``min_pixels``,
    polygonizes them (pixel space offset by window bounds), and reprojects to
    WGS-84 via :class:`STGeometry`.
    """
    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        return None, 0
    sizes = ndimage.sum(mask, labels, range(1, num_features + 1))
    good_labels = np.where(sizes >= min_pixels)[0] + 1
    if len(good_labels) == 0:
        return None, 0

    filtered = np.isin(labels, good_labels)
    total_pixels = int(filtered.sum())

    pixel_transform = rasterio.transform.Affine(1, 0, bounds[0], 0, 1, bounds[1])
    filtered_u8 = filtered.astype(np.uint8)
    shape_list = list(
        rasterio.features.shapes(
            filtered_u8, mask=filtered_u8, transform=pixel_transform
        )
    )
    polys = [shapely.geometry.shape(geom) for geom, val in shape_list if val == 1]
    if not polys:
        return None, 0

    # Disjoint CCs; unary_union produces a clean (Multi)Polygon.
    merged = polys[0] if len(polys) == 1 else shapely.unary_union(polys)
    if merged.is_empty:
        return None, 0

    geom_wgs84 = STGeometry(projection, merged, time_range=None).to_projection(
        WGS84_PROJECTION
    )
    return shapely.geometry.mapping(geom_wgs84.shp), total_pixels


def _process_window(
    window: Window,
    threshold: float,
    num_early: int,
    num_late: int,
    min_pixels: int,
    change_filename: str,
) -> list[dict]:
    """Process one window, returning a list of GeoJSON feature dicts."""
    window_root = window.storage.get_window_root(window.group, window.name)
    if not (window_root / change_filename).exists():
        return []

    change_raster: RasterArray = RASTER_FORMAT.decode_raster(
        window_root, window.projection, window.bounds, fname=change_filename
    )
    change_data = change_raster.get_chw_array()
    h, w = change_data.shape[1], change_data.shape[2]

    # Precompute per-class "confident early" and "confident late" masks so we
    # don't recompute them for every (src, dst) pair.
    early_confident: list[np.ndarray] = []
    late_confident: list[np.ndarray] = []
    for class_id in range(NUM_CLASSES):
        early_probs, late_probs = _class_probs(
            change_data, class_id, num_early, num_late
        )
        early_mask = np.ones((h, w), dtype=bool)
        for p in early_probs:
            early_mask &= p > threshold
        late_mask = np.ones((h, w), dtype=bool)
        for p in late_probs:
            late_mask &= p > threshold
        early_confident.append(early_mask)
        late_confident.append(late_mask)

    features: list[dict] = []

    # Change features: one per (src, dst) pair with at least one qualifying CC.
    for src_id in range(NUM_CLASSES):
        for dst_id in range(NUM_CLASSES):
            if src_id == dst_id:
                continue

            mask = early_confident[src_id] & late_confident[dst_id]
            if mask.sum() < min_pixels:
                continue

            geom, num_pixels = _mask_to_wgs84_multipolygon(
                mask, window.projection, window.bounds, min_pixels
            )
            if geom is None:
                continue

            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "feature_type": "change",
                        "src_class_id": src_id,
                        "src_class_name": CLASS_NAMES[src_id],
                        "dst_class_id": dst_id,
                        "dst_class_name": CLASS_NAMES[dst_id],
                        "window_group": window.group,
                        "window_name": window.name,
                        "num_pixels": num_pixels,
                    },
                }
            )

    # No-change feature: only emit if at least one change feature qualified
    # for this window (otherwise there's nothing to visualize against).
    if not features:
        return features

    # Pixels that are confidently the *same* class across every early and
    # late timestep.  Skip class 0 (nodata) since nodata regions aren't
    # meaningfully "unchanged".
    no_change_mask = np.zeros((h, w), dtype=bool)
    for class_id in range(1, NUM_CLASSES):
        no_change_mask |= early_confident[class_id] & late_confident[class_id]

    if no_change_mask.sum() >= min_pixels:
        geom, num_pixels = _mask_to_wgs84_multipolygon(
            no_change_mask, window.projection, window.bounds, min_pixels
        )
        if geom is not None:
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "feature_type": "no_change",
                        "window_group": window.group,
                        "window_name": window.name,
                        "num_pixels": num_pixels,
                    },
                }
            )

    return features


def _process_window_star(kwargs: dict) -> list[dict]:
    return _process_window(**kwargs)


def create_geojson(
    ds_path: str,
    threshold: float = 0.75,
    num_early: int = 3,
    num_late: int = 3,
    min_pixels: int = 10,
    out_path: str = "land_cover_change.geojson",
    change_filename: str = "land_cover_change.tif",
    workers: int = 32,
) -> None:
    """Scan all windows and write a GeoJSON FeatureCollection of change events."""
    dataset = Dataset(UPath(ds_path))
    windows = dataset.load_windows(workers=128, show_progress=True)

    kwargs_list = [
        dict(
            window=window,
            threshold=threshold,
            num_early=num_early,
            num_late=num_late,
            min_pixels=min_pixels,
            change_filename=change_filename,
        )
        for window in windows
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
        for features in tqdm.tqdm(
            results, total=len(kwargs_list), desc="Scanning windows"
        ):
            all_features.extend(features)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    with open(out_path, "w") as f:
        json.dump(geojson, f)

    print(f"Wrote {len(all_features)} features to {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Create GeoJSON of land cover change events"
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
        default="land_cover_change.geojson",
        help="Output GeoJSON path",
    )
    parser.add_argument(
        "--change_filename",
        default="land_cover_change.tif",
        help="Change GeoTIFF filename per window",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    create_geojson(
        ds_path=args.ds_path,
        threshold=args.threshold,
        num_early=args.num_early,
        num_late=args.num_late,
        min_pixels=args.min_pixels,
        out_path=args.out_path,
        change_filename=args.change_filename,
        workers=args.workers,
    )
