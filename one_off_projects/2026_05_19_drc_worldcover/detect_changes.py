"""Detect land cover change between two time periods in an rslearn dataset.

Compares per-pixel softmax probabilities from multiple pre-period groups
(e.g. dec2025, jan2026, feb2026) against a post-period group (e.g. may2026).
A pixel qualifies as changed when the **minimum** source-class probability
across all pre-period months exceeds ``pre_threshold`` and the post-period
probability falls below ``post_threshold``.

The pre-period confident mask is eroded by ``erode_pixels`` before computing
change to exclude boundary artifacts at forest edges.

Emits a GeoJSON of Point features at each qualifying connected component's
centroid.

Follows the same libraries/patterns as
``rslp.change_finder.scripts.create_land_cover_change_geojson``.
"""

import argparse
import json
import logging
import multiprocessing

import numpy as np
import rasterio.features
import rasterio.transform
import shapely
import shapely.geometry
import tqdm
from rslearn.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from rslearn.utils.mp import make_pool_and_star_imap_unordered
from rslearn.utils.raster_format import GeotiffRasterFormat
from scipy import ndimage
from upath import UPath

logger = logging.getLogger(__name__)

RASTER_FORMAT = GeotiffRasterFormat()

CLASS_NAMES = [
    "bare",
    "burnt",
    "crops",
    "fallow",
    "grassland",
    "lichen_moss",
    "shrub",
    "snow_ice",
    "tree",
    "urban",
    "water",
    "wetland",
]


def _spatial_key(name: str) -> str:
    """Extract the spatial bounds prefix from a window name.

    Window names are ``{left}_{top}_{right}_{bottom}_{start}_{end}``.
    """
    return "_".join(name.split("_")[:4])


def _read_probs(window: Window, band_names: list[str]) -> np.ndarray | None:
    """Read the output probability raster for a window.

    Returns a (num_classes, H, W) float32 array, or None if the layer is not
    completed.
    """
    if not window.is_layer_completed("output"):
        return None
    raster_dir = window.get_raster_dir("output", band_names)
    raster = RASTER_FORMAT.decode_raster(raster_dir, window.projection, window.bounds)
    return raster.get_chw_array()


def _process_windows(
    pre_windows: list[Window],
    post_window: Window,
    band_names: list[str],
    pre_class_index: int,
    pre_threshold: float,
    post_threshold: float,
    erode_pixels: int,
    min_pixels: int,
    timestamp: str,
) -> list[dict]:
    """Compare multiple pre-period windows against one post window."""
    pre_probs_list: list[np.ndarray] = []
    for w in pre_windows:
        probs = _read_probs(w, band_names)
        if probs is None:
            logger.warning("Skipping %s/%s: output not available", w.group, w.name)
            return []
        pre_probs_list.append(probs)

    post_probs = _read_probs(post_window, band_names)
    if post_probs is None:
        logger.warning("Skipping %s/%s: output not available", post_window.group, post_window.name)
        return []

    # Min source-class probability across all pre-period months.
    pre_source_min = np.minimum.reduce(
        [p[pre_class_index] for p in pre_probs_list]
    )

    confident_pre = pre_source_min > pre_threshold

    # Erode to exclude forest-edge pixels.
    if erode_pixels > 0:
        struct = ndimage.generate_binary_structure(2, 1)
        confident_pre = ndimage.binary_erosion(
            confident_pre, structure=struct, iterations=erode_pixels
        )

    mask = confident_pre & (post_probs[pre_class_index] < post_threshold)

    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        return []

    sizes = ndimage.sum(mask, labels, range(1, num_features + 1))
    good_labels = np.where(sizes >= min_pixels)[0] + 1
    if len(good_labels) == 0:
        return []

    bounds = post_window.bounds
    projection = post_window.projection

    # Mean pre-period probabilities for category labeling at centroids.
    pre_probs_mean = np.mean(pre_probs_list, axis=0)

    # Build a filtered label array containing only large-enough CCs, then
    # vectorize each CC into a polygon.
    filtered = np.where(np.isin(labels, good_labels), labels, 0).astype(np.int32)
    pixel_transform = rasterio.transform.Affine(1, 0, bounds[0], 0, 1, bounds[1])

    # Map label -> list of shapely polygons (rasterio may emit multiple shapes
    # per label value if the CC has holes or complex topology).
    label_polys: dict[int, list[shapely.Geometry]] = {}
    for geom, value in rasterio.features.shapes(filtered, transform=pixel_transform):
        value = int(value)
        if value == 0:
            continue
        label_polys.setdefault(value, []).append(shapely.geometry.shape(geom))

    features: list[dict] = []
    for lbl in good_labels:
        polys = label_polys.get(int(lbl))
        if not polys:
            continue

        merged = polys[0] if len(polys) == 1 else shapely.unary_union(polys)
        if merged.is_empty:
            continue

        geom_wgs84 = STGeometry(projection, merged, time_range=None).to_projection(
            WGS84_PROJECTION
        )

        centroid = merged.centroid
        cy, cx = int(round(centroid.y - bounds[1])), int(round(centroid.x - bounds[0]))
        cy = max(0, min(cy, post_probs.shape[1] - 1))
        cx = max(0, min(cx, post_probs.shape[2] - 1))

        pre_category = CLASS_NAMES[int(pre_probs_mean[:, cy, cx].argmax())]
        post_category = CLASS_NAMES[int(post_probs[:, cy, cx].argmax())]

        features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(geom_wgs84.shp),
                "properties": {
                    "pre_category": pre_category,
                    "post_category": post_category,
                    "timestamp": timestamp,
                    "num_pixels": int(sizes[lbl - 1]),
                },
            }
        )

    return features


def detect_changes(
    ds_path: str,
    pre_groups: list[str] | None = None,
    post_group: str = "may2026",
    pre_class_index: int = 8,
    pre_threshold: float = 0.75,
    post_threshold: float = 0.25,
    erode_pixels: int = 1,
    min_pixels: int = 10,
    timestamp: str = "2026-04-01",
    out_path: str = "deforestation_events.geojson",
    workers: int = 32,
) -> None:
    if pre_groups is None:
        pre_groups = ["dec2025", "jan2026", "feb2026"]

    dataset = Dataset(UPath(ds_path))
    band_names = dataset.layers["output"].band_sets[0].bands

    # Load pre-period windows keyed by spatial location, one list per key.
    pre_by_key: dict[str, list[Window]] = {}
    for group in pre_groups:
        windows = dataset.load_windows(groups=[group], workers=128, show_progress=True)
        for w in windows:
            pre_by_key.setdefault(_spatial_key(w.name), []).append(w)

    post_windows = dataset.load_windows(groups=[post_group], workers=128, show_progress=True)

    kwargs_list = []
    for post_w in post_windows:
        key = _spatial_key(post_w.name)
        pre_ws = pre_by_key.get(key)
        if pre_ws is None or len(pre_ws) != len(pre_groups):
            continue
        kwargs_list.append(
            dict(
                pre_windows=pre_ws,
                post_window=post_w,
                band_names=band_names,
                pre_class_index=pre_class_index,
                pre_threshold=pre_threshold,
                post_threshold=post_threshold,
                erode_pixels=erode_pixels,
                min_pixels=min_pixels,
                timestamp=timestamp,
            )
        )

    all_features: list[dict] = []

    with make_pool_and_star_imap_unordered(
        workers, _process_windows, kwargs_list
    ) as results:
        for features in tqdm.tqdm(
            results, total=len(kwargs_list), desc="Detecting changes"
        ):
            all_features.extend(features)

    geojson = {"type": "FeatureCollection", "features": all_features}
    with open(out_path, "w") as f:
        json.dump(geojson, f)

    print(f"Wrote {len(all_features)} features to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Detect land cover change between two time periods"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--pre_groups",
        nargs="+",
        default=["dec2025", "jan2026", "feb2026"],
        help="Pre-period group names",
    )
    parser.add_argument("--post_group", default="may2026")
    parser.add_argument(
        "--pre_class_index",
        type=int,
        default=8,
        help="Source class index to detect loss of (default: 8 = tree)",
    )
    parser.add_argument("--pre_threshold", type=float, default=0.75)
    parser.add_argument("--post_threshold", type=float, default=0.25)
    parser.add_argument(
        "--erode_pixels",
        type=int,
        default=1,
        help="Erode the pre-period confident mask by this many pixels to exclude boundary artifacts",
    )
    parser.add_argument("--min_pixels", type=int, default=10)
    parser.add_argument("--timestamp", default="2026-04-01")
    parser.add_argument("--out_path", default="deforestation_events.geojson")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    detect_changes(
        ds_path=args.ds_path,
        pre_groups=args.pre_groups,
        post_group=args.post_group,
        pre_class_index=args.pre_class_index,
        pre_threshold=args.pre_threshold,
        post_threshold=args.post_threshold,
        erode_pixels=args.erode_pixels,
        min_pixels=args.min_pixels,
        timestamp=args.timestamp,
        out_path=args.out_path,
        workers=args.workers,
    )
