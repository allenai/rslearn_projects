"""Convert v1 land-cover-change GeoJSON + annotations to v2 JSON format.

Takes the GeoJSON produced by ``create_land_cover_change_geojson.py``, the
sidecar annotations JSON from the viewer, and the source rslearn dataset, then
outputs a v2 annotation JSON suitable for ``change_finder_v2``.

Usage:
    python -m rslp.change_finder.land_cover_change_viewer.to_v2_json \
        --geojson land_cover_change_src_dst_sel100.geojson \
        --annotations land_cover_change_src_dst_sel100.annotations.json \
        --src-ds-path /path/to/ten_year_dataset_20260408/ \
        --out v2_annotations.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict

import numpy as np
import rasterio.features
import shapely
import shapely.geometry
from rslearn.dataset import Dataset
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from upath import UPath

BASE_YEAR = 2016
NUM_EARLY = 3
NUM_LATE = 3
MAX_NEGATIVE_POINTS = 100


def _shuffle_key(feat: dict) -> str:
    """Deterministic ordering matching the land_cover_change_viewer."""
    p = feat["properties"]
    ident = f"{p['window_group']}/{p['window_name']}/{p['src_class_id']}->{p['dst_class_id']}"
    return hashlib.md5(ident.encode()).hexdigest()


def _rasterize_geojson_geom(
    geom_wgs84: dict,
    projection: Projection,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Rasterize a WGS-84 GeoJSON geometry into the window's pixel grid.

    Returns a boolean HW mask.
    """
    shp = shapely.geometry.shape(geom_wgs84)
    if shp.is_empty:
        h = bounds[3] - bounds[1]
        w = bounds[2] - bounds[0]
        return np.zeros((h, w), dtype=bool)

    pixel_shp = (
        STGeometry(WGS84_PROJECTION, shp, time_range=None).to_projection(projection).shp
    )

    min_x, min_y, max_x, max_y = bounds
    h = max_y - min_y
    w = max_x - min_x

    clip = shapely.box(min_x, min_y, max_x, max_y)
    local = pixel_shp.intersection(clip)
    if local.is_empty:
        return np.zeros((h, w), dtype=bool)

    local = shapely.affinity.translate(local, xoff=-min_x, yoff=-min_y)
    import affine as affine_mod

    transform = affine_mod.Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    raster = rasterio.features.rasterize(
        [(local, 1)],
        out_shape=(h, w),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )
    return raster.astype(bool)


def _pixel_to_lonlat(
    row: int,
    col: int,
    projection: Projection,
    bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Convert a pixel (row, col) in window-local coords to lon/lat."""
    px_x = bounds[0] + col + 0.5
    px_y = bounds[1] + row + 0.5
    pt = shapely.geometry.Point(px_x, px_y)
    wgs84_pt = (
        STGeometry(projection, pt, time_range=None).to_projection(WGS84_PROJECTION).shp
    )
    return wgs84_pt.x, wgs84_pt.y


def _sample_points_from_mask(
    mask: np.ndarray,
    n: int,
    projection: Projection,
    bounds: tuple[int, int, int, int],
    rng: random.Random,
) -> list[dict]:
    """Sample up to n random pixels from mask, return as [{lon, lat}, ...]."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return []
    indices = list(range(len(ys)))
    if len(indices) > n:
        indices = rng.sample(indices, n)
    points = []
    for i in indices:
        lon, lat = _pixel_to_lonlat(int(ys[i]), int(xs[i]), projection, bounds)
        points.append({"lon": lon, "lat": lat})
    return points


def _month_to_date(ym: str) -> str:
    """Convert 'YYYY-MM' to 'YYYY-MM-01'."""
    return f"{ym}-01"


def convert(
    geojson_path: str,
    annotations_path: str,
    src_ds_path: str,
    out_path: str,
) -> None:
    """Run the v1 -> v2 conversion."""
    with open(geojson_path) as f:
        fc = json.load(f)
    all_features = fc["features"]
    print(f"Loaded {len(all_features)} features from {geojson_path}")

    with open(annotations_path) as f:
        annotations_list = json.load(f)
    annotations_by_idx: dict[int, dict] = {
        int(a["feature_idx"]): a for a in annotations_list
    }
    print(f"Loaded {len(annotations_by_idx)} annotations")

    # Separate change vs no_change features.
    change_features: list[dict] = []
    no_change_by_window: dict[tuple[str, str], dict] = {}
    for idx, feat in enumerate(all_features):
        props = feat["properties"]
        if props["feature_type"] == "change":
            props["feature_idx"] = idx
            change_features.append(feat)
        elif props["feature_type"] == "no_change":
            key = (props["window_group"], props["window_name"])
            no_change_by_window[key] = feat

    # Sort by the same MD5 key the viewer uses.
    change_features.sort(key=_shuffle_key)

    # Find last annotated index in sorted order.
    last_annotated_pos = -1
    for pos, feat in enumerate(change_features):
        fidx = feat["properties"]["feature_idx"]
        if fidx in annotations_by_idx:
            last_annotated_pos = pos

    # Keep: all annotated + everything after the last annotated position.
    kept_features: list[dict] = []
    for pos, feat in enumerate(change_features):
        fidx = feat["properties"]["feature_idx"]
        is_annotated = fidx in annotations_by_idx
        is_after_last = pos > last_annotated_pos
        if is_annotated or is_after_last:
            kept_features.append(feat)

    print(
        f"Keeping {len(kept_features)} features "
        f"({len(annotations_by_idx)} annotated + "
        f"{len(kept_features) - len(annotations_by_idx)} after last annotated)"
    )

    # Load source dataset windows.
    needed_keys: set[tuple[str, str]] = set()
    for feat in kept_features:
        p = feat["properties"]
        needed_keys.add((p["window_group"], p["window_name"]))

    src_dataset = Dataset(UPath(src_ds_path))
    print(f"Loading {len(needed_keys)} source windows...")
    all_src_windows = src_dataset.load_windows(
        groups=sorted({g for g, _ in needed_keys}),
        names=sorted({n for _, n in needed_keys}),
        workers=32,
        show_progress=True,
    )
    src_window_by_key = {(w.group, w.name): w for w in all_src_windows}

    # Group kept features by window, tracking the position of the first
    # feature per window so we can preserve the viewer's MD5 ordering.
    features_by_window: dict[tuple[str, str], list[dict]] = defaultdict(list)
    window_first_pos: dict[tuple[str, str], int] = {}
    for pos, feat in enumerate(kept_features):
        p = feat["properties"]
        key = (p["window_group"], p["window_name"])
        if key not in src_window_by_key:
            continue
        features_by_window[key].append(feat)
        if key not in window_first_pos:
            window_first_pos[key] = pos

    # Build v2 entries ordered by first appearance in the viewer's MD5 order.
    v2_entries: list[dict] = []
    rng = random.Random(42)

    ordered_keys = sorted(features_by_window.keys(), key=lambda k: window_first_pos[k])
    for wkey in ordered_keys:
        feats = features_by_window[wkey]
        window = src_window_by_key[wkey]
        projection = window.projection
        bounds = window.bounds

        # Compute time_range from pivot year (all features on same window share it).
        pivot_year = feats[0]["properties"]["pivot_year"]
        time_range_start = f"{pivot_year - NUM_EARLY}-01-01"
        time_range_end = f"{pivot_year + NUM_LATE + 1}-01-01"

        # Positive points: one per change feature.
        positive_points: list[dict] = []
        for feat in feats:
            p = feat["properties"]
            fidx = p["feature_idx"]
            mask = _rasterize_geojson_geom(feat["geometry"], projection, bounds)
            pts = _sample_points_from_mask(mask, 1, projection, bounds, rng)
            if not pts:
                continue
            point = pts[0]
            point["pre_category"] = p["src_class_name"]
            point["post_category"] = p["dst_class_name"]

            if fidx in annotations_by_idx:
                ann = annotations_by_idx[fidx]
                if ann.get("pre_change"):
                    point["pre_change"] = _month_to_date(ann["pre_change"])
                if ann.get("post_change"):
                    point["post_change"] = _month_to_date(ann["post_change"])
            else:
                point["pre_change"] = f"{pivot_year}-01-01"

            positive_points.append(point)

        # Negative points from no_change polygon.
        negative_points: list[dict] = []
        nc_feat = no_change_by_window.get(wkey)
        if nc_feat is not None:
            nc_mask = _rasterize_geojson_geom(nc_feat["geometry"], projection, bounds)
            negative_points = _sample_points_from_mask(
                nc_mask, MAX_NEGATIVE_POINTS, projection, bounds, rng
            )

        v2_entries.append(
            {
                "projection": projection.serialize(),
                "bounds": list(bounds),
                "time_range": [time_range_start, time_range_end],
                "positive_points": positive_points,
                "negative_points": negative_points,
                "window_name": window.name,
                "group": window.group,
            }
        )

    with open(out_path, "w") as f:
        json.dump(v2_entries, f, indent=2)
    print(f"Wrote {len(v2_entries)} entries to {out_path}")


def main() -> None:
    """Convert v1 land-cover-change GeoJSON + annotations to v2 JSON."""
    parser = argparse.ArgumentParser(
        description="Convert v1 land-cover-change GeoJSON + annotations to v2 JSON."
    )
    parser.add_argument(
        "--geojson",
        required=True,
        help="Path to the v1 polygons GeoJSON.",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to the v1 annotations sidecar JSON.",
    )
    parser.add_argument(
        "--src-ds-path",
        required=True,
        help="Path to the source rslearn dataset (for window projection/bounds).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for the v2 JSON.",
    )
    args = parser.parse_args()
    convert(
        geojson_path=args.geojson,
        annotations_path=args.annotations,
        src_ds_path=args.src_ds_path,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
