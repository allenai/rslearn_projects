"""Build per-AOI agriculture annotation sets from Studio predictions.

This is step 4 of the monocrop initial setup. Given the Studio prediction results (one
``result_batch{i}.geojson`` per batch, with point geometries and a ``new_label``
property) and the original per-batch polygon GeoJSONs, this script:

1. Restores polygon geometry: each Studio result point is matched back to its original
   polygon by nearest centroid (the simplification step replaced polygons with their
   centroids, so the match is exact). The label/probabilities are copied onto the
   polygon. Optionally writes per-batch ``merged_batch{i}.geojson``. If ``--regen`` is
   given (a non-smoothed prediction_request_geometry.geojson), each event's polygon
   geometry is replaced by the matching one from that file (matched by
   ``(tif_fname, center_pixel)``); unmatched events keep their smoothed geometry. All
   other attributes still come from the per-batch orig files.
2. For each AOI (a standard GeoJSON or Esri JSON polygon file), keeps agriculture
   polygons whose centroid falls inside the AOI, and writes two files:
     - ``{AOI}_agriculture.geojson``: up to ``--sample-agriculture`` (default 2000)
       polygons.
     - ``{AOI}_agriculture_large.geojson``: up to ``--sample-large`` (default 1000)
       polygons whose area exceeds ``--min-hectares`` (default 5). This set is drawn
       independently and may overlap ``{AOI}_agriculture.geojson``. Area is computed by
       reprojecting to the local UTM/UPS projection with a 100 m pixel size, so the
       projected area equals hectares directly (same approach as
       add_area_to_studio_tasks.py).

Run in an environment with rslearn, shapely, scipy, and numpy (e.g. the rslearn venv):

    python rslp/forest_loss_driver/scripts/monocrop_initial_setup_20260624/make_annotation_sets.py \
        --predictions-dir /path/to/studio_results \
        --orig-dir /weka/.../predictions_2022_to_2025 \
        --aoi Oilpalmecuador=/path/Oilpalmecuador.json \
        --aoi Oilpalmperu=/path/Oilpalmperu.json \
        --aoi Soybeanbolivia=/path/Soybeanbolivia.json \
        --output-dir /path/to/output \
        --write-merged
"""

import argparse
import glob
import json
import os
import random
import re
from multiprocessing import Pool

import numpy as np
import shapely.geometry
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from scipy.spatial import cKDTree


def load_aoi_polygon(path: str) -> shapely.geometry.base.BaseGeometry:
    """Load an AOI polygon file as a single unioned shapely geometry.

    Supports both standard GeoJSON (Polygon/MultiPolygon features with ``coordinates``)
    and Esri JSON (features whose geometry has ``rings``).
    """
    with open(path) as f:
        data = json.load(f)
    geoms = []
    for feat in data["features"]:
        geom = feat["geometry"]
        if "rings" in geom:
            shp = shapely.geometry.shape(
                {"type": "Polygon", "coordinates": geom["rings"]}
            )
        else:
            shp = shapely.geometry.shape(geom)
        geoms.append(shp)
    geom = geoms[0]
    for g in geoms[1:]:
        geom = geom.union(g)
    return geom


def feature_key(feat: dict) -> tuple:
    """Unique identity of a forest loss event (tif_fname + center_pixel)."""
    p = feat["properties"]
    cp = p.get("center_pixel")
    return (p.get("tif_fname"), tuple(cp) if cp is not None else None)


def build_regen_index(path: str) -> dict[tuple, dict]:
    """Map each event identity to its (non-smoothed) geometry from the regen GeoJSON."""
    with open(path) as f:
        features = json.load(f)["features"]
    return {feature_key(f): f["geometry"] for f in features}


def compute_hectares(shp: shapely.geometry.base.BaseGeometry) -> float:
    """Area of a WGS84 polygon in hectares, via local UTM/UPS reprojection."""
    wgs84_geom = STGeometry(WGS84_PROJECTION, shp, None)
    dst_proj = get_utm_ups_projection(shp.centroid.x, shp.centroid.y, 100, -100)
    return abs(wgs84_geom.to_projection(dst_proj).shp.area)


# Per-worker globals, populated by _init_worker.
_CENTROIDS: list[tuple[float, float]] | None = None
_SHAPES: list[shapely.geometry.base.BaseGeometry] | None = None
_AOIS: dict[str, shapely.geometry.base.BaseGeometry] | None = None


def _init_worker(
    centroids: list[tuple[float, float]],
    shapes: list[shapely.geometry.base.BaseGeometry],
    aois: dict[str, shapely.geometry.base.BaseGeometry],
) -> None:
    """Initialize per-worker globals shared across tasks."""
    global _CENTROIDS, _SHAPES, _AOIS
    _CENTROIDS = centroids
    _SHAPES = shapes
    _AOIS = aois


def _match_chunk(arg: tuple[str, list[int]]) -> list[int]:
    """Return target indices in the chunk whose centroid is inside the AOI."""
    assert _AOIS is not None and _CENTROIDS is not None
    aoi_name, idx_chunk = arg
    aoi = _AOIS[aoi_name]
    minx, miny, maxx, maxy = aoi.bounds
    out = []
    for i in idx_chunk:
        x, y = _CENTROIDS[i]
        if (
            minx <= x <= maxx
            and miny <= y <= maxy
            and aoi.contains(shapely.geometry.Point(x, y))
        ):
            out.append(i)
    return out


def _area_chunk(idx_chunk: list[int]) -> list[tuple[int, float]]:
    """Return (index, hectares) for each target index in the chunk."""
    assert _SHAPES is not None
    return [(i, compute_hectares(_SHAPES[i])) for i in idx_chunk]


def _chunked(seq: list[int], size: int) -> list[list[int]]:
    """Split a list into chunks of at most ``size`` items."""
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def merge_batch(
    orig_path: str, result_path: str, regen_index: dict[tuple, dict] | None = None
) -> list[dict]:
    """Attach new_label/probs from Studio result points onto original polygons.

    If ``regen_index`` is given, replace each polygon's geometry with the matching
    (non-smoothed) geometry, keyed by ``(tif_fname, center_pixel)``. Events without a
    match keep their original (smoothed) geometry.
    """
    with open(orig_path) as f:
        orig = json.load(f)["features"]
    with open(result_path) as f:
        res = json.load(f)["features"]

    res_pts = np.array([f["geometry"]["coordinates"][:2] for f in res])
    tree = cKDTree(res_pts)
    orig_centroids = np.array(
        [shapely.geometry.shape(f["geometry"]).centroid.coords[0] for f in orig]
    )
    dists, idxs = tree.query(orig_centroids, k=1)

    merged = []
    replaced = 0
    for o, ri in zip(orig, idxs):
        res_props = res[ri]["properties"]
        o = dict(o)
        o["properties"] = dict(o["properties"])
        o["properties"]["new_label"] = res_props.get("new_label")
        o["properties"]["probs"] = res_props.get("probs")
        if regen_index is not None:
            regen_geom = regen_index.get(feature_key(o))
            if regen_geom is not None:
                o["geometry"] = regen_geom
                replaced += 1
        merged.append(o)
    msg = (
        f"  {os.path.basename(orig_path)}: labeled {len(merged)} polygons "
        f"(max match dist {dists.max():.3e} deg)"
    )
    if regen_index is not None:
        msg += f"; regen geometry for {replaced}/{len(merged)}"
    print(msg)
    return merged


def write_geojson(path: str, features: list[dict]) -> None:
    """Write a FeatureCollection to path."""
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


def main() -> None:
    """Parse arguments and produce per-AOI agriculture annotation sets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        required=True,
        help="Directory with Studio results at {predictions_dir}/result_batch{i}.geojson.",
    )
    parser.add_argument(
        "--orig-dir",
        required=True,
        help="Directory with prediction_request_geometry_batch{i}_orig.geojson files.",
    )
    parser.add_argument(
        "--aoi",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="AOI as NAME=path to a GeoJSON or Esri JSON polygon file. Repeatable.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--regen",
        default=None,
        help=(
            "Optional non-smoothed prediction_request_geometry.geojson. When set, each "
            "event's polygon geometry is replaced by the matching one (by "
            "(tif_fname, center_pixel)); unmatched events keep the smoothed geometry. "
            "All other attributes still come from the per-batch orig files."
        ),
    )
    parser.add_argument(
        "--target-label",
        default="agriculture",
        help="Comma-separated label(s) to keep (default: agriculture).",
    )
    parser.add_argument("--sample-agriculture", type=int, default=2000)
    parser.add_argument("--sample-large", type=int, default=1000)
    parser.add_argument("--min-hectares", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Worker processes for AOI matching and area computation (default: 64).",
    )
    parser.add_argument(
        "--write-merged",
        action="store_true",
        help="Also write per-batch merged_batch{i}.geojson polygon files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    target_labels = {x.strip() for x in args.target_label.split(",")}
    aois = {}
    for spec in args.aoi:
        name, path = spec.split("=", 1)
        aois[name] = path

    # 1. Merge each batch's results back onto polygons.
    regen_index = build_regen_index(args.regen) if args.regen else None
    if regen_index is not None:
        print(f"Loaded {len(regen_index)} regen geometries from {args.regen}")
    merged = []
    orig_paths = sorted(glob.glob(os.path.join(args.orig_dir, "*_batch*_orig.geojson")))
    if not orig_paths:
        raise SystemExit(f"No *_batch*_orig.geojson files found in {args.orig_dir}")
    print(f"Merging {len(orig_paths)} batches")
    for orig_path in orig_paths:
        m = re.search(r"_batch(\d+)_orig\.geojson$", os.path.basename(orig_path))
        if m is None:
            continue
        batch_idx = m.group(1)
        result_path = os.path.join(
            args.predictions_dir, f"result_batch{batch_idx}.geojson"
        )
        if not os.path.exists(result_path):
            raise SystemExit(
                f"Missing Studio result for batch{batch_idx}: {result_path}"
            )
        batch_merged = merge_batch(orig_path, result_path, regen_index)
        if args.write_merged:
            write_geojson(
                os.path.join(args.output_dir, f"merged_batch{batch_idx}.geojson"),
                batch_merged,
            )
        merged.extend(batch_merged)
    print(f"Total merged polygon events: {len(merged)}")

    # Keep only target-label polygons; precompute shapely geometry + centroid.
    target_feats = []
    target_shapes = []
    target_centroids = []
    for feat in merged:
        if feat["properties"].get("new_label") not in target_labels:
            continue
        shp = shapely.geometry.shape(feat["geometry"])
        cen = shp.centroid
        target_feats.append(feat)
        target_shapes.append(shp)
        target_centroids.append((cen.x, cen.y))
    print(
        f"Target-label ({','.join(sorted(target_labels))}) events: {len(target_feats)}"
    )

    # Load all AOI polygons upfront (sent to workers via the pool initializer).
    aoi_geoms = {name: load_aoi_polygon(path) for name, path in aois.items()}

    chunk_size = max(1, len(target_feats) // (args.workers * 4) + 1)
    pool = Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(target_centroids, target_shapes, aoi_geoms),
    )
    try:
        # 2. Per-AOI processing.
        for name in aois:
            # AOI matching (parallel): keep target indices whose centroid is inside.
            match_tasks = [
                (name, chunk)
                for chunk in _chunked(list(range(len(target_feats))), chunk_size)
            ]
            in_aoi_idx = [i for sub in pool.map(_match_chunk, match_tasks) for i in sub]
            print(f"{name}: {len(in_aoi_idx)} target polygons in AOI")

            # X_agriculture: up to sample_agriculture.
            rng = random.Random(args.seed)
            if len(in_aoi_idx) <= args.sample_agriculture:
                agri_idx = list(in_aoi_idx)
            else:
                agri_idx = rng.sample(in_aoi_idx, args.sample_agriculture)
            write_geojson(
                os.path.join(args.output_dir, f"{name}_agriculture.geojson"),
                [target_feats[i] for i in agri_idx],
            )

            # X_agriculture_large: area > min_hectares (may overlap _agriculture).
            # Area computation (parallel) over all agriculture polygons in the AOI.
            areas: dict[int, float] = {}
            for sub in pool.map(_area_chunk, _chunked(in_aoi_idx, chunk_size)):
                areas.update(sub)
            large_idx = [i for i in in_aoi_idx if areas[i] > args.min_hectares]
            rng2 = random.Random(args.seed)
            if len(large_idx) > args.sample_large:
                large_idx = rng2.sample(large_idx, args.sample_large)
            write_geojson(
                os.path.join(args.output_dir, f"{name}_agriculture_large.geojson"),
                [target_feats[i] for i in large_idx],
            )
            print(
                f"{name}: wrote {len(agri_idx)} agriculture, {len(large_idx)} large "
                f"(>{args.min_hectares}ha; {len(in_aoi_idx)} candidates)"
            )
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
