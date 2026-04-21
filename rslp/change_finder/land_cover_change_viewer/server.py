"""Flask app for browsing land cover change examples.

Usage:
    python -m rslp.change_finder.land_cover_change_viewer.server \
        --geojson land_cover_change.geojson \
        --ds-path /weka/.../ten_year_dataset_20260408 \
        --port 8080
"""

import argparse
import hashlib
import io
import json
import multiprocessing
import re
import threading
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyproj
import rasterio
import rasterio.features
import tqdm
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import WindowLayerData
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.geometry import WGS84_PROJECTION
from rslearn.utils.raster_format import get_transform_from_projection_and_bounds
from shapely.geometry import mapping, shape
from shapely.ops import transform
from upath import UPath

BASE_YEAR = 2016
NUM_YEARS = 10
NUM_CLASSES = 13
METADATA_WORKERS = 64
ALL_SENTINEL = "__all__"
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
PROBS_FILENAME = "land_cover_probs.tif"
ANNOTATION_FIELDS = ("pre_change", "change_start", "change_end", "post_change")

WORLDCOVER_COLORS = np.array(
    [
        (0, 0, 0),  # 0  nodata
        (180, 180, 180),  # 1  bare
        (139, 69, 19),  # 2  burnt
        (240, 150, 255),  # 3  crops
        (200, 120, 200),  # 4  fallow/shifting
        (255, 255, 76),  # 5  grassland
        (250, 230, 160),  # 6  lichen and moss
        (255, 187, 34),  # 7  shrub
        (240, 240, 240),  # 8  snow and ice
        (0, 100, 0),  # 9  tree
        (250, 0, 0),  # 10 urban/built-up
        (0, 100, 200),  # 11 water
        (0, 150, 160),  # 12 wetland
    ],
    dtype=np.uint8,
)

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

LEGEND = [
    {
        "class_id": i,
        "class_name": CLASS_NAMES[i],
        "r": int(WORLDCOVER_COLORS[i][0]),
        "g": int(WORLDCOVER_COLORS[i][1]),
        "b": int(WORLDCOVER_COLORS[i][2]),
    }
    for i in range(len(CLASS_NAMES))
    if i != 0
]


def _get_year_timestamps(
    layer_datas: dict[str, WindowLayerData],
) -> dict[int, list[dict]]:
    """Extract per-year, per-group timestamps from WindowLayerData objects."""
    result: dict[int, list[dict]] = {}
    for layer_name, ld in layer_datas.items():
        if not layer_name.startswith("sentinel2_y"):
            continue
        year_idx = int(layer_name.removeprefix("sentinel2_y"))
        groups = []
        for gi, item_group in enumerate(ld.serialized_item_groups):
            if not item_group:
                continue
            ts = item_group[0].get("geometry", {}).get("time_range", [None])[0]
            date_str = ts[:10] if ts else None
            groups.append({"group_idx": gi, "date": date_str})
        result[year_idx] = groups
    return result


def _numpy_to_png(arr: np.ndarray, mode: str = "RGB") -> bytes:
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _get_window_center_wgs84(window: Window) -> tuple[float, float]:
    """Return (lat, lon) of the window center in WGS-84."""
    geom = window.get_geometry()
    wgs84_geom = geom.to_projection(WGS84_PROJECTION)
    centroid = wgs84_geom.shp.centroid
    return centroid.y, centroid.x


def _compute_window_meta(window: Window) -> tuple[tuple[str, str], dict]:
    """Worker: compute metadata for a single window (key + dict)."""
    layer_datas = window.load_layer_datas()
    years = _get_year_timestamps(layer_datas)
    lat, lon = _get_window_center_wgs84(window)
    return (window.group, window.name), {
        "lat": lat,
        "lon": lon,
        "years": {str(k): v for k, v in years.items()},
    }


def _feature_metadata(feat: dict, window_meta: dict, annotation: dict | None) -> dict:
    """Build the JSON-serializable metadata payload for one feature."""
    props = feat["properties"]
    payload = {
        "feature_idx": props["feature_idx"],
        "window_group": props["window_group"],
        "window_name": props["window_name"],
        "lat": window_meta["lat"],
        "lon": window_meta["lon"],
        "src_class_id": props["src_class_id"],
        "src_class_name": props["src_class_name"],
        "dst_class_id": props["dst_class_id"],
        "dst_class_name": props["dst_class_name"],
        "num_pixels": props["num_pixels"],
        "pivot_year": props.get("pivot_year"),
        "early_years": props.get("early_years", []),
        "late_years": props.get("late_years", []),
        "years": window_meta["years"],
    }
    for field in ANNOTATION_FIELDS:
        payload[field] = annotation.get(field) if annotation else None
    return payload


def _default_annotations_path(geojson_upath: UPath) -> UPath:
    """Return ``<geojson>.annotations.json`` next to the GeoJSON."""
    stem = geojson_upath.name
    if stem.endswith(".geojson"):
        stem = stem[: -len(".geojson")]
    elif stem.endswith(".json"):
        stem = stem[: -len(".json")]
    return geojson_upath.parent / f"{stem}.annotations.json"


def _load_annotations(
    annotations_path: UPath, all_features: list[dict]
) -> dict[int, dict]:
    """Load annotations by ``feature_idx``.

    If the annotations file does not yet exist, migrate any legacy
    ``change_start_month``/``change_end_month`` props from the GeoJSON
    into the new schema. This migration is purely in-memory; the
    GeoJSON itself is never rewritten.
    """
    if annotations_path.exists():
        with annotations_path.open() as f:
            raw = json.load(f)
        result: dict[int, dict] = {}
        for entry in raw:
            idx = int(entry["feature_idx"])
            result[idx] = {
                "feature_idx": idx,
                "window_group": entry.get("window_group", ""),
                "window_name": entry.get("window_name", ""),
                "src_class_name": entry.get("src_class_name", ""),
                "dst_class_name": entry.get("dst_class_name", ""),
                **{f: entry.get(f) for f in ANNOTATION_FIELDS},
            }
        return result

    result = {}
    for idx, feat in enumerate(all_features):
        props = feat["properties"]
        if props.get("feature_type", "change") != "change":
            continue
        start = props.get("change_start_month")
        end = props.get("change_end_month")
        if not start and not end:
            continue
        result[idx] = {
            "feature_idx": idx,
            "window_group": props["window_group"],
            "window_name": props["window_name"],
            "src_class_name": props.get("src_class_name", ""),
            "dst_class_name": props.get("dst_class_name", ""),
            "pre_change": None,
            "change_start": start or None,
            "change_end": end or None,
            "post_change": None,
        }
    return result


def _save_annotations(annotations_path: UPath, annotations: dict[int, dict]) -> None:
    """Persist annotations as a list sorted by ``feature_idx``."""
    data = [annotations[idx] for idx in sorted(annotations.keys())]
    with open_atomic(annotations_path, "w") as f:
        json.dump(data, f, indent=2)


def create_app(
    geojson_path: str, ds_path_str: str, annotations_path: str | None = None
) -> Flask:
    """Create the Flask app backing the land cover change viewer."""
    ds_path = UPath(ds_path_str)
    geojson_upath = UPath(geojson_path)
    annotations_upath = (
        UPath(annotations_path)
        if annotations_path
        else _default_annotations_path(geojson_upath)
    )
    dataset = Dataset(ds_path)

    # Figure out the band list for sentinel2 layers from the dataset config.
    s2_bands: list[str] | None = None
    for layer_name, layer_config in dataset.layers.items():
        if layer_name.startswith("sentinel2_y"):
            s2_bands = layer_config.band_sets[0].bands
            break
    if s2_bands is None:
        raise RuntimeError("No sentinel2_y* layer found in dataset config")

    b04_idx = s2_bands.index("B04")
    b03_idx = s2_bands.index("B03")
    b02_idx = s2_bands.index("B02")

    print(f"Sentinel-2 bands: {s2_bands}, RGB indices: {b04_idx},{b03_idx},{b02_idx}")

    print("Loading windows...")
    windows = dataset.load_windows(workers=64, show_progress=True)
    window_cache: dict[tuple[str, str], Window] = {
        (w.group, w.name): w for w in windows
    }
    print(f"Loaded {len(windows)} windows")

    print(f"Loading GeoJSON from {geojson_upath}")
    with geojson_upath.open() as f:
        fc = json.load(f)
    all_features: list[dict] = fc["features"]
    print(f"Loaded {len(all_features)} features")

    # Split features into "change" (navigable) and "no_change" (overlay-only,
    # keyed by window). Features without an explicit feature_type fall back to
    # "change" for backward compat with older GeoJSONs. Stamp each change
    # feature with its original index in the GeoJSON so annotations can be
    # keyed by a stable id independent of filtering/sorting.
    change_features: list[dict] = []
    no_change_by_window: dict[tuple[str, str], dict] = {}
    for idx, feat in enumerate(all_features):
        props = feat["properties"]
        ftype = props.get("feature_type", "change")
        wkey = (props["window_group"], props["window_name"])
        if ftype == "no_change":
            no_change_by_window[wkey] = feat
        else:
            props["feature_idx"] = idx
            change_features.append(feat)
    print(
        f"Split into {len(change_features)} change features "
        f"and {len(no_change_by_window)} no-change features"
    )

    print(f"Loading annotations from {annotations_upath}")
    annotations: dict[int, dict] = _load_annotations(annotations_upath, all_features)
    print(f"Loaded {len(annotations)} annotations")

    # Pre-compute per-window metadata only for windows referenced by change
    # features, since no-change features are never navigated to directly.
    window_meta_cache: dict[tuple[str, str], dict | None] = {}
    needed_keys = set()
    for feat in change_features:
        props = feat["properties"]
        key = (props["window_group"], props["window_name"])
        if key not in window_cache:
            window_meta_cache[key] = None
            continue
        needed_keys.add(key)

    todo_windows = [window_cache[k] for k in needed_keys]
    print(
        f"Pre-computing per-window metadata for {len(todo_windows)} windows "
        f"with {METADATA_WORKERS} workers..."
    )
    with multiprocessing.Pool(METADATA_WORKERS) as pool:
        for key, meta in tqdm.tqdm(
            pool.imap_unordered(_compute_window_meta, todo_windows),
            total=len(todo_windows),
            desc="Window metadata",
        ):
            window_meta_cache[key] = meta
    print("Done pre-computing window metadata")

    # Drop change features whose window is missing. The no-change dict is
    # kept as-is because missing windows just won't be overlaid anyway.
    change_features = [
        feat
        for feat in change_features
        if window_meta_cache.get(
            (feat["properties"]["window_group"], feat["properties"]["window_name"])
        )
        is not None
    ]

    # Build (src, dst) -> features index for category listings.
    pair_index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for feat in change_features:
        p = feat["properties"]
        pair_index[(p["src_class_name"], p["dst_class_name"])].append(feat)

    pairs = sorted(pair_index.keys())
    print(f"Found {len(pairs)} (src, dst) pairs")

    write_lock = threading.Lock()

    def _example_payload(feat: dict) -> dict:
        props = feat["properties"]
        wmeta = window_meta_cache[(props["window_group"], props["window_name"])]
        # `_filtered_features` only ever yields features whose window metadata
        # was computed (i.e. non-None); make that assumption explicit for mypy.
        assert wmeta is not None
        annotation = annotations.get(int(props["feature_idx"]))
        return _feature_metadata(feat, wmeta, annotation)

    def _shuffle_key(feat: dict) -> str:
        """Deterministic but src/dst-agnostic order so results are interleaved."""
        p = feat["properties"]
        ident = f"{p['window_group']}/{p['window_name']}/{p['src_class_id']}->{p['dst_class_id']}"
        return hashlib.md5(ident.encode()).hexdigest()

    def _filtered_features(src: str, dst: str) -> list[dict]:
        """Return change features matching the src/dst filter (sentinel `__all__` matches any)."""
        result = [
            feat
            for feat in change_features
            if (src == ALL_SENTINEL or feat["properties"]["src_class_name"] == src)
            and (dst == ALL_SENTINEL or feat["properties"]["dst_class_name"] == dst)
        ]
        # Hash-order the whole identity so All -> All shows an interleaved mix
        # rather than bucketing all (bare -> crop) examples together.
        result.sort(key=_shuffle_key)
        return result

    template_folder = Path(__file__).parent / "templates"
    static_folder = Path(__file__).parent / "static"
    app = Flask(
        __name__,
        template_folder=str(template_folder),
        static_folder=str(static_folder),
    )

    @app.route("/")
    def index() -> str:
        """Render the single-page UI."""
        return render_template("index.html")

    @app.route("/api/categories")
    def api_categories() -> Response:
        """Return the list of (src, dst) pairs that have at least one feature."""
        return jsonify(
            [
                {
                    "src_class_name": s,
                    "dst_class_name": d,
                    "count": len(pair_index[(s, d)]),
                }
                for s, d in pairs
            ]
        )

    @app.route("/api/examples")
    def api_examples() -> Response:
        """Return change features matching ``src`` and ``dst`` query params."""
        src = request.args.get("src", ALL_SENTINEL)
        dst = request.args.get("dst", ALL_SENTINEL)
        return jsonify([_example_payload(f) for f in _filtered_features(src, dst)])

    @app.route("/api/legend")
    def api_legend() -> Response:
        """Return the land-cover legend as a list of class/color entries."""
        return jsonify(LEGEND)

    feature_by_idx: dict[int, dict] = {
        int(feat["properties"]["feature_idx"]): feat for feat in change_features
    }

    @app.route("/api/annotate", methods=["POST"])
    def api_annotate() -> Response | tuple[Response, int]:
        """Persist annotation dates for one feature to the annotations file."""
        body = request.get_json(force=True, silent=True) or {}
        try:
            feature_idx = int(body["feature_idx"])
        except (KeyError, TypeError, ValueError):
            return jsonify(
                {"ok": False, "error": "missing or invalid feature_idx"}
            ), 400

        values: dict[str, str] = {}
        for field in ANNOTATION_FIELDS:
            v = body.get(field) or ""
            if v != "" and not MONTH_RE.match(v):
                return jsonify({"ok": False, "error": f"{field} must be YYYY-MM"}), 400
            values[field] = v

        with write_lock:
            target_feat = feature_by_idx.get(feature_idx)
            if target_feat is None:
                return jsonify({"ok": False, "error": "feature not found"}), 404

            all_empty = all(v == "" for v in values.values())
            if all_empty:
                annotations.pop(feature_idx, None)
            else:
                props = target_feat["properties"]
                entry = {
                    "feature_idx": feature_idx,
                    "window_group": props["window_group"],
                    "window_name": props["window_name"],
                    "src_class_name": props["src_class_name"],
                    "dst_class_name": props["dst_class_name"],
                }
                for field in ANNOTATION_FIELDS:
                    entry[field] = values[field] or None
                annotations[feature_idx] = entry

            _save_annotations(annotations_upath, annotations)
            payload = _example_payload(target_feat)

        return jsonify({"ok": True, "feature": payload})

    @app.route(
        "/image/sentinel2/<window_group>/<window_name>/<int:year_idx>/<int:group_idx>"
    )
    def image_sentinel2(
        window_group: str, window_name: str, year_idx: int, group_idx: int
    ) -> Response:
        """Render a Sentinel-2 RGB PNG for one (window, year, item-group)."""
        window = window_cache.get((window_group, window_name))
        if window is None:
            return Response("window not found", status=404)

        layer_name = f"sentinel2_y{year_idx}"
        raster_dir = window.get_raster_dir(layer_name, s2_bands, group_idx=group_idx)
        tiff_path = raster_dir / "geotiff.tif"
        if not tiff_path.exists():
            return Response("GeoTIFF not found", status=404)

        with rasterio.open(str(tiff_path)) as src:
            bands = src.read()

        r = np.clip(bands[b04_idx].astype(np.float32) / 10, 0, 255).astype(np.uint8)
        g = np.clip(bands[b03_idx].astype(np.float32) / 10, 0, 255).astype(np.uint8)
        b = np.clip(bands[b02_idx].astype(np.float32) / 10, 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)

        return Response(
            _numpy_to_png(rgb),
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.route("/image/landcover/<window_group>/<window_name>")
    def image_landcover(window_group: str, window_name: str) -> Response:
        """Render the dominant-class PNG averaged over the requested years."""
        years_arg = request.args.get("years", "")
        try:
            years = [int(y) for y in years_arg.split(",") if y.strip()]
        except ValueError:
            return Response("invalid years", status=400)
        if not years:
            return Response("years required", status=400)
        year_indices = [y - BASE_YEAR for y in years]
        if any(yi < 0 or yi >= NUM_YEARS for yi in year_indices):
            return Response("year out of range", status=400)

        window = window_cache.get((window_group, window_name))
        if window is None:
            return Response("window not found", status=404)

        wroot = dataset.storage.get_window_root(window_group, window_name)
        tiff_path = wroot / PROBS_FILENAME
        if not tiff_path.exists():
            return Response(f"{PROBS_FILENAME} not found", status=404)

        # probs layout is year-major: band (y*NUM_CLASSES + c) for y in [0, NUM_YEARS).
        # rasterio is 1-indexed, so offset by +1 when selecting bands.
        with rasterio.open(str(tiff_path)) as src:
            stack = np.stack(
                [
                    src.read(y * NUM_CLASSES + c + 1)
                    for y in year_indices
                    for c in range(NUM_CLASSES)
                ]
            ).astype(np.float32)
        # Shape: (len(year_indices) * NUM_CLASSES, H, W) -> mean across years
        stack = stack.reshape(len(year_indices), NUM_CLASSES, *stack.shape[1:])
        avg = stack.mean(axis=0)
        class_map = np.argmax(avg, axis=0).astype(np.int64)

        rgb = WORLDCOVER_COLORS[class_map]
        return Response(
            _numpy_to_png(rgb),
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.route("/image/change_polygon/<int:index>")
    def image_change_polygon(index: int) -> Response:
        """Render a translucent overlay showing the change + no-change masks."""
        src = request.args.get("src", ALL_SENTINEL)
        dst = request.args.get("dst", ALL_SENTINEL)
        feats = _filtered_features(src, dst)
        if index < 0 or index >= len(feats):
            return Response("index out of range", status=404)

        feat = feats[index]
        props = feat["properties"]
        wkey = (props["window_group"], props["window_name"])
        window = window_cache.get(wkey)
        if window is None:
            return Response("window not found", status=404)

        t = get_transform_from_projection_and_bounds(window.projection, window.bounds)
        width = window.bounds[2] - window.bounds[0]
        height = window.bounds[3] - window.bounds[1]

        proj_wgs84 = pyproj.CRS("EPSG:4326")
        proj_window = pyproj.CRS(window.projection.crs)
        transformer = pyproj.Transformer.from_crs(
            proj_wgs84, proj_window, always_xy=True
        )

        def _rasterize(geojson_geom: dict) -> np.ndarray:
            geom_wgs84 = shape(geojson_geom)
            geom_projected = transform(transformer.transform, geom_wgs84)
            return rasterio.features.rasterize(
                [(mapping(geom_projected), 1)],
                out_shape=(height, width),
                transform=t,
                fill=0,
                dtype=np.uint8,
            )

        mask_change = _rasterize(feat["geometry"])
        nc_feat = no_change_by_window.get(wkey)
        mask_nc = (
            _rasterize(nc_feat["geometry"])
            if nc_feat is not None
            else np.zeros((height, width), dtype=np.uint8)
        )

        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        # No-change painted first so the change layer overrides on any overlap.
        rgba[mask_nc == 1] = [60, 180, 255, 140]
        rgba[mask_change == 1] = [255, 200, 0, 180]

        return Response(
            _numpy_to_png(rgba, mode="RGBA"),
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    return app


def main() -> None:
    """CLI entrypoint: parse args and serve the land cover change viewer."""
    parser = argparse.ArgumentParser(description="Land cover change viewer")
    parser.add_argument(
        "--geojson",
        required=True,
        help="GeoJSON from create_land_cover_change_geojson.py",
    )
    parser.add_argument("--ds-path", required=True, help="Path to rslearn dataset root")
    parser.add_argument(
        "--annotations",
        default=None,
        help=(
            "Path to annotations JSON file. Defaults to "
            "<geojson_stem>.annotations.json next to the GeoJSON."
        ),
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(args.geojson, args.ds_path, args.annotations)
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
