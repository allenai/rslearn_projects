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

NUM_YEARS = 10
METADATA_WORKERS = 64
ALL_SENTINEL = "__all__"
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

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


def _feature_metadata(feat: dict, window_meta: dict) -> dict:
    """Build the JSON-serializable metadata payload for one feature."""
    props = feat["properties"]
    return {
        "window_group": props["window_group"],
        "window_name": props["window_name"],
        "lat": window_meta["lat"],
        "lon": window_meta["lon"],
        "src_class_id": props["src_class_id"],
        "src_class_name": props["src_class_name"],
        "dst_class_id": props["dst_class_id"],
        "dst_class_name": props["dst_class_name"],
        "num_pixels": props["num_pixels"],
        "change_start_month": props.get("change_start_month"),
        "change_end_month": props.get("change_end_month"),
        "years": window_meta["years"],
    }


def _feature_key(props: dict) -> tuple[str, str, int, int]:
    return (
        props["window_group"],
        props["window_name"],
        int(props["src_class_id"]),
        int(props["dst_class_id"]),
    )


def create_app(geojson_path: str, ds_path_str: str) -> Flask:
    """Create the Flask app backing the land cover change viewer."""
    ds_path = UPath(ds_path_str)
    geojson_upath = UPath(geojson_path)
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
    # "change" for backward compat with older GeoJSONs.
    change_features: list[dict] = []
    no_change_by_window: dict[tuple[str, str], dict] = {}
    for feat in all_features:
        props = feat["properties"]
        ftype = props.get("feature_type", "change")
        wkey = (props["window_group"], props["window_name"])
        if ftype == "no_change":
            no_change_by_window[wkey] = feat
        else:
            change_features.append(feat)
    print(
        f"Split into {len(change_features)} change features "
        f"and {len(no_change_by_window)} no-change features"
    )

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
    # Rebuild fc["features"] so file persistence only drops orphaned change
    # features, not the no-change features.
    fc["features"] = change_features + list(no_change_by_window.values())

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
        return _feature_metadata(feat, wmeta)

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

    @app.route("/api/annotate", methods=["POST"])
    def api_annotate() -> Response | tuple[Response, int]:
        """Persist ``change_start_month`` / ``change_end_month`` on one feature."""
        body = request.get_json(force=True, silent=True) or {}
        try:
            window_group = body["window_group"]
            window_name = body["window_name"]
            src_class_id = int(body["src_class_id"])
            dst_class_id = int(body["dst_class_id"])
            start_month = body.get("change_start_month") or ""
            end_month = body.get("change_end_month") or ""
        except (KeyError, TypeError, ValueError):
            return jsonify({"ok": False, "error": "missing or invalid fields"}), 400

        for label, value in (
            ("change_start_month", start_month),
            ("change_end_month", end_month),
        ):
            if value != "" and not MONTH_RE.match(value):
                return jsonify({"ok": False, "error": f"{label} must be YYYY-MM"}), 400

        target_key = (window_group, window_name, src_class_id, dst_class_id)

        with write_lock:
            target_feat = None
            for feat in change_features:
                if _feature_key(feat["properties"]) == target_key:
                    target_feat = feat
                    break
            if target_feat is None:
                return jsonify({"ok": False, "error": "feature not found"}), 404

            target_feat["properties"]["change_start_month"] = start_month or None
            target_feat["properties"]["change_end_month"] = end_month or None

            with open_atomic(geojson_upath, "w") as f:
                json.dump(fc, f)

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

    @app.route("/image/landcover/<window_group>/<window_name>/<period>")
    def image_landcover(window_group: str, window_name: str, period: str) -> Response:
        """Render the early/late dominant-class PNG for one window."""
        if period not in ("early", "late"):
            return Response("period must be 'early' or 'late'", status=400)

        window = window_cache.get((window_group, window_name))
        if window is None:
            return Response("window not found", status=404)

        wroot = dataset.storage.get_window_root(window_group, window_name)
        tiff_path = wroot / "land_cover_change.tif"
        if not tiff_path.exists():
            return Response("land_cover_change.tif not found", status=404)

        # rasterio 1-indexed: band 2 = early dominant class, band 3 = late
        band_idx = 2 if period == "early" else 3
        with rasterio.open(str(tiff_path)) as src:
            class_map = src.read(band_idx)

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
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(args.geojson, args.ds_path)
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
