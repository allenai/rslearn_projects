"""Flask app for browsing embedding-based change examples.

Usage:
    python -m rslp.change_finder.embedding_change_viewer.server \
        --json embedding_change_list.json \
        --ds-path /weka/.../ten_year_dataset_20260408 \
        --port 8080
"""

import argparse
import hashlib
import io
import json
import multiprocessing
from pathlib import Path

import numpy as np
import rasterio
import tqdm
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import WindowLayerData
from rslearn.utils.geometry import WGS84_PROJECTION
from upath import UPath

BASE_YEAR = 2016
NUM_YEARS = 10
METADATA_WORKERS = 64


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


def _shuffle_key(window_group: str, window_name: str) -> str:
    """Deterministic hash order so examples aren't clumped by group."""
    return hashlib.md5(f"{window_group}/{window_name}".encode()).hexdigest()


def create_app(
    json_path: str,
    ds_path_str: str,
    mask_filename: str = "embeddings.tif",
    threshold: float = 150,
) -> Flask:
    """Create the Flask app backing the embedding change viewer."""
    ds_path = UPath(ds_path_str)
    json_upath = UPath(json_path)
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

    print(f"Loading change list from {json_upath}")
    with json_upath.open() as f:
        raw_entries = json.load(f)
    print(f"Loaded {len(raw_entries)} entries")

    # Keep only entries whose window exists in the dataset, deduped, sorted by
    # a stable shuffled order so downstream browsing interleaves groups.
    seen: set[tuple[str, str]] = set()
    examples: list[tuple[str, str]] = []
    for entry in raw_entries:
        key = (entry["window_group"], entry["window_name"])
        if key in seen:
            continue
        if key not in window_cache:
            continue
        seen.add(key)
        examples.append(key)
    examples.sort(key=lambda k: _shuffle_key(k[0], k[1]))
    print(f"Kept {len(examples)} entries after filtering / dedup")

    # Pre-compute per-window metadata only for windows we'll actually show.
    todo_windows = [window_cache[k] for k in examples]
    print(
        f"Pre-computing per-window metadata for {len(todo_windows)} windows "
        f"with {METADATA_WORKERS} workers..."
    )
    window_meta_cache: dict[tuple[str, str], dict] = {}
    with multiprocessing.Pool(METADATA_WORKERS) as pool:
        for key, meta in tqdm.tqdm(
            pool.imap_unordered(_compute_window_meta, todo_windows),
            total=len(todo_windows),
            desc="Window metadata",
        ):
            window_meta_cache[key] = meta
    print("Done pre-computing window metadata")

    # Drop examples whose metadata failed to compute.
    examples = [k for k in examples if k in window_meta_cache]
    print(f"Final example count: {len(examples)}")

    template_folder = Path(__file__).parent / "templates"
    static_folder = Path(__file__).parent / "static"
    app = Flask(
        __name__,
        template_folder=str(template_folder),
        static_folder=str(static_folder),
    )

    def _example_payload(idx: int) -> dict:
        group, name = examples[idx]
        meta = window_meta_cache[(group, name)]
        return {
            "idx": idx,
            "window_group": group,
            "window_name": name,
            "lat": meta["lat"],
            "lon": meta["lon"],
            "years": meta["years"],
        }

    @app.route("/")
    def index() -> str:
        """Render the single-page UI."""
        return render_template("index.html")

    @app.route("/api/examples")
    def api_examples() -> Response:
        """Return all examples in display order."""
        return jsonify([_example_payload(i) for i in range(len(examples))])

    @app.route("/api/config")
    def api_config() -> Response:
        """Return viewer config (default threshold etc)."""
        return jsonify({"threshold": threshold})

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

    def _read_mask(window_group: str, window_name: str) -> np.ndarray | None:
        """Read the change-mask tif for a window, or None if missing."""
        if (window_group, window_name) not in window_cache:
            return None
        wroot = dataset.storage.get_window_root(window_group, window_name)
        tiff_path = wroot / mask_filename
        if not tiff_path.exists():
            return None
        with rasterio.open(str(tiff_path)) as src:
            return src.read(1)

    @app.route("/image/change_mask/<window_group>/<window_name>")
    def image_change_mask(window_group: str, window_name: str) -> Response:
        """Render the per-patch change mask as a grayscale PNG."""
        arr = _read_mask(window_group, window_name)
        if arr is None:
            return Response(f"{mask_filename} not found", status=404)
        return Response(
            _numpy_to_png(arr.astype(np.uint8), mode="L"),
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.route("/image/change_overlay/<window_group>/<window_name>")
    def image_change_overlay(window_group: str, window_name: str) -> Response:
        """Render an RGBA overlay highlighting patches above the threshold."""
        arr = _read_mask(window_group, window_name)
        if arr is None:
            return Response(f"{mask_filename} not found", status=404)

        try:
            t = float(request.args.get("threshold", threshold))
        except ValueError:
            return Response("invalid threshold", status=400)

        h, w = arr.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        hot = arr > t
        rgba[hot] = [255, 200, 0, 180]
        return Response(
            _numpy_to_png(rgba, mode="RGBA"),
            mimetype="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    return app


def main() -> None:
    """CLI entrypoint: parse args and serve the embedding change viewer."""
    parser = argparse.ArgumentParser(description="Embedding change viewer")
    parser.add_argument(
        "--json",
        required=True,
        help="JSON list from create_embedding_change_list.py",
    )
    parser.add_argument("--ds-path", required=True, help="Path to rslearn dataset root")
    parser.add_argument(
        "--mask-filename",
        default="embeddings.tif",
        help="Per-patch change mask GeoTIFF filename inside each window",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=150,
        help="Default overlay threshold (pixels > this are highlighted)",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(
        json_path=args.json,
        ds_path_str=args.ds_path,
        mask_filename=args.mask_filename,
        threshold=args.threshold,
    )
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
