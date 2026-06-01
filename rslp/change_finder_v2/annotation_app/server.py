"""Flask annotation app for change_finder_v2.

Usage:
    python -m rslp.change_finder_v2.annotation_app.server \
        --json annotations.json \
        --ds-path /path/to/dataset \
        --port 8080
"""

from __future__ import annotations

import argparse
import io
import json
import math
import threading
from pathlib import Path

import numpy as np
import rasterio
import shapely.geometry
import tqdm
from flask import Flask, Response, jsonify, render_template, request
from PIL import Image, ImageDraw
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import WindowLayerData
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from upath import UPath

POINT_RADIUS = 2
POSITIVE_COLOR = (0, 200, 0, 180)
NEGATIVE_COLOR = (200, 0, 0, 180)
SELECTED_COLOR = (0, 255, 0, 255)


def _numpy_to_png(arr: np.ndarray, mode: str = "RGB") -> bytes:
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _get_window_center_wgs84(window: Window) -> tuple[float, float]:
    geom = window.get_geometry()
    wgs84_geom = geom.to_projection(WGS84_PROJECTION)
    centroid = wgs84_geom.shp.centroid
    return centroid.y, centroid.x


def _lonlat_to_pixel(
    lon: float,
    lat: float,
    projection: Projection,
    bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Convert lon/lat to window-local pixel coords (col, row)."""
    pt = shapely.geometry.Point(lon, lat)
    pixel_pt = (
        STGeometry(WGS84_PROJECTION, pt, time_range=None).to_projection(projection).shp
    )
    col = pixel_pt.x - bounds[0]
    row = pixel_pt.y - bounds[1]
    return col, row


def _pixel_to_lonlat(
    col: float,
    row: float,
    projection: Projection,
    bounds: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Convert window-local pixel coords (col, row) to lon/lat."""
    px_x = bounds[0] + col
    px_y = bounds[1] + row
    pt = shapely.geometry.Point(px_x, px_y)
    wgs84_pt = (
        STGeometry(projection, pt, time_range=None).to_projection(WGS84_PROJECTION).shp
    )
    return wgs84_pt.x, wgs84_pt.y


def _get_timestamps(
    layer_datas: dict[str, WindowLayerData],
) -> dict[int, list[dict]]:
    """Extract per-year timestamps from the sentinel2 layer's item groups."""
    ld = layer_datas.get("sentinel2")
    if ld is None:
        return {}

    result: dict[int, list[dict]] = {}
    for gi, item_group in enumerate(ld.serialized_item_groups):
        if not item_group:
            continue
        ts = item_group[0].get("geometry", {}).get("time_range", [None])[0]
        if ts is None:
            continue
        date_str = ts[:10]
        try:
            year = int(date_str[:4])
        except (ValueError, IndexError):
            continue
        if year not in result:
            result[year] = []
        result[year].append({"group_idx": gi, "date": date_str})

    return result


def create_app(v2_json_path: str, ds_path_str: str) -> Flask:
    """Create the Flask annotation app."""
    v2_path = UPath(v2_json_path)
    ds_path = UPath(ds_path_str)
    dataset = Dataset(ds_path)

    s2_bands: list[str] | None = None
    for layer_name, layer_config in dataset.layers.items():
        if layer_name == "sentinel2":
            s2_bands = layer_config.band_sets[0].bands
            break
    if s2_bands is None:
        raise RuntimeError("No sentinel2 layer found in dataset config")

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

    print(f"Loading v2 JSON from {v2_path}")
    with v2_path.open() as f:
        entries: list[dict] = json.load(f)
    print(f"Loaded {len(entries)} entries")

    # Pre-compute per-entry metadata.
    entry_meta: list[dict] = []
    print("Pre-computing entry metadata...")
    for idx, entry in enumerate(tqdm.tqdm(entries, desc="Entry metadata")):
        wkey = (entry["group"], entry["window_name"])
        window = window_cache.get(wkey)
        if window is None:
            entry_meta.append({"available": False, "years": {}})
            continue
        layer_datas = window.load_layer_datas()
        years = _get_timestamps(layer_datas)
        lat, lon = _get_window_center_wgs84(window)
        entry_meta.append(
            {
                "available": True,
                "years": years,
                "lat": lat,
                "lon": lon,
            }
        )
    print("Done pre-computing metadata")

    write_lock = threading.Lock()

    def _save_entries() -> None:
        with open_atomic(v2_path, "w") as f:
            json.dump(entries, f, indent=2)

    template_folder = Path(__file__).parent / "templates"
    static_folder = Path(__file__).parent / "static"
    app = Flask(
        __name__,
        template_folder=str(template_folder),
        static_folder=str(static_folder),
    )

    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/api/entries")
    def api_entries() -> Response:
        result = []
        for idx, entry in enumerate(entries):
            meta = entry_meta[idx]
            result.append(
                {
                    "index": idx,
                    "window_name": entry["window_name"],
                    "group": entry["group"],
                    "available": meta["available"],
                    "lat": meta.get("lat"),
                    "lon": meta.get("lon"),
                    "num_positive": len(entry.get("positive_points", [])),
                    "num_negative": len(entry.get("negative_points", [])),
                }
            )
        return jsonify(result)

    @app.route("/api/entry/<int:idx>")
    def api_entry(idx: int) -> Response:
        if idx < 0 or idx >= len(entries):
            return Response("index out of range", status=404)
        entry = entries[idx]
        meta = entry_meta[idx]

        # Pre-compute pixel coords for all points so frontend can do hit-testing.
        projection = Projection.deserialize(entry["projection"])
        bounds = (
            entry["bounds"][0],
            entry["bounds"][1],
            entry["bounds"][2],
            entry["bounds"][3],
        )
        positive_pixels = []
        for pt in entry.get("positive_points", []):
            col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
            positive_pixels.append({"col": col, "row": row})
        negative_pixels = []
        for pt in entry.get("negative_points", []):
            col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
            negative_pixels.append({"col": col, "row": row})

        return jsonify(
            {
                "index": idx,
                "entry": entry,
                "meta": meta,
                "positive_pixels": positive_pixels,
                "negative_pixels": negative_pixels,
            }
        )

    @app.route("/api/update_points", methods=["POST"])
    def api_update_points() -> Response:
        body = request.get_json(force=True, silent=True) or {}
        idx = body.get("entry_idx")
        if idx is None or idx < 0 or idx >= len(entries):
            return jsonify({"ok": False, "error": "invalid entry_idx"}), 400

        action = body.get("action")
        with write_lock:
            entry = entries[idx]
            if action == "add_positive":
                lon = body["lon"]
                lat = body["lat"]
                point = {"lon": lon, "lat": lat}
                existing = entry.get("positive_points", [])
                if existing:
                    for field in (
                        "pre_change",
                        "first_date_change_noticeable",
                        "post_change",
                        "pre_category",
                        "post_category",
                    ):
                        if field in existing[0]:
                            point[field] = existing[0][field]
                entry.setdefault("positive_points", []).append(point)
            elif action == "add_negative":
                lon = body["lon"]
                lat = body["lat"]
                point = {"lon": lon, "lat": lat}
                entry.setdefault("negative_points", []).append(point)
            elif action == "remove_positive":
                point_idx = body.get("point_idx")
                pts = entry.get("positive_points", [])
                if point_idx is not None and 0 <= point_idx < len(pts):
                    pts.pop(point_idx)
            elif action == "remove_negative":
                point_idx = body.get("point_idx")
                pts = entry.get("negative_points", [])
                if point_idx is not None and 0 <= point_idx < len(pts):
                    pts.pop(point_idx)
            else:
                return jsonify({"ok": False, "error": "unknown action"}), 400
            _save_entries()

        return jsonify({"ok": True, "entry": entries[idx]})

    @app.route("/api/update_annotation", methods=["POST"])
    def api_update_annotation() -> Response:
        body = request.get_json(force=True, silent=True) or {}
        idx = body.get("entry_idx")
        point_idx = body.get("point_idx")
        if idx is None or idx < 0 or idx >= len(entries):
            return jsonify({"ok": False, "error": "invalid entry_idx"}), 400

        with write_lock:
            entry = entries[idx]
            pts = entry.get("positive_points", [])
            if point_idx is None or point_idx < 0 or point_idx >= len(pts):
                return jsonify({"ok": False, "error": "invalid point_idx"}), 400

            point = pts[point_idx]
            for field in (
                "pre_change",
                "first_date_change_noticeable",
                "post_change",
                "pre_category",
                "post_category",
            ):
                val = body.get(field)
                if val is not None:
                    if val == "":
                        point.pop(field, None)
                    else:
                        point[field] = val

            _save_entries()

        return jsonify({"ok": True, "entry": entries[idx]})

    @app.route("/api/pixel_to_lonlat", methods=["POST"])
    def api_pixel_to_lonlat() -> Response:
        """Convert pixel coords to lon/lat for an entry."""
        body = request.get_json(force=True, silent=True) or {}
        idx = body.get("entry_idx")
        if idx is None or idx < 0 or idx >= len(entries):
            return jsonify({"ok": False, "error": "invalid entry_idx"}), 400

        entry = entries[idx]
        projection = Projection.deserialize(entry["projection"])
        bounds = (
            entry["bounds"][0],
            entry["bounds"][1],
            entry["bounds"][2],
            entry["bounds"][3],
        )
        col = body["col"]
        row = body["row"]
        lon, lat = _pixel_to_lonlat(col, row, projection, bounds)
        return jsonify({"lon": lon, "lat": lat})

    @app.route("/image/sentinel2/<group>/<name>/<int:group_idx>")
    def image_sentinel2(group: str, name: str, group_idx: int) -> Response:
        window = window_cache.get((group, name))
        if window is None:
            return Response("window not found", status=404)

        raster_dir = window.get_raster_dir("sentinel2", s2_bands, group_idx=group_idx)
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

    @app.route("/image/points_overlay/<int:entry_idx>")
    def image_points_overlay(entry_idx: int) -> Response:
        if entry_idx < 0 or entry_idx >= len(entries):
            return Response("index out of range", status=404)

        entry = entries[entry_idx]
        projection = Projection.deserialize(entry["projection"])
        bounds = (
            entry["bounds"][0],
            entry["bounds"][1],
            entry["bounds"][2],
            entry["bounds"][3],
        )
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]

        selected_idx = request.args.get("selected", None)
        if selected_idx is not None:
            try:
                selected_idx = int(selected_idx)
            except ValueError:
                selected_idx = None

        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        CENTER_COLOR = (255, 255, 0, 255)

        for pt in entry.get("negative_points", []):
            col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
            cx, cy = math.floor(col), math.floor(row)
            x0 = cx - POINT_RADIUS
            y0 = cy - POINT_RADIUS
            x1 = cx + POINT_RADIUS
            y1 = cy + POINT_RADIUS
            draw.ellipse([x0, y0, x1, y1], fill=NEGATIVE_COLOR)
            if 0 <= cx < width and 0 <= cy < height:
                img.putpixel((cx, cy), CENTER_COLOR)

        for pi, pt in enumerate(entry.get("positive_points", [])):
            col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
            cx, cy = math.floor(col), math.floor(row)
            x0 = cx - POINT_RADIUS
            y0 = cy - POINT_RADIUS
            x1 = cx + POINT_RADIUS
            y1 = cy + POINT_RADIUS
            if pi == selected_idx:
                draw.ellipse(
                    [x0 - 2, y0 - 2, x1 + 2, y1 + 2], outline=SELECTED_COLOR, width=2
                )
            draw.ellipse([x0, y0, x1, y1], fill=POSITIVE_COLOR)
            if 0 <= cx < width and 0 <= cy < height:
                img.putpixel((cx, cy), CENTER_COLOR)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(
            buf.getvalue(),
            mimetype="image/png",
            headers={"Cache-Control": "no-cache"},
        )

    return app


def main() -> None:
    """Run the Change Finder V2 Annotation App server."""
    parser = argparse.ArgumentParser(description="Change Finder V2 Annotation App")
    parser.add_argument(
        "--json", required=True, dest="json_path", help="Path to v2 JSON file"
    )
    parser.add_argument("--ds-path", required=True, help="Path to rslearn dataset")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(args.json_path, args.ds_path)
    print(f"Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
