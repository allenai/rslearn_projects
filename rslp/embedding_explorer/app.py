"""Flask app for exploring embedding similarity in rslearn datasets.

Usage:
    python -m rslp.embedding_explorer.app --dataset-path /path/to/dataset --port 5000
    python -m rslp.embedding_explorer.app --dataset-path /path/to/dataset \
        --embedding-layer embeddings aef --port 5000
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform
import rasterio.warp
from flask import Flask, Response, render_template, request
from PIL import Image
from pyproj import Transformer
from rasterio.enums import Resampling
from sklearn.linear_model import LogisticRegression

EPSG_3857 = "EPSG:3857"


def find_geotiff(layer_dir: Path) -> Path | None:
    """Find geotiff.tif inside a layer directory (inside the bandset subdir)."""
    for subdir in layer_dir.iterdir():
        if subdir.is_dir() and subdir.name != "completed":
            tif = subdir / "geotiff.tif"
            if tif.exists():
                return tif
    return None


def reproject_to_webmercator(
    data: np.ndarray,
    src_crs: str,
    src_transform: rasterio.transform.Affine,
    src_shape: tuple,
) -> tuple[np.ndarray, rasterio.transform.Affine, tuple[int, int]]:
    """Reproject a (C, H, W) or (H, W) array to EPSG:3857.

    Returns (reprojected_data, dst_transform, (dst_height, dst_width)).
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False

    dst_transform, dst_width, dst_height = rasterio.warp.calculate_default_transform(
        src_crs,
        EPSG_3857,
        src_shape[1],
        src_shape[0],
        *rasterio.transform.array_bounds(src_shape[0], src_shape[1], src_transform),
    )

    dst = np.zeros((data.shape[0], dst_height, dst_width), dtype=data.dtype)
    for i in range(data.shape[0]):
        rasterio.warp.reproject(
            source=data[i],
            destination=dst[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=EPSG_3857,
            resampling=Resampling.nearest,
        )

    if squeeze:
        dst = dst[0]
    return dst, dst_transform, (dst_height, dst_width)


def webmercator_bounds(
    transform: rasterio.transform.Affine, shape: tuple[int, int]
) -> tuple[float, float, float, float]:
    """Get (left, bottom, right, top) in EPSG:3857 meters from a raster grid."""
    return rasterio.transform.array_bounds(shape[0], shape[1], transform)


def webmercator_bounds_to_latlon(
    transform: rasterio.transform.Affine, shape: tuple[int, int]
) -> list[list[float]]:
    """Get [[lat_min, lon_min], [lat_max, lon_max]] from a Web Mercator raster."""
    bounds = webmercator_bounds(transform, shape)
    transformer = Transformer.from_crs(EPSG_3857, "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(bounds[0], bounds[1])
    lon_max, lat_max = transformer.transform(bounds[2], bounds[3])
    return [[lat_min, lon_min], [lat_max, lon_max]]


def latlon_to_pixel(
    lat: float, lon: float, crs: str, transform: rasterio.transform.Affine, shape: tuple
) -> tuple[int, int]:
    """Convert lat/lon to pixel row, col using the raster's transform."""
    proj_transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = proj_transformer.transform(lon, lat)
    row, col = rasterio.transform.rowcol(transform, x, y)
    row = max(0, min(int(row), shape[0] - 1))
    col = max(0, min(int(col), shape[1] - 1))
    return row, col


def render_rgb_png(data: np.ndarray, bands: tuple[int, int, int] = (0, 1, 2)) -> bytes:
    """Render selected bands as a stretched RGB PNG."""
    rgb = np.stack([data[b] for b in bands], axis=-1).astype(np.float32)
    for i in range(3):
        band = rgb[:, :, i]
        valid = band[np.isfinite(band) & (band != 0)]
        if len(valid) == 0:
            continue
        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
        if hi - lo < 1e-6:
            hi = lo + 1
        rgb[:, :, i] = (band - lo) / (hi - lo)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def render_rgb_png_with_alpha(
    data: np.ndarray, bands: tuple[int, int, int] = (0, 1, 2)
) -> bytes:
    """Render selected bands as stretched RGBA PNG (nodata pixels transparent)."""
    rgb = np.stack([data[b] for b in bands], axis=-1).astype(np.float32)
    # Mask where all selected bands are zero (nodata from reprojection)
    nodata_mask = np.all(rgb == 0, axis=-1)

    for i in range(3):
        band = rgb[:, :, i]
        valid = band[np.isfinite(band) & (~nodata_mask)]
        if len(valid) == 0:
            continue
        lo = np.percentile(valid, 2)
        hi = np.percentile(valid, 98)
        if hi - lo < 1e-6:
            hi = lo + 1
        rgb[:, :, i] = (band - lo) / (hi - lo)

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    alpha = np.where(nodata_mask, 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb, alpha])
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compute_cosine_similarity(
    embeddings: np.ndarray, ref_vector: np.ndarray
) -> np.ndarray:
    """Cosine similarity between all pixels and a reference vector.

    Returns (H, W) float array in [-1, 1].
    """
    ref_norm = np.linalg.norm(ref_vector)
    if ref_norm < 1e-8:
        return np.zeros(embeddings.shape[1:], dtype=np.float32)
    ref_unit = ref_vector / ref_norm
    pixel_norms = np.linalg.norm(embeddings, axis=0)
    valid = pixel_norms > 1e-8
    dot = np.tensordot(ref_unit, embeddings, axes=(0, 0))
    similarity = np.zeros(embeddings.shape[1:], dtype=np.float32)
    similarity[valid] = dot[valid] / pixel_norms[valid]
    return np.clip(similarity, -1.0, 1.0)


def compute_knn(embeddings: np.ndarray, points: list[dict], k: int) -> np.ndarray:
    """KNN classification: positive_votes / k for each pixel.

    Returns (H, W) float array in [0, 1].
    """
    C, H, W = embeddings.shape
    flat = embeddings.reshape(C, -1).T  # (N, C)
    flat_norms = np.linalg.norm(flat, axis=1, keepdims=True)
    flat_norms[flat_norms < 1e-8] = 1.0
    flat_unit = flat / flat_norms

    n_points = len(points)
    labels = np.array([1.0 if p["label"] == "positive" else 0.0 for p in points])
    point_vecs = np.stack([p["vector"] for p in points])
    point_norms = np.linalg.norm(point_vecs, axis=1, keepdims=True)
    point_norms[point_norms < 1e-8] = 1.0
    point_unit = point_vecs / point_norms

    sims = flat_unit @ point_unit.T  # (N, n_points)

    effective_k = min(k, n_points)
    if effective_k == n_points:
        top_k_indices = np.broadcast_to(np.arange(n_points), (sims.shape[0], n_points))
    else:
        top_k_indices = np.argpartition(sims, -effective_k, axis=1)[:, -effective_k:]

    top_k_labels = labels[top_k_indices]
    score = top_k_labels.mean(axis=1)
    return score.reshape(H, W).astype(np.float32)


def compute_linear_probe(embeddings: np.ndarray, points: list[dict]) -> np.ndarray:
    """Train a logistic regression on labeled points and predict per-pixel probability.

    Returns (H, W) float array in [0, 1] giving probability of the positive class.
    """
    C, H, W = embeddings.shape

    X = np.stack([p["vector"] for p in points]).astype(np.float32)
    y = np.array([1 if p["label"] == "positive" else 0 for p in points], dtype=np.int64)

    # max_iter is generous; with ~10-30 samples LBFGS converges in a handful of steps.
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(X, y)

    pos_idx = int(np.where(clf.classes_ == 1)[0][0])
    flat = embeddings.reshape(C, -1).T.astype(np.float32)
    probs = clf.predict_proba(flat)[:, pos_idx]
    return probs.reshape(H, W).astype(np.float32)


def similarity_to_png(similarity: np.ndarray, mode: str) -> bytes:
    """Encode similarity as 8-bit grayscale PNG.

    Cosine: [-1, 1] -> [0, 255]
    KNN / linear_probe: [0, 1] -> [0, 255]
    """
    if mode == "cosine":
        img_data = ((similarity + 1.0) / 2.0 * 255.0).astype(np.uint8)
    else:
        img_data = (similarity * 255.0).astype(np.uint8)
    img = Image.fromarray(img_data, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_dataset(dataset_path: Path, embedding_layers: list[str]) -> dict:
    """Load dataset: embeddings into memory, image layer paths stored for on-demand serving."""
    windows_dir = dataset_path / "windows"
    dataset: dict = {"windows": {}, "embedding_layers": embedding_layers}

    for group_dir in sorted(windows_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        for window_dir in sorted(group_dir.iterdir()):
            if not window_dir.is_dir():
                continue
            meta_path = window_dir / "metadata.json"
            if not meta_path.exists():
                continue

            with open(meta_path) as f:
                metadata = json.load(f)

            layers_dir = window_dir / "layers"
            if not layers_dir.exists():
                continue

            window_key = f"{group_dir.name}/{window_dir.name}"
            window_info: dict = {
                "metadata": metadata,
                "embeddings": {},
                "image_layers": {},
                "path": window_dir,
            }

            for layer_dir in sorted(layers_dir.iterdir()):
                if not layer_dir.is_dir():
                    continue
                layer_name = layer_dir.name
                base_name = layer_name.split(".")[0]

                tif = find_geotiff(layer_dir)
                if tif is None:
                    continue

                if base_name in embedding_layers:
                    print(f"  Loading embedding: {window_key}/{layer_name}")
                    with rasterio.open(tif) as src:
                        data = src.read().astype(np.float32)
                        src_crs = str(src.crs)
                        src_transform = src.transform
                        src_shape = (src.height, src.width)
                    print(f"    shape: {data.shape}")

                    print("    computing Web Mercator reprojection info...")
                    _, wm_transform, wm_shape = reproject_to_webmercator(
                        data[:1], src_crs, src_transform, src_shape
                    )
                    wm_b = webmercator_bounds(wm_transform, wm_shape)
                    print(f"    Web Mercator shape: {wm_shape}")

                    window_info["embeddings"][base_name] = {
                        "data": data,
                        "crs": src_crs,
                        "transform": src_transform,
                        "shape": src_shape,
                        "wm_transform": wm_transform,
                        "wm_shape": wm_shape,
                        "wm_bounds": wm_b,
                    }
                else:
                    with rasterio.open(tif) as src:
                        layer_crs = str(src.crs)
                        layer_transform = src.transform
                        layer_shape = (src.height, src.width)
                    wm_dst_transform, wm_dst_width, wm_dst_height = (
                        rasterio.warp.calculate_default_transform(
                            layer_crs,
                            EPSG_3857,
                            layer_shape[1],
                            layer_shape[0],
                            *rasterio.transform.array_bounds(
                                layer_shape[0], layer_shape[1], layer_transform
                            ),
                        )
                    )
                    layer_wm_bounds = webmercator_bounds(
                        wm_dst_transform, (wm_dst_height, wm_dst_width)
                    )
                    if base_name not in window_info["image_layers"]:
                        window_info["image_layers"][base_name] = []
                    window_info["image_layers"][base_name].append(
                        {
                            "name": layer_name,
                            "path": str(tif),
                            "wm_bounds": layer_wm_bounds,
                        }
                    )

            if not window_info["embeddings"]:
                print(f"  WARNING: no embedding layers found in {window_key}, skipping")
                continue

            dataset["windows"][window_key] = window_info
            print(f"Loaded window: {window_key}")
            for emb_name, emb_info in window_info["embeddings"].items():
                print(f"  embedding '{emb_name}': shape={emb_info['data'].shape}")
                print(f"    mercator bounds: {emb_info['wm_bounds']}")
            print(f"  image layers: {list(window_info['image_layers'].keys())}")

    return dataset


def reproject_single_band_to_wm(
    data: np.ndarray,
    src_crs: str,
    src_transform: rasterio.transform.Affine,
    src_shape: tuple,
    wm_transform: rasterio.transform.Affine,
    wm_shape: tuple,
) -> np.ndarray:
    """Reproject a single (H, W) array to a pre-computed Web Mercator grid."""
    dst = np.zeros(wm_shape, dtype=data.dtype)
    rasterio.warp.reproject(
        source=data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=wm_transform,
        dst_crs=EPSG_3857,
        resampling=Resampling.nearest,
    )
    return dst


def create_app(dataset_path: Path, embedding_layers: list[str] | None = None) -> Flask:
    """Create and configure the Flask application."""
    if embedding_layers is None:
        embedding_layers = ["embeddings"]
    app_dir = Path(__file__).parent
    app = Flask(
        __name__,
        template_folder=str(app_dir / "templates"),
        static_folder=str(app_dir / "static"),
    )

    print(f"Loading dataset from {dataset_path}...")
    print(f"Embedding layers: {embedding_layers}")
    dataset = load_dataset(dataset_path, embedding_layers)
    print(f"Loaded {len(dataset['windows'])} window(s)")

    @app.route("/")
    def index() -> str:
        windows_info: dict = {}
        for key, win in dataset["windows"].items():
            # Use the first available embedding layer's bounds as the window bounds
            first_emb = next(iter(win["embeddings"].values()))
            wm = first_emb["wm_bounds"]
            windows_info[key] = {
                "mercator_bounds": [wm[0], wm[1], wm[2], wm[3]],
                "embedding_layers": list(win["embeddings"].keys()),
                "image_layers": {
                    name: [
                        {
                            "name": g["name"],
                            "mercator_bounds": list(g["wm_bounds"]),
                        }
                        for g in groups
                    ]
                    for name, groups in win["image_layers"].items()
                },
            }
        return render_template(
            "index.html",
            windows=json.dumps(windows_info),
            embedding_layers=json.dumps(embedding_layers),
        )

    @app.route("/api/image/<path:layer_path>")
    def serve_image(layer_path: str) -> Response | tuple[str, int]:
        """Serve a layer as RGB PNG reprojected to Web Mercator.

        layer_path: window_key/layer_name (e.g. default/default/sentinel2_l2a)
        Query params:
            bands: comma-separated 0-based band indices (default: 0,1,2)
        """
        parts = layer_path.split("/")
        if len(parts) < 3:
            return "Invalid path", 400
        window_key = f"{parts[0]}/{parts[1]}"
        layer_name = "/".join(parts[2:])

        win = dataset["windows"].get(window_key)
        if win is None:
            return "Window not found", 404

        bands_param = request.args.get("bands", "0,1,2")
        bands = tuple(int(b) for b in bands_param.split(","))

        # Serve embedding RGB from memory (reproject to WM)
        if layer_name in win["embeddings"]:
            emb = win["embeddings"][layer_name]
            data = emb["data"]
            selected = np.stack([data[b] for b in bands])
            reprojected, wm_transform, wm_shape = reproject_to_webmercator(
                selected, emb["crs"], emb["transform"], emb["shape"]
            )
            png = render_rgb_png_with_alpha(reprojected, (0, 1, 2))
            left, bottom, right, top = webmercator_bounds(wm_transform, wm_shape)
            return Response(
                png,
                mimetype="image/png",
                headers={
                    "X-Mercator-Bounds": f"{left},{bottom},{right},{top}",
                    "Cache-Control": "public, max-age=3600",
                },
            )

        # Find in image layers
        tif_path = None
        for base_name, groups in win["image_layers"].items():
            for g in groups:
                if g["name"] == layer_name:
                    tif_path = g["path"]
                    break
            if tif_path:
                break

        if tif_path is None:
            return "Layer not found", 404

        with rasterio.open(tif_path) as src:
            band_indices = [b + 1 for b in bands]  # rasterio is 1-indexed
            data = src.read(indexes=band_indices).astype(np.float32)
            src_crs = str(src.crs)
            src_transform = src.transform
            src_shape = (src.height, src.width)

        reprojected, wm_transform, wm_shape = reproject_to_webmercator(
            data, src_crs, src_transform, src_shape
        )
        png = render_rgb_png_with_alpha(reprojected, (0, 1, 2))
        left, bottom, right, top = webmercator_bounds(wm_transform, wm_shape)
        return Response(
            png,
            mimetype="image/png",
            headers={
                "X-Mercator-Bounds": f"{left},{bottom},{right},{top}",
                "Cache-Control": "public, max-age=3600",
            },
        )

    @app.route("/api/similarity", methods=["POST"])
    def compute_similarity_route() -> Response | tuple[str, int]:
        """Compute similarity in native CRS, reproject result to Web Mercator.

        Request JSON:
            mode: "cosine" or "knn" or "linear_probe"
            points: list of {lat, lon, label}
            k: int (for knn mode)
            window: window key
            layer: embedding layer name (default: first available)
        """
        body = request.get_json()
        window_key = body.get("window")
        mode = body.get("mode", "cosine")
        points = body.get("points", [])
        k = body.get("k", 3)
        layer_name = body.get("layer")

        win = dataset["windows"].get(window_key)
        if win is None:
            return "Window not found", 404

        if not layer_name:
            layer_name = next(iter(win["embeddings"]))
        if layer_name not in win["embeddings"]:
            return f"Embedding layer '{layer_name}' not found", 404

        emb = win["embeddings"][layer_name]
        embeddings = emb["data"]
        crs = emb["crs"]
        transform = emb["transform"]
        shape = embeddings.shape[1:]

        if not points:
            return "No points provided", 400

        if mode == "cosine":
            pos_points = [p for p in points if p.get("label") == "positive"]
            if not pos_points:
                return "No positive points for cosine mode", 400
            p = pos_points[0]
            row, col = latlon_to_pixel(p["lat"], p["lon"], crs, transform, shape)
            ref_vector = embeddings[:, row, col]
            similarity = compute_cosine_similarity(embeddings, ref_vector)
        elif mode == "linear_probe":
            labeled_points = []
            for p in points:
                row, col = latlon_to_pixel(p["lat"], p["lon"], crs, transform, shape)
                labeled_points.append(
                    {
                        "vector": embeddings[:, row, col].copy(),
                        "label": p.get("label", "positive"),
                    }
                )
            n_pos = sum(1 for p in labeled_points if p["label"] == "positive")
            n_neg = len(labeled_points) - n_pos
            if n_pos < 1 or n_neg < 1:
                return (
                    "Linear probe requires at least one positive and one negative point",
                    400,
                )
            similarity = compute_linear_probe(embeddings, labeled_points)
        else:
            labeled_points = []
            for p in points:
                row, col = latlon_to_pixel(p["lat"], p["lon"], crs, transform, shape)
                labeled_points.append(
                    {
                        "vector": embeddings[:, row, col].copy(),
                        "label": p.get("label", "positive"),
                    }
                )
            similarity = compute_knn(embeddings, labeled_points, k)

        # Reproject similarity to Web Mercator
        sim_wm = reproject_single_band_to_wm(
            similarity,
            crs,
            transform,
            shape,
            emb["wm_transform"],
            emb["wm_shape"],
        )

        png = similarity_to_png(sim_wm, mode)
        left, bottom, right, top = webmercator_bounds(
            emb["wm_transform"], emb["wm_shape"]
        )
        return Response(
            png,
            mimetype="image/png",
            headers={
                "X-Mercator-Bounds": f"{left},{bottom},{right},{top}",
                "X-Mode": mode,
            },
        )

    return app


def main() -> None:
    """CLI entry point for the embedding explorer web app."""
    parser = argparse.ArgumentParser(description="Embedding similarity explorer")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument(
        "--embedding-layer",
        nargs="+",
        default=["embeddings"],
        help="Embedding layer name(s) to load (default: 'embeddings')",
    )
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    app = create_app(args.dataset_path, embedding_layers=args.embedding_layer)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
