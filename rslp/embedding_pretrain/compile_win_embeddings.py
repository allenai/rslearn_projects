"""Export rslearn window embeddings from GeoJSON into sharded NPZ files.

This script scans a dataset group for per-window embedding GeoJSON files at:

    windows/<group>/<window_name>/layers/embeddings/data.geojson

Each GeoJSON is expected to contain exactly one point feature whose
``embedding`` property is the per-window embedding vector. The feature geometry
stores the top-left corner of the crop in the layer CRS, so this script shifts
the point by half the crop size (using the GeoJSON resolutions) to recover the
crop centroid before transforming it to WGS84 latitude/longitude.

Output format mirrors the existing embedding shard layout:

  - ``shard_0000.npz``, ``shard_0001.npz``, ...
  - ``index.csv`` with columns:
        sample_idx, shard, row, lat, lon, window_name

Usage example:
  python ./rslp/embedding_pretrain/compile_window_embeddings.py \
    --dataset {dataset_path} \
    --group s50ix24 \
    --output-path {output_path}
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

WGS84 = CRS.from_epsg(4326)
EMBEDDING_LAYER_NAME = "embeddings"
EMBEDDING_PROPERTY_NAME = "embedding"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the embedding export script."""
    parser = argparse.ArgumentParser(
        description="Export window embedding GeoJSONs into sharded NPZ files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Root path of the rslearn dataset.",
    )
    parser.add_argument(
        "--group",
        required=True,
        help="Window group to export (for example: s50ix24).",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory where shard_*.npz and index.csv will be written.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=96,
        help="Crop size in pixels at the layer resolution.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Number of embeddings per NPZ shard.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Only process the first 10 windows in the group.",
    )
    return parser.parse_args(argv)


def get_window_dirs(dataset_path: Path, group: str) -> list[Path]:
    """Return sorted list of window directories for the given group."""
    windows_dir = dataset_path / "windows" / group
    if not windows_dir.exists():
        raise FileNotFoundError(f"Windows directory not found: {windows_dir}")

    window_dirs = sorted(path for path in windows_dir.iterdir() if path.is_dir())
    if not window_dirs:
        raise FileNotFoundError(f"No window directories found in {windows_dir}")
    return window_dirs


def get_embedding_geojson_path(window_dir: Path) -> Path:
    """Return the path to the embedding GeoJSON file for a window."""
    return window_dir / "layers" / EMBEDDING_LAYER_NAME / "data.geojson"


def get_transformer(
    crs_str: str,
    transformer_cache: dict[str, Transformer],
) -> Transformer:
    """Get or create a cached CRS-to-WGS84 transformer."""
    transformer = transformer_cache.get(crs_str)
    if transformer is None:
        transformer = Transformer.from_crs(
            CRS.from_string(crs_str), WGS84, always_xy=True
        )
        transformer_cache[crs_str] = transformer
    return transformer


def load_window_embedding(
    geojson_path: Path,
    crop_size: int,
    transformer_cache: dict[str, Transformer],
) -> tuple[np.ndarray, float, float]:
    """Load an embedding vector and WGS84 centroid from a GeoJSON file."""
    with geojson_path.open() as f:
        data = json.load(f)

    properties = data["properties"]
    crs_str = properties["crs"]
    x_resolution = float(properties["x_resolution"])
    y_resolution = float(properties["y_resolution"])

    features = data["features"]
    if len(features) != 1:
        raise ValueError(
            f"Expected exactly one feature in {geojson_path}, found {len(features)}"
        )

    feature = features[0]
    geometry = feature["geometry"]
    if geometry["type"] != "Point":
        raise ValueError(
            f"Expected Point geometry in {geojson_path}, found {geometry['type']}"
        )

    embedding = np.asarray(
        feature["properties"][EMBEDDING_PROPERTY_NAME],
        dtype=np.float32,
    )
    if embedding.ndim != 1:
        raise ValueError(
            f"Expected 1D embedding in {geojson_path}, got shape {embedding.shape}"
        )

    top_left_col, top_left_row = geometry["coordinates"][:2]
    centroid_col = top_left_col + (crop_size / 2.0)
    centroid_row = top_left_row + (crop_size / 2.0)

    # Rslearn writes this GeoJSON in pixel mode, with the projection stored in the
    # FeatureCollection properties. Convert the centroid from pixel coordinates to
    # CRS coordinates before transforming to WGS84.
    centroid_x = centroid_col * x_resolution
    centroid_y = centroid_row * y_resolution

    transformer = get_transformer(crs_str, transformer_cache)
    lon, lat = transformer.transform(centroid_x, centroid_y)
    return embedding, float(lat), float(lon)


def write_shard(output_dir: Path, shard_idx: int, embeddings: list[np.ndarray]) -> str:
    """Stack embeddings into an NPZ shard and write it to disk."""
    shard_name = f"shard_{shard_idx:04d}.npz"
    shard_path = output_dir / shard_name
    stacked = np.stack(embeddings, axis=0).astype(np.float32, copy=False)
    np.savez(shard_path, embeddings=stacked)
    logger.info(
        "Wrote %s with shape %s",
        shard_name,
        tuple(stacked.shape),
    )
    return shard_name


def export_embeddings(
    dataset_path: Path,
    group: str,
    output_dir: Path,
    crop_size: int,
    chunk_size: int,
    test_mode: bool,
) -> None:
    """Export per-window embeddings from GeoJSON files into sharded NPZ files."""
    if crop_size <= 0:
        raise ValueError(f"crop_size must be > 0, got {crop_size}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.csv"
    window_dirs = get_window_dirs(dataset_path, group)
    if test_mode:
        window_dirs = window_dirs[:10]
        logger.info(
            "Test mode enabled: limiting export to first %d windows", len(window_dirs)
        )

    transformer_cache: dict[str, Transformer] = {}
    shard_embeddings: list[np.ndarray] = []
    pending_index_rows: list[dict[str, object]] = []
    sample_idx = 0
    shard_idx = 0
    embedding_dim: int | None = None
    missing_count = 0
    error_count = 0

    with index_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["sample_idx", "shard", "row", "lat", "lon", "window_name"],
        )
        writer.writeheader()

        for i, window_dir in enumerate(window_dirs, start=1):
            geojson_path = get_embedding_geojson_path(window_dir)
            if not geojson_path.exists():
                missing_count += 1
                if missing_count <= 10:
                    logger.warning(
                        "Skipping %s (missing %s)", window_dir.name, geojson_path
                    )
                continue

            try:
                embedding, lat, lon = load_window_embedding(
                    geojson_path=geojson_path,
                    crop_size=crop_size,
                    transformer_cache=transformer_cache,
                )
            except Exception as exc:
                error_count += 1
                logger.warning("Skipping %s (%s)", window_dir.name, exc)
                continue

            if embedding_dim is None:
                embedding_dim = int(embedding.shape[0])
                logger.info("Detected embedding dimension: %d", embedding_dim)
            elif embedding.shape[0] != embedding_dim:
                raise ValueError(
                    f"Inconsistent embedding dimension for {window_dir.name}: "
                    f"expected {embedding_dim}, got {embedding.shape[0]}"
                )

            row_in_shard = len(shard_embeddings)
            shard_embeddings.append(embedding)
            pending_index_rows.append(
                {
                    "sample_idx": sample_idx,
                    "shard": f"shard_{shard_idx:04d}.npz",
                    "row": row_in_shard,
                    "lat": lat,
                    "lon": lon,
                    "window_name": window_dir.name,
                }
            )
            sample_idx += 1

            if len(shard_embeddings) == chunk_size:
                shard_name = write_shard(output_dir, shard_idx, shard_embeddings)
                for row in pending_index_rows:
                    row["shard"] = shard_name
                    writer.writerow(row)
                shard_embeddings.clear()
                pending_index_rows.clear()
                shard_idx += 1

            if i % 10000 == 0:
                logger.info(
                    "Scanned %d/%d windows; exported %d embeddings so far",
                    i,
                    len(window_dirs),
                    sample_idx,
                )

        if shard_embeddings:
            shard_name = write_shard(output_dir, shard_idx, shard_embeddings)
            for row in pending_index_rows:
                row["shard"] = shard_name
                writer.writerow(row)
            shard_embeddings.clear()
            pending_index_rows.clear()
            shard_idx += 1

    if sample_idx == 0:
        raise RuntimeError(
            f"No embeddings were exported from dataset={dataset_path} group={group}"
        )

    logger.info("Finished export")
    logger.info("  windows scanned:    %d", len(window_dirs))
    logger.info("  embeddings written: %d", sample_idx)
    logger.info("  shards written:     %d", shard_idx)
    logger.info("  missing layers:     %d", missing_count)
    logger.info("  malformed windows:  %d", error_count)
    logger.info("  index csv:          %s", index_path)


def main(argv: list[str] | None = None) -> None:
    """Run the embedding export pipeline from CLI arguments."""
    args = parse_args(argv)
    export_embeddings(
        dataset_path=Path(args.dataset),
        group=args.group,
        output_dir=Path(args.output_path),
        crop_size=args.crop_size,
        chunk_size=args.chunk_size,
        test_mode=args.test_mode,
    )


if __name__ == "__main__":
    main()
