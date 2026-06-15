"""Shared helpers for the embeddings change-evaluation scripts.

Both the cosine (``predict_change.py``) and linear-probe (``linear_probe.py``) modes
read a per-point embedding from the ``embeddings`` raster layer written for each
prediction window, and write the same standardized CSV schema (shared with the
WorldCover flow) so a single metric script can consume any method's output.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.warp
from rasterio.crs import CRS
from upath import UPath

PREDICTION_GROUP = "predict"
TRAIN_GROUP = "train"

EMBEDDINGS_LAYER = "embeddings"

# Standardized output schema, shared with the WorldCover eval flow. Embeddings give no
# land-cover class, so the category / predicted_changed columns stay blank here.
MERGED_FIELDS = [
    "row_index",
    "lon",
    "lat",
    "src_year",
    "dst_year",
    "has_changed",
    "src_category",
    "dst_category",
    "has_prediction",
    "predicted_changed",
    "change_score",
    "pred_src_category",
    "pred_dst_category",
]


class PointRow:
    """One eval/train CSV row (a single annotated point)."""

    def __init__(self, idx: int, raw: dict[str, str]) -> None:
        """Parse a CSV row into a point with the fields used downstream."""
        self.row_index = idx
        self.lon = float(raw["lon"])
        self.lat = float(raw["lat"])
        self.src_year = int(raw["src_year"])
        self.dst_year = int(raw["dst_year"])
        self.has_changed = raw["has_changed"].strip().lower() == "true"
        self.src_category = raw.get("src_category", "")
        self.dst_category = raw.get("dst_category", "")


def load_points(csv_path: Path) -> list[PointRow]:
    """Load all points from an evaluation/training CSV in row order."""
    with csv_path.open(newline="") as f:
        return [PointRow(idx, raw) for idx, raw in enumerate(csv.DictReader(f))]


def base_row(point: PointRow) -> dict[str, Any]:
    """Build a merged-CSV row with ground-truth fields and blank prediction fields."""
    return {
        "row_index": point.row_index,
        "lon": point.lon,
        "lat": point.lat,
        "src_year": point.src_year,
        "dst_year": point.dst_year,
        "has_changed": point.has_changed,
        "src_category": point.src_category,
        "dst_category": point.dst_category,
        "has_prediction": False,
        "predicted_changed": "",
        "change_score": "",
        "pred_src_category": "",
        "pred_dst_category": "",
    }


def write_merged_csv(rows: list[dict[str, Any]], output: Path) -> None:
    """Write merged rows to a CSV using the shared schema."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MERGED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _find_embeddings_geotiff(window_dir: UPath) -> UPath | None:
    """Find the embeddings geotiff under a window directory, if materialized."""
    matches = list(window_dir.glob(f"layers/{EMBEDDINGS_LAYER}/*/geotiff.tif"))
    if not matches:
        return None
    return matches[0]


def read_embedding_at_point(
    ds_upath: UPath, group: str, window_name: str, lon: float, lat: float
) -> np.ndarray | None:
    """Read the embedding vector at a lon/lat for one prediction window.

    Uses the embeddings geotiff's own CRS + transform to locate the pixel, so it works
    regardless of the output resolution (AlphaEarth is at the 10 m window resolution,
    while OlmoEarth embeddings are downsampled by the model patch size).

    Returns None if the geotiff is missing, the point falls outside it, or the pixel is
    all-nodata (AlphaEarth nodata is -1.0 across all bands; an all-zero pixel is also
    treated as missing).
    """
    window_dir = ds_upath / "windows" / group / window_name
    tif_path = _find_embeddings_geotiff(window_dir)
    if tif_path is None:
        return None

    with tif_path.open("rb") as f:
        with rasterio.open(f) as src:
            xs, ys = rasterio.warp.transform(CRS.from_epsg(4326), src.crs, [lon], [lat])
            row, col = src.index(xs[0], ys[0])
            if not (0 <= row < src.height and 0 <= col < src.width):
                return None
            vec = src.read()[:, row, col].astype(np.float64)

    if np.all(vec == 0) or np.all(vec <= -1.0):
        return None
    return vec


def iter_points_with_embeddings(
    ds_upath: UPath,
    points: list[PointRow],
    group: str,
    name_prefix: str,
) -> Iterator[tuple[PointRow, np.ndarray, np.ndarray]]:
    """Yield (point, e_src, e_dst) for points whose src and dst embeddings both exist.

    Window names follow ``{name_prefix}{idx:06d}_{kind}`` (kind in src/dst).
    """
    for point in points:
        e_src = read_embedding_at_point(
            ds_upath,
            group,
            f"{name_prefix}{point.row_index:06d}_src",
            point.lon,
            point.lat,
        )
        e_dst = read_embedding_at_point(
            ds_upath,
            group,
            f"{name_prefix}{point.row_index:06d}_dst",
            point.lon,
            point.lat,
        )
        if e_src is None or e_dst is None:
            continue
        yield point, e_src, e_dst
