"""Derive change scores from WorldCover land-cover predictions at eval points.

The WorldCover model is run twice per point (on ``src_year`` and ``dst_year``
imagery) by ``create_prediction_dataset_from_csv.py`` + ``rslearn model predict``,
which writes a 13-channel land-cover probability raster (the ``output`` layer) for
each window. This script samples both windows' rasters at the annotated point's
pixel, compares the two probability distributions, and writes one standardized CSV
row per point.

Per point (nodata at index 0 is excluded when taking argmax / categories):
- ``pred_src_category`` / ``pred_dst_category``: argmax land-cover class of p_src /
  p_dst.
- ``predicted_changed``: argmax(p_src) != argmax(p_dst).
- ``change_score`` (the "method"): ``max(p_src) - p_dst[argmax(p_src)]``, i.e. the drop
  in the src-year top class's probability from src to dst year. This is in [0, 1];
  higher means more change (a downstream metric script can use it directly for AUROC).

The output CSV schema is shared across change methods (e.g. the LCC model), so a
single metric script can consume any method's CSV.

    python -m rslp.change_finder_v2.evaluation.worldcover.predict_change \
        --csv eval.csv --ds-path "$EVAL_DS" --output eval_worldcover.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import shapely
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.raster_format import get_bandset_dirname
from upath import UPath

from rslp.change_finder_v2.lcc_model.postprocess import LC_CLASS_NAMES, NUM_LC_CLASSES

PREDICTION_GROUP = "predict"

OUTPUT_LAYER = "output"
# Must match the band names of the "output" layer in config_predict.json.
OUTPUT_BANDS = [
    "nodata",
    "bare",
    "burnt",
    "crops",
    "fallow",
    "grassland",
    "lichen",
    "shrub",
    "snow",
    "tree",
    "urban",
    "water",
    "wetland",
]

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


def _src_class_prob_drop(p_src: np.ndarray, p_dst: np.ndarray) -> float:
    """Change score: max(p_src) - p_dst[argmax(p_src)] (in [0, 1], higher = more change).

    The drop in the src-year top class's probability from src to dst year: a large
    drop means that land-cover class is much less likely in the dst year (change).
    """
    c = int(p_src[1:].argmax()) + 1
    return float(p_src[c]) - float(p_dst[c])


# Registry of change-score methods. Each maps (p_src, p_dst) -> float; structured so
# more methods can be added later.
METHODS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "src_class_prob_drop": _src_class_prob_drop,
}
DEFAULT_METHOD = "src_class_prob_drop"


def _get_geotiff_path(window_dir: UPath) -> UPath | None:
    """Find the output land-cover probability geotiff under a window directory."""
    bandset_dir = get_bandset_dirname(OUTPUT_BANDS)
    tif = window_dir / "layers" / OUTPUT_LAYER / bandset_dir / "geotiff.tif"
    if tif.exists():
        return tif
    return None


def _point_pixel(
    lon: float, lat: float, projection: Projection, bounds: list[int]
) -> tuple[int, int]:
    """Convert lon/lat to (col, row) pixel coords within the window bounds."""
    st = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    projected = st.to_projection(projection)
    col = math.floor(projected.shp.x) - bounds[0]
    row = math.floor(projected.shp.y) - bounds[1]
    return col, row


def _read_probs_at_point(
    ds_upath: UPath, window_name: str, lon: float, lat: float
) -> np.ndarray | None:
    """Read the 13-class probability vector at a point for one prediction window."""
    window_dir = ds_upath / "windows" / PREDICTION_GROUP / window_name
    tif_path = _get_geotiff_path(window_dir)
    metadata_path = window_dir / "metadata.json"
    if tif_path is None or not metadata_path.exists():
        return None

    with metadata_path.open() as f:
        metadata = json.load(f)
    projection = Projection.deserialize(metadata["projection"])
    bounds = metadata["bounds"]
    try:
        col, row = _point_pixel(lon, lat, projection, bounds)
    except Exception as e:
        print(
            f"[worldcover.predict_change] {window_name}: failed to project point "
            f"(lon={lon}, lat={lat}) into window CRS {projection.crs}; skipping. "
            f"Error: {e}"
        )
        return None

    with tif_path.open("rb") as f:
        with rasterio.open(f) as src:
            arr = src.read()
    _, h, w = arr.shape
    col = min(max(col, 0), w - 1)
    row = min(max(row, 0), h - 1)
    probs = arr[:NUM_LC_CLASSES, row, col].astype(np.float64)
    return probs


def predict_change(
    csv_path: Path, ds_path: str, output: Path, method: str
) -> list[dict[str, Any]]:
    """Join WorldCover src/dst predictions with the CSV and return merged rows."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choices: {sorted(METHODS)}")
    score_fn = METHODS[method]
    ds_upath = UPath(ds_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    merged: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        lon = float(row["lon"])
        lat = float(row["lat"])
        gt_changed = row["has_changed"].strip().lower() == "true"

        out: dict[str, Any] = {
            "row_index": idx,
            "lon": lon,
            "lat": lat,
            "src_year": row["src_year"],
            "dst_year": row["dst_year"],
            "has_changed": gt_changed,
            "src_category": row.get("src_category", ""),
            "dst_category": row.get("dst_category", ""),
            "has_prediction": False,
            "predicted_changed": "",
            "change_score": "",
            "pred_src_category": "",
            "pred_dst_category": "",
        }

        p_src = _read_probs_at_point(ds_upath, f"eval_{idx:06d}_src", lon, lat)
        p_dst = _read_probs_at_point(ds_upath, f"eval_{idx:06d}_dst", lon, lat)
        if p_src is not None and p_dst is not None:
            # Argmax over classes 1..12 (skip nodata at index 0), then +1.
            src_id = int(p_src[1:].argmax()) + 1
            dst_id = int(p_dst[1:].argmax()) + 1
            out["predicted_changed"] = bool(src_id != dst_id)
            out["change_score"] = round(score_fn(p_src, p_dst), 4)
            out["pred_src_category"] = LC_CLASS_NAMES[src_id]
            out["pred_dst_category"] = LC_CLASS_NAMES[dst_id]
            out["has_prediction"] = True

        merged.append(out)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MERGED_FIELDS)
        writer.writeheader()
        writer.writerows(merged)

    scored = sum(1 for r in merged if r["has_prediction"])
    missing = len(merged) - scored
    print(
        f"Wrote {len(merged)} rows to {output} (method={method}); "
        f"{scored} with predictions, {missing} missing (no src and/or dst raster)"
    )
    return merged


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Derive WorldCover change scores at eval points into a standardized CSV."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Evaluation CSV used to create the prediction dataset.",
    )
    parser.add_argument(
        "--ds-path",
        required=True,
        help="Prediction dataset path (with materialized predictions).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output method CSV path.",
    )
    parser.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        choices=sorted(METHODS),
        help=f"Change-score method. Default: {DEFAULT_METHOD}.",
    )
    args = parser.parse_args()

    predict_change(
        csv_path=args.csv,
        ds_path=args.ds_path,
        output=args.output,
        method=args.method,
    )


if __name__ == "__main__":
    main()
