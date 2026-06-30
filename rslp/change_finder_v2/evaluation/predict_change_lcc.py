"""Produce the LCC model's standardized change CSV from prediction rasters.

Reads the ``output_change`` raster of each prediction window (created by
create_prediction_dataset_from_csv.py, group ``predict``, name ``eval_{i:06d}``),
samples the prediction at the annotated point's pixel, and writes one standardized
CSV row per point. The schema matches ``worldcover/predict_change.py`` so a single
metric script (``metrics.py``) can consume either model's output.

The LCC ``change_score`` is the binary change probability the model outputs:
``binary_change / 255`` (in [0, 1], higher = more change).

    python -m rslp.change_finder_v2.evaluation.predict_change_lcc \
        --csv eval.csv --ds-path "$EVAL_DS" --output eval_lcc.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import shapely
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from upath import UPath

from rslp.change_finder_v2.lcc_model.postprocess import (
    BINARY_CHANGE_BAND,
    DST_BAND_OFFSET,
    LC_CLASS_NAMES,
    NUM_LC_CLASSES,
    SRC_BAND_OFFSET,
    _get_geotiff_path,
)

PREDICTION_GROUP = "predict"

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


def _point_pixel(
    lon: float, lat: float, projection: Projection, bounds: list[int]
) -> tuple[int, int]:
    """Convert lon/lat to (col, row) pixel coords within the window bounds."""
    st = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    projected = st.to_projection(projection)
    col = math.floor(projected.shp.x) - bounds[0]
    row = math.floor(projected.shp.y) - bounds[1]
    return col, row


def _window_center_lonlat(
    projection: Projection, bounds: list[int]
) -> tuple[float, float] | None:
    """Project the window's center pixel back to lon/lat (for diagnostics)."""
    cx = (bounds[0] + bounds[2]) / 2
    cy = (bounds[1] + bounds[3]) / 2
    try:
        pt = (
            STGeometry(projection, shapely.Point(cx, cy), None)
            .to_projection(WGS84_PROJECTION)
            .shp
        )
        return float(pt.x), float(pt.y)
    except Exception:
        return None


def _predict_at_point(arr: np.ndarray, col: int, row: int) -> dict[str, Any]:
    """Extract the prediction at a single pixel from the output_change array."""
    _, h, w = arr.shape
    col = min(max(col, 0), w - 1)
    row = min(max(row, 0), h - 1)

    binary = arr[0:3, row, col]
    predicted_changed = bool(int(binary.argmax()) == BINARY_CHANGE_BAND)
    change_score = float(arr[BINARY_CHANGE_BAND, row, col]) / 255.0

    # Argmax over classes 1..12 (skip nodata at index 0), then +1.
    src_probs = arr[SRC_BAND_OFFSET : SRC_BAND_OFFSET + NUM_LC_CLASSES, row, col]
    dst_probs = arr[DST_BAND_OFFSET : DST_BAND_OFFSET + NUM_LC_CLASSES, row, col]
    src_id = int(src_probs[1:].argmax()) + 1
    dst_id = int(dst_probs[1:].argmax()) + 1

    return {
        "predicted_changed": predicted_changed,
        "change_score": round(change_score, 4),
        "pred_src_category": LC_CLASS_NAMES[src_id],
        "pred_dst_category": LC_CLASS_NAMES[dst_id],
    }


def predict_change(csv_path: Path, ds_path: str, output: Path) -> list[dict[str, Any]]:
    """Join LCC predictions with the CSV and write the standardized CSV."""
    ds_upath = UPath(ds_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    merged: list[dict[str, Any]] = []
    projection_errors = 0
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

        window_dir = ds_upath / "windows" / PREDICTION_GROUP / f"eval_{idx:06d}"
        tif_path = _get_geotiff_path(window_dir)
        metadata_path = window_dir / "metadata.json"
        if tif_path is not None and metadata_path.exists():
            with metadata_path.open() as f:
                metadata = json.load(f)
            projection = Projection.deserialize(metadata["projection"])
            bounds = metadata["bounds"]
            try:
                col, prow = _point_pixel(lon, lat, projection, bounds)
            except Exception as e:
                center = _window_center_lonlat(projection, bounds)
                print(
                    f"[predict_change_lcc] row {idx}: failed to project point "
                    f"(lon={lon}, lat={lat}) into window CRS {projection.crs}; "
                    f"window center lon/lat={center}; skipping. Error: {e}"
                )
                projection_errors += 1
                merged.append(out)
                continue
            with tif_path.open("rb") as f:
                with rasterio.open(f) as src:
                    arr = src.read()
            out.update(_predict_at_point(arr, col, prow))
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
        f"Wrote {len(merged)} rows to {output}; "
        f"{scored} with predictions, {missing} missing "
        f"(no output_change raster or projection error); "
        f"{projection_errors} projection errors"
    )
    return merged


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Produce the LCC standardized change CSV from prediction rasters."
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
        help="Output standardized CSV path.",
    )
    args = parser.parse_args()

    predict_change(csv_path=args.csv, ds_path=args.ds_path, output=args.output)


if __name__ == "__main__":
    main()
