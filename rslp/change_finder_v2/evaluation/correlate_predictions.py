"""Correlate LCC model predictions with the evaluation CSV ground truth.

Reads the ``output_change`` raster of each prediction window (created by
create_prediction_dataset_from_csv.py, group ``predict``, name ``eval_{i:06d}``),
samples the prediction at the annotated point's pixel, and joins it with the CSV.

Writes a merged per-point CSV and prints summary metrics:
- binary change accuracy / precision / recall (changed vs no-change)
- src/dst category accuracy over points where both GT and prediction are "changed"
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

PR_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

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
    "change_prob",
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


def _predict_at_point(arr: np.ndarray, col: int, row: int) -> dict[str, Any]:
    """Extract the prediction at a single pixel from the output_change array."""
    _, h, w = arr.shape
    col = min(max(col, 0), w - 1)
    row = min(max(row, 0), h - 1)

    binary = arr[0:3, row, col]
    predicted_changed = bool(int(binary.argmax()) == BINARY_CHANGE_BAND)
    change_prob = float(arr[BINARY_CHANGE_BAND, row, col]) / 255.0

    # Argmax over classes 1..12 (skip nodata at index 0), then +1.
    src_probs = arr[SRC_BAND_OFFSET : SRC_BAND_OFFSET + NUM_LC_CLASSES, row, col]
    dst_probs = arr[DST_BAND_OFFSET : DST_BAND_OFFSET + NUM_LC_CLASSES, row, col]
    src_id = int(src_probs[1:].argmax()) + 1
    dst_id = int(dst_probs[1:].argmax()) + 1

    return {
        "predicted_changed": predicted_changed,
        "change_prob": round(change_prob, 4),
        "pred_src_category": LC_CLASS_NAMES[src_id],
        "pred_dst_category": LC_CLASS_NAMES[dst_id],
    }


def correlate(csv_path: Path, ds_path: str, output: Path) -> list[dict[str, Any]]:
    """Join predictions with the CSV and return merged rows."""
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
            "change_prob": "",
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
            col, prow = _point_pixel(lon, lat, projection, bounds)
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

    _print_metrics(merged, output)
    return merged


def _print_metrics(merged: list[dict[str, Any]], output: Path) -> None:
    """Print binary and category metrics for rows that have a prediction."""
    scored = [r for r in merged if r["has_prediction"]]
    missing = len(merged) - len(scored)

    print(f"\nWrote {len(merged)} merged rows to {output}")
    print(f"  {len(scored)} with predictions, {missing} missing (no output raster)")

    if not scored:
        print("No predictions to score.")
        return

    # Binary confusion (positive class = changed).
    tp = sum(1 for r in scored if r["has_changed"] and r["predicted_changed"])
    fp = sum(1 for r in scored if not r["has_changed"] and r["predicted_changed"])
    fn = sum(1 for r in scored if r["has_changed"] and not r["predicted_changed"])
    tn = sum(1 for r in scored if not r["has_changed"] and not r["predicted_changed"])

    accuracy = (tp + tn) / len(scored)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    print("\nBinary change (positive = changed):")
    print(f"  accuracy : {accuracy:.3f}  ({tp + tn}/{len(scored)})")
    print(f"  precision: {precision:.3f}")
    print(f"  recall   : {recall:.3f}")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")

    _print_pr_curve(scored)

    # Category accuracy where both GT and prediction are "changed".
    both_changed = [r for r in scored if r["has_changed"] and r["predicted_changed"]]
    if both_changed:
        src_correct = sum(
            1 for r in both_changed if r["src_category"] == r["pred_src_category"]
        )
        dst_correct = sum(
            1 for r in both_changed if r["dst_category"] == r["pred_dst_category"]
        )
        n = len(both_changed)
        print(f"\nCategory accuracy (over {n} points changed in both GT & pred):")
        print(f"  src: {src_correct / n:.3f}  ({src_correct}/{n})")
        print(f"  dst: {dst_correct / n:.3f}  ({dst_correct}/{n})")
    else:
        print("\nNo points where both GT and prediction are 'changed'.")


def _print_pr_curve(
    scored: list[dict[str, Any]], thresholds: list[float] = PR_THRESHOLDS
) -> None:
    """Print a precision/recall/F1 table for the binary change task.

    A point is predicted "changed" when its change_prob >= threshold, so this
    re-thresholds the raw change probability rather than using the argmax-based
    predicted_changed.
    """
    print("\nBinary change PR curve (positive = changed):")
    print("  thresh  precision  recall      F1   TP  FP  FN  TN")
    for threshold in thresholds:
        tp = sum(
            1 for r in scored if r["has_changed"] and r["change_prob"] >= threshold
        )
        fp = sum(
            1 for r in scored if not r["has_changed"] and r["change_prob"] >= threshold
        )
        fn = sum(1 for r in scored if r["has_changed"] and r["change_prob"] < threshold)
        tn = sum(
            1 for r in scored if not r["has_changed"] and r["change_prob"] < threshold
        )

        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else float("nan")
        )

        print(
            f"  {threshold:.2f}      {precision:.3f}     {recall:.3f}   {f1:.3f}  "
            f"{tp:>3} {fp:>3} {fn:>3} {tn:>3}"
        )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Correlate LCC predictions with the evaluation CSV ground truth."
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
        help="Output merged CSV path.",
    )
    args = parser.parse_args()

    correlate(csv_path=args.csv, ds_path=args.ds_path, output=args.output)


if __name__ == "__main__":
    main()
