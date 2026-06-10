"""Correlate ESRI land cover stacks with the evaluation CSV ground truth.

For each evaluation point, finds the ESRI stacked GeoTIFF tile that covers
that lon/lat, reads the land cover class at ``src_year`` and ``dst_year``,
and determines whether a change occurred and what the transition was.

Writes a merged per-point CSV and prints the same metrics as
``correlate_predictions.py``: binary accuracy/precision/recall and
src/dst category accuracy.

Usage::

    python -m rslp.change_finder_v2.evaluation.correlate_esri \
        --csv eval_points_exported.csv \
        --stack-path /weka/.../esri_lc_stacks/ \
        --output eval_esri_merged.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import rasterio
from rasterio.transform import rowcol
from upath import UPath

YEARS = list(range(2017, 2025))

# ESRI class value → LCC annotation category name.
ESRI_TO_LCC: dict[int, str] = {
    1: "water",
    2: "tree",
    4: "wetland (herbaceous)",
    5: "crops",
    7: "urban/built-up",
    8: "bare",
    9: "snow and ice",
    10: "",  # Clouds → treat as nodata
    11: "grassland",  # Rangeland (includes shrub)
}

# LCC categories that map to the same ESRI class (for fairer comparison).
# "shrub" and "grassland" both become ESRI Rangeland (11).
LCC_TO_ESRI_VALUE: dict[str, int] = {
    "water": 1,
    "tree": 2,
    "wetland (herbaceous)": 4,
    "crops": 5,
    "urban/built-up": 7,
    "bare": 8,
    "snow and ice": 9,
    "grassland": 11,
    "shrub": 11,
}

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
    "pred_src_category",
    "pred_dst_category",
    "esri_src_value",
    "esri_dst_value",
    "skip_reason",
]


def _find_tile_for_point(
    lon: float,
    lat: float,
    tile_datasets: dict[str, rasterio.DatasetReader],
) -> tuple[str, rasterio.DatasetReader] | None:
    """Find which open ESRI tile dataset contains the given lon/lat."""
    for tile_id, ds in tile_datasets.items():
        try:
            r, c = rowcol(ds.transform, lon, lat)
        except (ValueError, TypeError):
            continue
        if 0 <= r < ds.height and 0 <= c < ds.width:
            return tile_id, ds
    return None


def correlate_esri(
    csv_path: Path,
    stack_path: str,
    output: Path,
) -> list[dict[str, Any]]:
    """Correlate ESRI stacks with the evaluation CSV."""
    stack_dir = UPath(stack_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    # Open all tile datasets up front (only file handles, not pixel data).
    tile_files = sorted(
        f for f in os.listdir(str(stack_dir)) if f.endswith("_lc_stack.tif")
    )
    tile_datasets: dict[str, rasterio.DatasetReader] = {}
    for fname in tile_files:
        tile_id = fname.replace("_lc_stack.tif", "")
        tile_datasets[tile_id] = rasterio.open(str(stack_dir / fname))

    print(f"Opened {len(tile_datasets)} ESRI tile datasets")

    merged: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        lon = float(row["lon"])
        lat = float(row["lat"])
        src_year = int(row["src_year"])
        dst_year = int(row["dst_year"])
        gt_changed = row["has_changed"].strip().lower() == "true"

        out: dict[str, Any] = {
            "row_index": idx,
            "lon": lon,
            "lat": lat,
            "src_year": src_year,
            "dst_year": dst_year,
            "has_changed": gt_changed,
            "src_category": row.get("src_category", ""),
            "dst_category": row.get("dst_category", ""),
            "has_prediction": False,
            "predicted_changed": "",
            "pred_src_category": "",
            "pred_dst_category": "",
            "esri_src_value": "",
            "esri_dst_value": "",
            "skip_reason": "",
        }

        # Check year range.
        if src_year not in YEARS or dst_year not in YEARS:
            out["skip_reason"] = "year_out_of_range"
            merged.append(out)
            continue

        src_band = YEARS.index(src_year) + 1  # 1-indexed bands
        dst_band = YEARS.index(dst_year) + 1

        # Find the tile.
        result = _find_tile_for_point(lon, lat, tile_datasets)
        if result is None:
            out["skip_reason"] = "no_tile"
            merged.append(out)
            continue

        tile_id, ds = result
        r, c = rowcol(ds.transform, lon, lat)

        src_val = int(
            ds.read(src_band, window=rasterio.windows.Window(c, r, 1, 1))[0, 0]
        )
        dst_val = int(
            ds.read(dst_band, window=rasterio.windows.Window(c, r, 1, 1))[0, 0]
        )

        out["esri_src_value"] = src_val
        out["esri_dst_value"] = dst_val

        # Nodata check (class 0 or class 10=Clouds).
        if src_val == 0 or dst_val == 0 or src_val == 10 or dst_val == 10:
            out["skip_reason"] = "nodata_or_cloud"
            merged.append(out)
            continue

        src_name = ESRI_TO_LCC.get(src_val, "")
        dst_name = ESRI_TO_LCC.get(dst_val, "")

        predicted_changed = src_val != dst_val
        out["has_prediction"] = True
        out["predicted_changed"] = predicted_changed
        out["pred_src_category"] = src_name
        out["pred_dst_category"] = dst_name

        merged.append(out)

    # Close datasets.
    for ds in tile_datasets.values():
        ds.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MERGED_FIELDS)
        writer.writeheader()
        writer.writerows(merged)

    _print_metrics(merged, output)
    return merged


def _normalize_category(cat: str) -> str:
    """Normalize a GT category for comparison with ESRI predictions.

    Maps 'shrub' → 'grassland' since ESRI merges both into Rangeland.
    """
    if cat == "shrub":
        return "grassland"
    return cat


def _print_metrics(merged: list[dict[str, Any]], output: Path) -> None:
    """Print binary and category metrics."""
    scored = [r for r in merged if r["has_prediction"]]
    missing = len(merged) - len(scored)
    skip_reasons: dict[str, int] = {}
    for r in merged:
        reason = r.get("skip_reason", "")
        if reason:
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    print(f"\nWrote {len(merged)} merged rows to {output}")
    print(f"  {len(scored)} with predictions, {missing} missing")
    if skip_reasons:
        for reason, cnt in sorted(skip_reasons.items()):
            print(f"    {reason}: {cnt}")

    if not scored:
        print("No predictions to score.")
        return

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

    both_changed = [r for r in scored if r["has_changed"] and r["predicted_changed"]]
    if both_changed:
        src_correct = sum(
            1
            for r in both_changed
            if _normalize_category(r["src_category"]) == r["pred_src_category"]
        )
        dst_correct = sum(
            1
            for r in both_changed
            if _normalize_category(r["dst_category"]) == r["pred_dst_category"]
        )
        n = len(both_changed)
        print(f"\nCategory accuracy (over {n} points changed in both GT & pred):")
        print(f"  src: {src_correct / n:.3f}  ({src_correct}/{n})")
        print(f"  dst: {dst_correct / n:.3f}  ({dst_correct}/{n})")
    else:
        print("\nNo points where both GT and prediction are 'changed'.")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Correlate ESRI land cover stacks with evaluation CSV ground truth."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Evaluation CSV (from export_annotations_to_csv).",
    )
    parser.add_argument(
        "--stack-path",
        required=True,
        help="Directory containing ESRI stacked GeoTIFFs ({tile}_lc_stack.tif).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output merged CSV path.",
    )
    args = parser.parse_args()

    correlate_esri(
        csv_path=args.csv,
        stack_path=args.stack_path,
        output=args.output,
    )


if __name__ == "__main__":
    main()
