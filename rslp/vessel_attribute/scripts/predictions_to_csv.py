"""Combine info (ground truth) and output (prediction) layers into a CSV for val windows."""

import argparse
import csv
import json
import multiprocessing
from typing import Any

import tqdm
from upath import UPath

import rslp.utils.mp

GT_FIELDS = ["length", "width", "sog", "cog", "type"]
PRED_FIELDS = ["length", "width", "sog", "heading", "type"]

CSV_COLUMNS = (
    ["window_name"]
    + [f"gt_{f}" for f in GT_FIELDS]
    + [f"pred_{f}" for f in PRED_FIELDS]
)


def process_window(window_dir: UPath) -> dict[str, Any] | None:
    """Read info and output layers for one window and return a row dict.

    Args:
        window_dir: path to a window directory (e.g. windows/default/<name>).

    Returns:
        dict with CSV column values, or None if the window should be skipped.
    """
    metadata_fname = window_dir / "metadata.json"
    with metadata_fname.open() as f:
        metadata = json.load(f)

    if metadata.get("options", {}).get("split") != "val":
        return None

    output_fname = window_dir / "layers" / "output" / "data.geojson"
    if not output_fname.exists():
        return None

    info_fname = window_dir / "layers" / "info" / "data.geojson"
    with info_fname.open() as f:
        info_fc = json.load(f)
    with output_fname.open() as f:
        output_fc = json.load(f)

    gt_props = info_fc["features"][0]["properties"] if info_fc["features"] else {}
    pred_props = output_fc["features"][0]["properties"] if output_fc["features"] else {}

    row: dict[str, Any] = {"window_name": window_dir.name}
    for f in GT_FIELDS:
        row[f"gt_{f}"] = gt_props.get(f, "")
    for f in PRED_FIELDS:
        row[f"pred_{f}"] = pred_props.get(f, "")
    return row


if __name__ == "__main__":
    rslp.utils.mp.init_mp()

    parser = argparse.ArgumentParser(
        description="Combine GT and predicted vessel attributes into a CSV",
    )
    parser.add_argument("--ds_path", type=str, required=True, help="Dataset root path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    windows_root = ds_path / "windows"
    window_dirs = [
        child for group_dir in windows_root.iterdir() for child in group_dir.iterdir()
    ]
    print(f"Found {len(window_dirs)} windows")

    p = multiprocessing.Pool(32)
    rows: list[dict[str, Any]] = []
    for row in tqdm.tqdm(
        p.imap_unordered(process_window, window_dirs), total=len(window_dirs)
    ):
        if row is not None:
            rows.append(row)
    p.close()

    rows.sort(key=lambda r: r["window_name"])

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} val rows to {args.output}")
