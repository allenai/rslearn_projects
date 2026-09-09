"""Create annotation timestamp helper rslearn windows.

This script creates one 32x32 window per annotation entry, centered on the first
positive point. It validates that each entry has a usable time range for a
five-year temporal crop and writes vector labels with the annotated timestamp strings. Run
``rslearn dataset prepare`` after this script to create Sentinel-2 item groups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import shapely
from rslearn.dataset import Dataset, Window
from rslearn.utils import Feature
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from .constants import (
    DATE_FIELDS,
    LABEL_LAYER,
    MANIFEST_FNAME,
    NUM_CROP_MONTHS,
    TIMESTAMP_HEADS,
    WINDOW_SIZE,
)


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO timestamp and return UTC datetime."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _validate_time_range(entry: dict[str, Any]) -> tuple[datetime, datetime]:
    """Validate and return the entry time range."""
    if "time_range" not in entry or len(entry["time_range"]) != 2:
        raise ValueError(f"{entry.get('window_name')} missing two-element time_range")
    start = _parse_datetime(entry["time_range"][0])
    end = _parse_datetime(entry["time_range"][1])
    if end <= start:
        raise ValueError(
            f"{entry.get('window_name')} time_range end must be after start, "
            f"got {entry['time_range']}"
        )

    min_duration = timedelta(days=NUM_CROP_MONTHS * 30)
    if end - start < min_duration:
        raise ValueError(
            f"{entry.get('window_name')} time_range must span at least "
            f"{NUM_CROP_MONTHS} 30-day periods, got {entry['time_range']}"
        )
    return start, end


def _lonlat_to_pixel(lon: float, lat: float, projection: Projection) -> tuple[int, int]:
    """Convert lon/lat to global pixel coordinates in a projection."""
    st = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), time_range=None)
    projected = st.to_projection(projection)
    return math.floor(projected.shp.x), math.floor(projected.shp.y)


def _safe_name(value: str) -> str:
    """Make a filesystem-friendly window-name component."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "entry"


def _window_name(source_json: str, entry: dict[str, Any], entry_index: int) -> str:
    """Create a stable unique window name."""
    digest = hashlib.sha256(f"{source_json}:{entry_index}".encode()).hexdigest()[:12]
    stem = _safe_name(Path(source_json).stem)
    base = _safe_name(entry.get("window_name", f"entry_{entry_index}"))
    return f"{stem}_{base}_{entry_index}_{digest}"


def _split_for(source_json: str, entry_index: int, role: str) -> str:
    """Assign train/val/inference group."""
    if role == "inference":
        return "inference"
    digest = hashlib.sha256(f"{source_json}:{entry_index}".encode()).hexdigest()
    return "val" if digest[0] in "01" else "train"


def _write_label(
    window: Window,
    projection: Projection,
    center: tuple[int, int],
    labels: dict[str, str],
) -> None:
    """Write the vector timestamp label."""
    props = {f"{head}_date": labels[head] for head in TIMESTAMP_HEADS}
    feature = Feature(
        STGeometry(projection, shapely.Point(center[0], center[1]), None),
        props,
    )
    GeojsonVectorFormat().encode_vector(window.get_layer_dir(LABEL_LAYER), [feature])
    window.mark_layer_completed(LABEL_LAYER)


def _process_record(record: dict[str, Any], ds_path: str) -> dict[str, Any]:
    """Create one rslearn window and return its manifest record."""
    entry = record["entry"]
    time_start = _parse_datetime(record["time_start"])
    time_end = _parse_datetime(record["time_end"])
    projection = Projection.deserialize(entry["projection"])
    point = entry["positive_points"][0]
    center = _lonlat_to_pixel(point["lon"], point["lat"], projection)
    half = WINDOW_SIZE // 2
    bounds = (
        center[0] - half,
        center[1] - half,
        center[0] + half,
        center[1] + half,
    )
    group = record["group"]
    name = record["window_name"]

    dataset = Dataset(UPath(ds_path))
    window = Window(
        storage=dataset.storage,
        group=group,
        name=name,
        projection=projection,
        bounds=bounds,
        time_range=(time_start, time_end),
        options={
            "split": group,
            "source_role": record["source_role"],
        },
    )

    window_root = Window.get_window_root(UPath(ds_path), group, name)
    created = not (window_root / "metadata.json").exists()
    if created:
        window.save()
    if record.get("labels") is not None:
        _write_label(window, projection, center, record["labels"])

    return {
        "source_role": record["source_role"],
        "source_json": record["source_json"],
        "entry_index": record["entry_index"],
        "point_index": 0,
        "group": group,
        "window_name": name,
        "time_start": time_start.isoformat(),
        "time_end": time_end.isoformat(),
        "created": created,
    }


def _records_from_json(path: str, role: str) -> list[dict[str, Any]]:
    """Load input annotations and make processing records."""
    source_json = str(Path(path).resolve())
    with open(path) as f:
        entries = json.load(f)

    records: list[dict[str, Any]] = []
    skipped_no_positive = 0
    skipped_incomplete = 0
    for entry_index, entry in enumerate(entries):
        if not entry.get("positive_points"):
            skipped_no_positive += 1
            continue
        point = entry["positive_points"][0]
        if role == "train" and any(
            not point.get(field) for field in DATE_FIELDS.values()
        ):
            skipped_incomplete += 1
            continue
        time_start, time_end = _validate_time_range(entry)
        labels = None
        if role == "train":
            labels = {
                head: _parse_datetime(point[DATE_FIELDS[head]]).isoformat()
                for head in TIMESTAMP_HEADS
            }
        records.append(
            {
                "source_role": role,
                "source_json": source_json,
                "entry_index": entry_index,
                "entry": entry,
                "time_start": time_start.isoformat(),
                "time_end": time_end.isoformat(),
                "labels": labels,
                "group": _split_for(source_json, entry_index, role),
                "window_name": _window_name(source_json, entry, entry_index),
            }
        )

    print(
        f"{path}: {len(records)} {role} records, "
        f"{skipped_no_positive} no-positive skipped, "
        f"{skipped_incomplete} incomplete skipped"
    )
    return records


def create_windows(
    *,
    train_jsons: list[str],
    infer_jsons: list[str],
    ds_path: str,
) -> None:
    """Create timestamp helper windows and labels."""
    records: list[dict[str, Any]] = []
    for path in train_jsons:
        records.extend(_records_from_json(path, "train"))
    for path in infer_jsons:
        records.extend(_records_from_json(path, "inference"))

    manifests = [_process_record(record=record, ds_path=ds_path) for record in records]
    manifests.sort(key=lambda m: (m["source_json"], m["entry_index"]))
    manifest_path = UPath(ds_path) / MANIFEST_FNAME
    with manifest_path.open("w") as f:
        json.dump({"version": 1, "windows": manifests}, f)
    print(f"Wrote {len(manifests)} manifest records to {manifest_path}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Create annotation timestamp helper rslearn windows."
    )
    parser.add_argument("ds_path", help="Output rslearn dataset path.")
    parser.add_argument("--train-json", action="append", default=[])
    parser.add_argument("--infer-json", action="append", default=[])
    args = parser.parse_args()

    create_windows(
        train_jsons=args.train_json,
        infer_jsons=args.infer_json,
        ds_path=args.ds_path,
    )


if __name__ == "__main__":
    main()
