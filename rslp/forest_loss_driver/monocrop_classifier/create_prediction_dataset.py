"""Create the monocrop prediction dataset from forest loss event polygons."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any

import shapely
import shapely.geometry
from rslearn.dataset import Dataset, Window
from upath import UPath

from .create_dataset import (
    MAX_POST_MONTHS,
    PERIOD_DAYS,
    get_window_geometry,
    parse_datetime,
)

PREDICT_GROUP = "predict"
SOURCE_PROPERTY_KEYS = (
    "tif_fname",
    "center_pixel",
    "oe_start_time",
    "oe_end_time",
    "country",
    "new_label",
    "probs",
    "area_ha",
)


def parse_feature_polygon(geometry: dict[str, Any] | None) -> shapely.Geometry | None:
    """Parse a valid Polygon or MultiPolygon from GeoJSON, repairing when possible."""
    if not geometry:
        return None
    try:
        shape = shapely.geometry.shape(geometry)
    except (shapely.GEOSException, ValueError, TypeError, AttributeError, KeyError):
        return None
    if not shape.is_valid:
        try:
            shape = shapely.make_valid(shape)
        except shapely.GEOSException:
            return None
    if shape.is_empty or shape.geom_type not in {"Polygon", "MultiPolygon"}:
        return None
    return shape


def feature_window_name(properties: dict[str, Any], geometry: shapely.Geometry) -> str:
    """Return a deterministic window name for one forest loss event feature."""
    tif_fname = properties.get("tif_fname")
    center_pixel = properties.get("center_pixel")
    if tif_fname and center_pixel is not None:
        identity = json.dumps(
            [tif_fname, center_pixel, properties["oe_start_time"]],
            sort_keys=True,
        )
    else:
        geometry_digest = hashlib.sha256(shapely.normalize(geometry).wkb).hexdigest()
        identity = json.dumps([geometry_digest, properties["oe_start_time"]])
    return "feat_" + hashlib.sha256(identity.encode()).hexdigest()[:16]


def create_prediction_dataset(
    *,
    ds_path: str,
    features: list[dict[str, Any]],
    group: str = PREDICT_GROUP,
) -> dict[str, Any]:
    """Create one prediction window per forest loss event polygon."""
    dataset = Dataset(UPath(ds_path))
    existing_names = {window.name for window in dataset.load_windows(groups=[group])}
    outcome_counts: Counter[str] = Counter()

    for feature in features:
        properties = feature.get("properties") or {}
        geometry = parse_feature_polygon(feature.get("geometry"))
        if geometry is None:
            outcome_counts["invalid_geometry"] += 1
            continue
        if not properties.get("oe_start_time"):
            outcome_counts["missing_event_time"] += 1
            continue
        event_time = parse_datetime(properties["oe_start_time"])

        name = feature_window_name(properties, geometry)
        if name in existing_names:
            outcome_counts["existing"] += 1
            continue

        projection, bounds, _ = get_window_geometry(geometry)
        # Prediction omits PostLossMonthSampler, so the window covers only the 12
        # post-loss periods: the un-sampled stack is exactly the 12-month elapsed
        # view from training, which uses zero pre-loss frames.
        time_range = (
            event_time,
            event_time + timedelta(days=PERIOD_DAYS * MAX_POST_MONTHS),
        )
        window = Window(
            storage=dataset.storage,
            group=group,
            name=name,
            projection=projection,
            bounds=bounds,
            time_range=time_range,
            options={
                "event_time": event_time.isoformat(),
                **{
                    key: properties[key]
                    for key in SOURCE_PROPERTY_KEYS
                    if key in properties
                },
            },
            data_factory=dataset.window_data_storage_factory,
        )
        window.save()
        existing_names.add(name)
        outcome_counts["created"] += 1

    return {"outcomes": dict(sorted(outcome_counts.items()))}


def main() -> None:
    """Create prediction windows from a forest loss event GeoJSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", required=True, type=Path)
    parser.add_argument("--ds-path", required=True)
    parser.add_argument(
        "--group",
        default=PREDICT_GROUP,
        help="Window group; the model predict_config expects 'predict'.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("data/forest_loss_driver/monocrop_classifier/config.json"),
        help="Dataset config copied to DS_PATH/config.json when it does not exist.",
    )
    args = parser.parse_args()

    ds_root = UPath(args.ds_path)
    ds_root.mkdir(parents=True, exist_ok=True)
    dst_config = ds_root / "config.json"
    if not dst_config.exists():
        with args.config_path.open("rb") as src, dst_config.open("wb") as dst:
            dst.write(src.read())

    with args.geojson.open() as f:
        collection = json.load(f)
    result = create_prediction_dataset(
        ds_path=args.ds_path,
        features=collection["features"],
        group=args.group,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
