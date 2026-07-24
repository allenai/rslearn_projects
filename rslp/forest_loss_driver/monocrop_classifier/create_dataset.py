"""Create the monocrop segmentation dataset from confirmed Studio annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import affine
import numpy as np
import shapely
import shapely.affinity
from rasterio.features import rasterize
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from .studio import DEFAULT_PROJECT_IDS, StudioClient

WINDOW_SIZE = 128
WINDOW_RESOLUTION = 10
PRE_EVENT_DAYS = 330
POST_EVENT_DAYS = 360
PERIOD_DAYS = 30
MAX_POST_MONTHS = POST_EVENT_DAYS // PERIOD_DAYS
DEFAULT_IMAGERY_CUTOFF = datetime.fromisoformat("2026-07-20T00:00:00+00:00")
LABEL_LAYER = "label"
LABEL_BAND = "label"

CLASS_NAMES = (
    "nodata",
    "mennonites_nonsoybean",
    "mennonites_soybean",
    "oil_palm",
    "other_agriculture",
    "pastures",
    "rice",
    "soybean",
)
CLASS_TO_ID = {name: class_id for class_id, name in enumerate(CLASS_NAMES)}
ACCEPTED_CONFIDENCE = frozenset({"high", "medium", "low"})
REJECTED_STATUSES = frozenset({"rejected"})


def metadata_label(annotation: dict[str, Any], field_name: str) -> str | None:
    """Return one labelset value by metadata field name."""
    for value in annotation.get("metadata_values") or []:
        if value.get("name") != field_name:
            continue
        label = value.get("label_name")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return None


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 timestamp returned by Studio."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_polygon(wkt: str | None) -> shapely.Geometry | None:
    """Parse a valid Polygon or MultiPolygon, repairing when possible."""
    if not wkt:
        return None
    try:
        geometry = shapely.from_wkt(wkt)
    except (shapely.GEOSException, TypeError):
        return None
    if not geometry.is_valid:
        try:
            geometry = shapely.make_valid(geometry)
        except shapely.GEOSException:
            return None
    if geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        return None
    return geometry


def source_identity(task: dict[str, Any]) -> str:
    """Return a stable event identity for deterministic split grouping."""
    task_geometry = parse_polygon(task.get("geom_wkt"))
    event_time = task.get("start_time")
    if task_geometry is None or event_time is None:
        return f"task:{task['id']}"
    geometry_digest = hashlib.sha256(shapely.normalize(task_geometry).wkb).hexdigest()
    return f"event-geometry:{event_time}:{geometry_digest}"


def assign_split(identity: str) -> str:
    """Assign an approximately 80/20 split from a stable source identity."""
    value = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
    return "val" if value % 5 == 0 else "train"


@dataclass(frozen=True)
class SelectedAnnotation:
    """A Studio annotation accepted for the monocrop dataset."""

    project_id: str
    project_name: str
    task_id: str
    task_name: str
    annotation_id: str
    annotation_status: str
    confidence: str
    class_name: str
    class_id: int
    event_time: datetime
    geometry: shapely.Geometry
    source_identity: str


def select_annotation(
    project: dict[str, Any],
    task_by_id: dict[str, dict[str, Any]],
    annotation: dict[str, Any],
) -> tuple[SelectedAnnotation | None, str]:
    """Apply the confirmed class, status, confidence, date, and geometry policy."""
    status = annotation.get("status")
    if status in REJECTED_STATUSES:
        return None, "rejected_status"

    class_name = metadata_label(annotation, "monoculture_tag")
    if class_name is None:
        return None, "missing_target_label"
    if class_name not in CLASS_TO_ID:
        return None, "unknown_target_label"

    confidence = metadata_label(annotation, "confidence")
    if confidence is None:
        return None, "missing_confidence"
    if confidence not in ACCEPTED_CONFIDENCE:
        return None, "unknown_confidence"

    task_id = annotation.get("task_id")
    task = task_by_id.get(task_id) if task_id is not None else None
    if task is None:
        return None, "missing_task"
    if not task.get("start_time"):
        return None, "missing_event_time"

    geometry = parse_polygon(annotation.get("geom_wkt"))
    if geometry is None:
        return None, "invalid_annotation_geometry"

    return (
        SelectedAnnotation(
            project_id=project["id"],
            project_name=project["name"],
            task_id=task["id"],
            task_name=task["name"],
            annotation_id=annotation["id"],
            annotation_status=status or "<missing>",
            confidence=confidence,
            class_name=class_name,
            class_id=CLASS_TO_ID[class_name],
            event_time=parse_datetime(task["start_time"]),
            geometry=geometry,
            source_identity=source_identity(task),
        ),
        "selected",
    )


def project_group(project_name: str) -> str:
    """Convert a Studio project name to a stable window group."""
    return project_name.removeprefix("Monocrop - ").strip().lower().replace(" ", "_")


def get_window_geometry(
    geometry: shapely.Geometry,
) -> tuple[Projection, tuple[int, int, int, int], shapely.Geometry]:
    """Project the WGS84 polygon and build a centered 128x128 pixel window."""
    center = geometry.representative_point()
    projection = get_utm_ups_projection(
        center.x,
        center.y,
        WINDOW_RESOLUTION,
        -WINDOW_RESOLUTION,
    )
    projected = (
        STGeometry(
            WGS84_PROJECTION,
            geometry,
            time_range=None,
        )
        .to_projection(projection)
        .shp
    )
    projected_center = projected.representative_point()
    center_col = math.floor(projected_center.x)
    center_row = math.floor(projected_center.y)
    half = WINDOW_SIZE // 2
    bounds = (
        center_col - half,
        center_row - half,
        center_col + half,
        center_row + half,
    )
    return projection, bounds, projected


def rasterize_label(
    geometry: shapely.Geometry,
    bounds: tuple[int, int, int, int],
    class_id: int,
) -> np.ndarray:
    """Rasterize one projected annotation polygon into window pixel coordinates."""
    clip = shapely.box(*bounds)
    clipped = geometry.intersection(clip)
    if clipped.is_empty:
        raise ValueError("annotation geometry does not overlap its centered window")
    local = shapely.affinity.translate(
        clipped,
        xoff=-bounds[0],
        yoff=-bounds[1],
    )
    return rasterize(
        [(local, class_id)],
        out_shape=(WINDOW_SIZE, WINDOW_SIZE),
        transform=affine.Affine.identity(),
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )


def write_label(window: Window, dataset: Dataset, label: np.ndarray) -> None:
    """Write a pre-materialized label raster and mark it complete."""
    band_set = dataset.layers[LABEL_LAYER].band_sets[0]
    with window.data.open_layer_writer(LABEL_LAYER) as writer:
        writer.write_raster(
            band_set.bands,
            band_set.instantiate_raster_format(),
            window.projection,
            window.bounds,
            RasterArray(chw_array=label[np.newaxis, :, :]),
        )
    window.mark_layer_completed(LABEL_LAYER)


def create_window(
    record: SelectedAnnotation,
    dataset: Dataset,
    max_post_months: int = MAX_POST_MONTHS,
) -> tuple[Window, str]:
    """Create one window and label, or verify an existing matching window."""
    group = project_group(record.project_name)
    name = record.annotation_id
    existing = dataset.load_windows(groups=[group], names=[name])
    if existing:
        window = existing[0]
        expected_options = {
            "annotation_id": record.annotation_id,
            "class_id": record.class_id,
            "source_identity": record.source_identity,
            "confidence": record.confidence,
            "max_post_months": max_post_months,
        }
        for key, expected in expected_options.items():
            if window.options.get(key) != expected:
                raise ValueError(f"existing window {group}/{name} has mismatched {key}")
        if not window.is_layer_completed(LABEL_LAYER):
            projected_geometry = (
                STGeometry(
                    WGS84_PROJECTION,
                    record.geometry,
                    time_range=None,
                )
                .to_projection(window.projection)
                .shp
            )
            label = rasterize_label(projected_geometry, window.bounds, record.class_id)
            write_label(window, dataset, label)
            return window, "repaired"
        return window, "existing"

    projection, bounds, projected_geometry = get_window_geometry(record.geometry)
    time_range = (
        record.event_time - timedelta(days=PRE_EVENT_DAYS),
        record.event_time + timedelta(days=PERIOD_DAYS * max_post_months),
    )
    split = assign_split(record.source_identity)
    window = Window(
        storage=dataset.storage,
        group=group,
        name=name,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        options={
            "split": split,
            "project_id": record.project_id,
            "project_name": record.project_name,
            "task_id": record.task_id,
            "task_name": record.task_name,
            "annotation_id": record.annotation_id,
            "annotation_status": record.annotation_status,
            "confidence": record.confidence,
            "class_name": record.class_name,
            "class_id": record.class_id,
            "event_time": record.event_time.isoformat(),
            "source_identity": record.source_identity,
            "max_post_months": max_post_months,
        },
        data_factory=dataset.window_data_storage_factory,
    )
    window.save()
    label = rasterize_label(projected_geometry, bounds, record.class_id)
    write_label(window, dataset, label)
    return window, "created"


def create_dataset(
    *,
    ds_path: str,
    project_data: list[dict[str, Any]],
    imagery_cutoff: datetime = DEFAULT_IMAGERY_CUTOFF,
) -> dict[str, Any]:
    """Select Studio records and create all accepted dataset windows."""
    ds_root = UPath(ds_path)
    dataset = Dataset(ds_root)
    outcome_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    horizon_counts: Counter[int] = Counter()

    for data in project_data:
        project = data["project"]
        task_by_id = {task["id"]: task for task in data["tasks"]}
        for annotation in data["annotations"]:
            selected, reason = select_annotation(project, task_by_id, annotation)
            if selected is None:
                outcome_counts[reason] += 1
                continue

            elapsed_days = (imagery_cutoff - selected.event_time).days
            max_post_months = min(MAX_POST_MONTHS, elapsed_days // PERIOD_DAYS)
            if max_post_months < 1:
                reason = "no_complete_post_event_month"
                outcome_counts[reason] += 1
                continue

            window, outcome = create_window(selected, dataset, max_post_months)
            outcome_counts[outcome] += 1
            class_counts[selected.class_name] += 1
            split_counts[window.options["split"]] += 1
            horizon_counts[max_post_months] += 1

    return {
        "outcomes": dict(sorted(outcome_counts.items())),
        "selected_by_class": dict(sorted(class_counts.items())),
        "selected_by_split": dict(sorted(split_counts.items())),
        "selected_by_max_post_months": {
            str(months): count for months, count in sorted(horizon_counts.items())
        },
    }


def main() -> None:
    """Fetch confirmed Studio annotations and create the rslearn dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds-path", required=True)
    parser.add_argument(
        "--project-id",
        action="append",
        dest="project_ids",
        help="Studio project ID; defaults to the three confirmed monocrop projects.",
    )
    parser.add_argument(
        "--imagery-cutoff",
        type=parse_datetime,
        default=DEFAULT_IMAGERY_CUTOFF,
        help=(
            "Latest available imagery timestamp. Each window includes all complete "
            "30-day post-loss periods available by this timestamp, up to 12."
        ),
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

    client = StudioClient()
    project_ids = args.project_ids or list(DEFAULT_PROJECT_IDS)
    result = create_dataset(
        ds_path=args.ds_path,
        project_data=[
            client.get_project_data(project_id) for project_id in project_ids
        ],
        imagery_cutoff=args.imagery_cutoff,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
