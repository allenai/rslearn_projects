"""Pure conversions between v2 annotation entries and Studio records.

One entry becomes one task, and each of its points one annotation on that task.
The window identity, the exact time_range strings, the sample-level fields and
any unrecognized entry keys are kept in task.attributes, so an entry survives
the round trip unchanged apart from empty point fields, which are dropped.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import shapely
import shapely.wkt
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry

from rslp.olmoearth_lcc.studio.schema import (
    NEGATIVE,
    POINT_TYPE_FIELD,
    POSITIVE,
    POSITIVE_POINT_FIELDS,
    is_labelset_field,
)

# Entry keys with a home of their own on the task; anything else goes in
# attributes["extra"].
WINDOW_KEYS = ("projection", "bounds", "window_name", "group")
SAMPLE_KEYS = ("description", "anchor_date")
KNOWN_ENTRY_KEYS = (
    WINDOW_KEYS + ("time_range",) + SAMPLE_KEYS + ("positive_points", "negative_points")
)
POINT_KEYS = ("lon", "lat")
TASK_INDEX_DIGITS = 5


def task_name(index: int, window_name: str) -> str:
    """Studio task name: the zero-padded JSON index, then the window name."""
    return f"{index:0{TASK_INDEX_DIGITS}d} {window_name}"


def task_index(name: str) -> int:
    """The JSON index encoded in a task name."""
    return int(name.split(" ", 1)[0])


def window_polygon(projection: dict[str, Any], bounds: list[int]) -> shapely.Polygon:
    """The window's four pixel corners reprojected to a WGS84 polygon."""
    pixel_box = shapely.box(*bounds)
    geom = STGeometry(Projection.deserialize(projection), pixel_box, None)
    return geom.to_projection(WGS84_PROJECTION).shp


def normalize_time(value: str) -> str:
    """An ISO datetime Studio accepts; dates and naive times are taken as UTC."""
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.isoformat()


def point_wkt(lon: float, lat: float) -> str:
    """WKT for a point, at full float precision."""
    return f"POINT ({lon!r} {lat!r})"


def entry_to_task_payload(index: int, entry: dict[str, Any]) -> dict[str, Any]:
    """The task write for an entry (without project_id)."""
    attributes: dict[str, Any] = {key: entry[key] for key in WINDOW_KEYS}
    time_range = entry.get("time_range")
    if time_range is not None:
        attributes["time_range"] = time_range
    for key in SAMPLE_KEYS:
        if entry.get(key):
            attributes[key] = entry[key]
    extra = {key: value for key, value in entry.items() if key not in KNOWN_ENTRY_KEYS}
    if extra:
        attributes["extra"] = extra

    payload: dict[str, Any] = {
        "name": task_name(index, entry["window_name"]),
        "geom": window_polygon(entry["projection"], entry["bounds"]).wkt,
        "status": "queued",
        "attributes": attributes,
    }
    if time_range:
        payload["start_time"] = normalize_time(time_range[0])
        payload["end_time"] = normalize_time(time_range[1])
    return payload


def point_to_annotation_payload(
    point: dict[str, Any],
    kind: str,
    field_ids: dict[str, str],
    label_ids: dict[str, dict[str, str]],
    task_id: str,
    time_range: list[str] | None,
) -> dict[str, Any]:
    """The annotation write for one point.

    Args:
        point: the v2 point dict.
        kind: POSITIVE or NEGATIVE. Negative points carry no other fields.
        field_ids: {field name: metadata field id}.
        label_ids: {field name: {label name: label id}}.
        task_id: the entry's task.
        time_range: the entry's time_range, if any.
    """
    values = [
        {
            "metadata_field_id": field_ids[POINT_TYPE_FIELD],
            "value": None,
            "label_id": label_ids[POINT_TYPE_FIELD][kind],
        }
    ]
    if kind == POSITIVE:
        for name in POSITIVE_POINT_FIELDS:
            value = point.get(name)
            if not value:
                continue
            if is_labelset_field(name):
                values.append(
                    {
                        "metadata_field_id": field_ids[name],
                        "value": None,
                        "label_id": label_ids[name][value],
                    }
                )
            else:
                values.append({"metadata_field_id": field_ids[name], "value": value})

    payload: dict[str, Any] = {
        "geom": point_wkt(point["lon"], point["lat"]),
        "task_id": task_id,
        "status": "pending",
        "metadata_values": values,
    }
    if time_range:
        payload["start_time"] = normalize_time(time_range[0])
        payload["end_time"] = normalize_time(time_range[1])
    return payload


def unmapped_point_keys(point: dict[str, Any], kind: str) -> list[str]:
    """Point keys that have no Studio field and would be lost on upload."""
    allowed = POINT_KEYS + (POSITIVE_POINT_FIELDS if kind == POSITIVE else ())
    return [key for key in point if key not in allowed]


def annotation_to_point(
    annotation: dict[str, Any],
    field_names: dict[str, str],
    label_names: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    """(kind, v2 point) for one annotation record."""
    geom = shapely.wkt.loads(annotation["geom_wkt"])
    point: dict[str, Any] = {"lon": geom.x, "lat": geom.y}
    fields: dict[str, str] = {}
    for value in annotation.get("metadata_values") or []:
        name = field_names.get(value["metadata_field_id"])
        if name is None:
            continue
        if is_labelset_field(name):
            label_id = value.get("label_id")
            if label_id:
                fields[name] = label_names.get(label_id) or value["label_name"]
        elif value.get("value"):
            fields[name] = value["value"]

    kind = fields.get(POINT_TYPE_FIELD, POSITIVE)
    if kind == POSITIVE:
        for name in POSITIVE_POINT_FIELDS:
            if name in fields:
                point[name] = fields[name]
    return kind, point


def task_and_annotations_to_entry(
    task: dict[str, Any],
    annotations: list[dict[str, Any]],
    field_names: dict[str, str],
    label_names: dict[str, str],
) -> dict[str, Any]:
    """Rebuild a v2 entry from its task and the task's annotations.

    Points keep their creation order, positives and negatives separately.
    """
    attributes = task.get("attributes") or {}
    entry: dict[str, Any] = {key: attributes[key] for key in WINDOW_KEYS}
    if "time_range" in attributes:
        entry["time_range"] = attributes["time_range"]
    elif task.get("start_time") and task.get("end_time"):
        entry["time_range"] = [task["start_time"], task["end_time"]]
    for key in SAMPLE_KEYS:
        if attributes.get(key):
            entry[key] = attributes[key]

    points: dict[str, list[dict[str, Any]]] = {POSITIVE: [], NEGATIVE: []}
    ordered = sorted(annotations, key=lambda a: (a["creation_time"], a["id"]))
    for annotation in ordered:
        kind, point = annotation_to_point(annotation, field_names, label_names)
        points[kind].append(point)
    entry["positive_points"] = points[POSITIVE]
    entry["negative_points"] = points[NEGATIVE]

    for key, value in (attributes.get("extra") or {}).items():
        entry.setdefault(key, value)
    return entry
