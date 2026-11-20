"""Create the forest loss driver classification dataset from OlmoEarth Studio labels.

The dataset is built purely from the labels in Studio: for each labeled task in the
registered Studio projects, we create a 128x128 window at 10 m/pixel in the UTM zone
appropriate for the location, and write a `label` vector layer containing the forest
loss polygon and the label. The Sentinel-2 images are then obtained by the standard
`rslearn dataset prepare` / `materialize` steps.

Each label is resolved to the set of allowed leaves of ACA's label hierarchy (see
rslp.forest_loss_driver.hierarchy), written as `leaf_labels`. Where the label also
corresponds to one of the legacy flat categories, that is written as `new_label` so
that the configs for the flat model still work.

Re-running the script is incremental: existing windows are skipped unless the Studio
annotation has been updated since the window was created (or the label format has
changed), in which case only the label layer is rewritten.

See data/forest_loss_driver/20261120_hierarchy/README.md for usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import shapely
import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

from rslp.forest_loss_driver import hierarchy
from rslp.forest_loss_driver.monocrop_classifier.create_dataset import (
    assign_split as assign_monocrop_split,
)
from rslp.forest_loss_driver.monocrop_classifier.create_dataset import (
    source_identity as monocrop_source_identity,
)
from rslp.forest_loss_driver.monocrop_classifier.studio import StudioClient

WINDOW_SIZE = 128
WINDOW_RESOLUTION = 10
LABEL_LAYER = "label"
REJECTED_STATUSES = frozenset({"rejected"})
VAL_HASH_PREFIXES = frozenset({"0", "1", "2", "3"})
# Bumped when the label feature format changes, so that existing windows get their
# label rewritten even if the Studio annotation has not changed.
LABEL_VERSION = 2
LEVEL_FIELDS = ("level1", "level2", "level3", "level4")

# The legacy flat categories.
CLASSES = hierarchy.LEGACY_CLASSES

# Mappings from raw Studio label to the legacy flat category (written as new_label).
# These are separate from the leaf mappings in rslp.forest_loss_driver.hierarchy, and
# are kept as they were before the hierarchy was introduced.
FLAT_LABEL_MAP: dict[str, str] = {category: category for category in CLASSES}

# Original Peru label hierarchy, with the sub-categories consolidated as in
# rslp.forest_loss_driver.train.CATEGORY_MAPPING.
PERU_LEGACY_LABEL_MAP: dict[str, str] = FLAT_LABEL_MAP | {
    "agriculture-generic": "agriculture",
    "agriculture-small": "agriculture",
    "agriculture-mennonite": "agriculture",
    "agriculture-rice": "agriculture",
    "coca": "agriculture",
    "flood": "river",
}

# Label hierarchy proposed by ACA for the Brazil/Colombia/Peru annotation projects,
# mapped back to the flat categories for compatibility with the Peru labels.
ACA_LABEL_MAP: dict[str, str] = {
    "Airstrips": "airstrip",
    "Agriculture-Large": "agriculture",
    "Agriculture-Medium": "agriculture",
    "Agriculture-Small": "agriculture",
    "Burned_areas_Fire": "burned",
    "Dry_areas_seasonal": "none",
    "False_positives_not_forest_loss_an_alert_error": "none",
    "Flooded_rivers": "river",
    "Landslides": "landslide",
    "Logging_clear-cut": "logging",
    "Logging_roads": "road",
    "Mining": "mining",
    "Pasture_praderas_ganaderia": "agriculture",
    "Roads": "road",
    "Selective_Logging": "logging",
    "Windthrowsblowdowns_Hurricane_winds": "hurricane",
}

# The monocrop projects only contain events that are agriculture.
MONOCROP_LEGACY_LABEL_MAP: dict[str, str] = {
    label: "agriculture" for label in hierarchy.MONOCROP_LABEL_LEAVES
}

# Groups where the split is assigned by hashing the legacy rslearn window name (from
# the Studio task attributes). This reproduces the split used by the historical
# `combined` dataset so that the validation set stays comparable.
HASH_SPLIT_GROUPS = frozenset(
    {
        "20250428_brazil_phase1",
        "20250428_colombia_phase1",
        "20250428_brazil_phase2",
        "20250428_colombia_phase2",
        "peru3_flagged_in_peru",
        "peru_interesting",
    }
)
# Groups where the split is assigned by hashing the Studio task ID.
TASK_HASH_SPLIT_GROUPS = frozenset({"20260929_validation"})
# Groups from the monocrop projects, where the split matches the one used by the
# monocrop classifier dataset (rslp.forest_loss_driver.monocrop_classifier).
MONOCROP_SPLIT_GROUPS = frozenset(
    {
        "monocrop_peru",
        "monocrop_ecuador",
        "monocrop_bolivia",
        "monocrop_phase2",
        "monocrop_validation_202610",
    }
)
# Groups that are used only for training.
TRAIN_ONLY_GROUPS = frozenset(
    {
        "peru3",
        "nadia2",
        "nadia3",
        "brazil_interesting",
        "20260112_peru",
        "20260821_validatetest",
    }
)


@dataclass(frozen=True)
class ParsedLabel:
    """The label of a Studio annotation.

    Args:
        leaves: the allowed leaves of the label hierarchy.
        raw_label: the raw Studio label (the level labels joined by "/" when the
            annotation uses the level fields).
        new_label: the corresponding legacy flat category, if any.
        levels: the raw level1-level4 labels that are set on the annotation.
    """

    leaves: frozenset[str]
    raw_label: str
    new_label: str | None
    levels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StudioProject:
    """A Studio project to include in the dataset.

    Args:
        project_id: the Studio project ID.
        label_field: the annotation metadata field containing the labeled category.
        leaf_map: mapping from raw Studio label to the allowed leaves. Labels that are
            not in the map are skipped.
        legacy_map: mapping from raw Studio label to the legacy flat category.
        use_levels: whether the annotations have the level1-level4 fields of the new
            hierarchy. When any of level1-level3 is set, the levels take precedence and
            label_field is only used as a fallback.
        required_task_status: if set, only use tasks with this status (e.g. only
            reviewed tasks in projects where annotation is in progress).
        group_map: mapping from the `group` task attribute (the group in the legacy
            rslearn dataset that was imported into Studio) to the dataset group to
            use. Tasks whose group is not in the map are skipped.
        fixed_group: the dataset group for all tasks in this project, for projects
            that were not imported from an rslearn dataset.
        extra_fields: additional annotation metadata fields to copy into the label
            feature properties (e.g. the model prediction being validated).
    """

    project_id: str
    label_field: str
    leaf_map: dict[str, frozenset[str]]
    legacy_map: dict[str, str]
    use_levels: bool = False
    required_task_status: str | None = None
    group_map: dict[str, str] | None = None
    fixed_group: str | None = None
    extra_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the project definition."""
        if (self.group_map is None) == (self.fixed_group is None):
            raise ValueError("exactly one of group_map or fixed_group must be set")

    def get_group(self, task: dict[str, Any]) -> str | None:
        """Get the dataset group for the task, or None if it should be skipped."""
        if self.fixed_group is not None:
            return self.fixed_group
        assert self.group_map is not None
        legacy_group = (task.get("attributes") or {}).get("group")
        if legacy_group is None:
            return None
        return self.group_map.get(legacy_group)

    def parse_label(self, annotation: dict[str, Any]) -> tuple[ParsedLabel | None, str]:
        """Get the label of an annotation.

        Returns:
            a tuple (label, reason) where label is None if the annotation should be
                skipped, and reason is a short string describing the outcome.
        """
        levels: dict[str, str] = {}
        if self.use_levels:
            for name in LEVEL_FIELDS:
                value = metadata_value(annotation, name)
                if value is not None:
                    levels[name] = value

        if any(name in levels for name in LEVEL_FIELDS[:3]):
            level_label = "/".join(
                levels[name] for name in LEVEL_FIELDS[:3] if name in levels
            )
            leaves = hierarchy.parse_levels(
                levels.get("level1"), levels.get("level2"), levels.get("level3")
            )
            if leaves is None:
                return None, f"unmapped_levels:{level_label}"
            legacy = {hierarchy.LEAF_TO_LEGACY.get(leaf) for leaf in leaves}
            new_label = legacy.pop() if len(legacy) == 1 else None
            return ParsedLabel(leaves, level_label, new_label, levels), "selected"

        raw_label = metadata_value(annotation, self.label_field)
        if raw_label is None:
            return None, "missing_label"
        if raw_label not in self.leaf_map:
            return None, f"unmapped_label:{raw_label}"
        return (
            ParsedLabel(
                self.leaf_map[raw_label],
                raw_label,
                self.legacy_map.get(raw_label),
                levels,
            ),
            "selected",
        )


def _monocrop_project(
    project_id: str, group: str, in_progress: bool = False
) -> StudioProject:
    return StudioProject(
        project_id=project_id,
        label_field="monoculture_tag",
        leaf_map=hierarchy.MONOCROP_LABEL_LEAVES,
        legacy_map=MONOCROP_LEGACY_LABEL_MAP,
        use_levels=in_progress,
        required_task_status="reviewed" if in_progress else None,
        fixed_group=group,
    )


PROJECTS: tuple[StudioProject, ...] = (
    # Forest Loss Driver Peru 7: the original Peru labels (plus a few Brazil examples)
    # imported into Studio from the legacy rslearn dataset.
    StudioProject(
        project_id="25dfbe5f-4349-4646-b250-508afb2d42ba",
        label_field="tag_name",
        leaf_map=hierarchy.PERU_LEGACY_LABEL_LEAVES,
        legacy_map=PERU_LEGACY_LABEL_MAP,
        group_map={
            "nadia2": "nadia2",
            "nadia3": "nadia3",
            "peru3": "peru3",
            "peru3_flagged_in_peru": "peru3_flagged_in_peru",
            "brazil": "brazil_interesting",
            "peru": "peru_interesting",
        },
    ),
    # Forest Loss Driver Brazil 7 (phase 1).
    StudioProject(
        project_id="f56e41c6-83ab-4a7f-9b14-443391f9b2ba",
        label_field="tag_name",
        leaf_map=hierarchy.ACA_LABEL_LEAVES,
        legacy_map=ACA_LABEL_MAP,
        group_map={"20250428_brazil_phase1": "20250428_brazil_phase1"},
    ),
    # Forest Loss Driver Colombia 7 (phase 1).
    StudioProject(
        project_id="a493cba0-466f-4604-8359-c437b78f7009",
        label_field="tag_name",
        leaf_map=hierarchy.ACA_LABEL_LEAVES,
        legacy_map=ACA_LABEL_MAP,
        group_map={"20250428_colombia_phase1": "20250428_colombia_phase1"},
    ),
    # Forest Loss Driver Brazil 12 (phase 2).
    StudioProject(
        project_id="f8137f81-15ac-4f94-b0fd-8ce5f62a6f78",
        label_field="tag_name",
        leaf_map=hierarchy.ACA_LABEL_LEAVES,
        legacy_map=ACA_LABEL_MAP,
        group_map={"20250428_brazil_phase2": "20250428_brazil_phase2"},
    ),
    # Forest Loss Driver Colombia 12 (phase 2).
    StudioProject(
        project_id="7732a6c0-cea0-46a5-9498-5d93eed51364",
        label_field="tag_name",
        leaf_map=hierarchy.ACA_LABEL_LEAVES,
        legacy_map=ACA_LABEL_MAP,
        group_map={"20250428_colombia_phase2": "20250428_colombia_phase2"},
    ),
    # Forest Loss Driver Peru 20260112c.
    StudioProject(
        project_id="077f374b-2a72-45f4-9945-0290cc311201",
        label_field="tag_name",
        leaf_map=hierarchy.ACA_LABEL_LEAVES,
        legacy_map=ACA_LABEL_MAP,
        group_map={"20260112_peru": "20260112_peru"},
    ),
    # Validatetest: validation of the deployed model's outputs (Jul 2025 - Jun 2026)
    # across Brazil, Peru, Bolivia, Colombia, and Ecuador. The labeled category is in
    # the "validate" field while "category" is the model prediction.
    StudioProject(
        project_id="1d3727e6-5844-4e89-87f1-c1147417e180",
        label_field="validate",
        leaf_map=hierarchy.FLAT_LABEL_LEAVES,
        legacy_map=FLAT_LABEL_MAP,
        fixed_group="20260821_validatetest",
        extra_fields=("category", "probs", "country", "estrato"),
    ),
    # Forest Loss Driver Validation 2026-09: events (Jan 2025 - Jul 2026) selected with
    # the 20260924 model, annotated with the new level1-level4 hierarchy. The
    # "validate" field is the annotator's flat category, used when no level is set.
    StudioProject(
        project_id="3ba58828-52a0-485a-8a9b-0cd0adfc3d08",
        label_field="validate",
        leaf_map=hierarchy.FLAT_LABEL_LEAVES,
        legacy_map=FLAT_LABEL_MAP,
        use_levels=True,
        required_task_status="reviewed",
        fixed_group="20260929_validation",
        extra_fields=("country",),
    ),
    # Monocrop projects: agriculture events labeled with the crop type.
    _monocrop_project("30b9bbb2-1ac9-4cf6-baa5-1b77ce79881c", "monocrop_peru"),
    _monocrop_project("8188a029-fd50-4670-a6f2-243afc3e1b83", "monocrop_ecuador"),
    _monocrop_project("dd3e9ecb-4060-4a9b-af71-05b4bf8ad747", "monocrop_bolivia"),
    _monocrop_project("ffe01218-87d6-4ebe-85c8-70b5515b951a", "monocrop_phase2"),
    # Monocrop Validation 2026-10 also has the level1-level4 fields.
    _monocrop_project(
        "9241501a-5fa6-49e1-a1aa-dafafe24d70b",
        "monocrop_validation_202610",
        in_progress=True,
    ),
)
PROJECTS_BY_ID = {project.project_id: project for project in PROJECTS}


def metadata_value(annotation: dict[str, Any], field_name: str) -> str | None:
    """Return one annotation metadata value by field name.

    For labelset fields this is the label name, otherwise it is the raw value.
    """
    for value in annotation.get("metadata_values") or []:
        if value.get("name") != field_name:
            continue
        if value.get("data_type") == "labelset" or value.get("label_name"):
            label = value.get("label_name")
        else:
            label = value.get("value")
        if label is None:
            return None
        label = str(label).strip()
        return label if label else None
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


def _hash_split(identity: str) -> str:
    prefix = hashlib.sha256(identity.encode()).hexdigest()[0]
    return "val" if prefix in VAL_HASH_PREFIXES else "train"


def assign_split(group: str, task: dict[str, Any]) -> str:
    """Assign the train/val split for a window.

    Args:
        group: the dataset group of the window.
        task: the Studio task. Depending on the group, the split is derived from the
            name of the window in the legacy rslearn dataset (the `window` task
            attribute), the task ID, or the event geometry and time.

    Returns:
        "train" or "val".
    """
    if group in HASH_SPLIT_GROUPS:
        legacy_window = (task.get("attributes") or {}).get("window")
        if legacy_window is None:
            raise ValueError(f"group {group} requires a legacy window name for split")
        return _hash_split(legacy_window)
    if group in TASK_HASH_SPLIT_GROUPS:
        return _hash_split(task["id"])
    if group in MONOCROP_SPLIT_GROUPS:
        return assign_monocrop_split(monocrop_source_identity(task))
    if group in TRAIN_ONLY_GROUPS:
        return "train"
    raise ValueError(f"no split policy for group {group}")


@dataclass(frozen=True)
class SelectedTask:
    """A Studio task with a usable label for the dataset."""

    project_id: str
    task_id: str
    task_name: str
    annotation_id: str
    annotation_status: str
    annotation_updated_time: str
    group: str
    split: str
    leaves: frozenset[str]
    new_label: str | None
    raw_label: str
    confidence: str | None
    time_range: tuple[datetime, datetime]
    # Forest loss polygon in WGS84.
    geometry: shapely.Geometry
    legacy_group: str | None
    legacy_window: str | None
    levels: dict[str, str] = field(default_factory=dict)
    extra_properties: dict[str, str] = field(default_factory=dict)


def pick_annotation(
    annotations: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the most recently updated annotation among those for one task."""
    if not annotations:
        return None
    return max(
        annotations,
        key=lambda annotation: (
            annotation.get("updated_time") or annotation.get("creation_time") or ""
        ),
    )


def select_task(
    project: StudioProject,
    task: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> tuple[SelectedTask | None, str]:
    """Determine whether a task should become a window, and collect its metadata.

    Args:
        project: the Studio project definition.
        task: the Studio task.
        annotations: the annotations belonging to this task.

    Returns:
        a tuple (selected, reason) where selected is None if the task is skipped, and
            reason is a short string describing the outcome.
    """
    group = project.get_group(task)
    if group is None:
        return None, "unknown_group"

    if (
        project.required_task_status is not None
        and task.get("status") != project.required_task_status
    ):
        return None, f"task_status:{task.get('status')}"

    annotation = pick_annotation(annotations)
    if annotation is None:
        return None, "missing_annotation"
    status = annotation.get("status")
    if status in REJECTED_STATUSES:
        return None, "rejected_status"

    label, reason = project.parse_label(annotation)
    if label is None:
        return None, reason

    if not task.get("start_time"):
        return None, "missing_event_time"
    start_time = parse_datetime(task["start_time"])
    end_time = parse_datetime(task["end_time"]) if task.get("end_time") else start_time

    # Prefer the annotation polygon; fall back to the task polygon (window bounds) if
    # the annotation geometry is a point or otherwise unusable.
    geometry = parse_polygon(annotation.get("geom_wkt"))
    if geometry is None:
        geometry = parse_polygon(task.get("geom_wkt"))
    if geometry is None:
        return None, "invalid_geometry"

    attributes = task.get("attributes") or {}
    legacy_group = attributes.get("group")
    legacy_window = attributes.get("window")

    extra_properties: dict[str, str] = {}
    for field_name in project.extra_fields:
        value = metadata_value(annotation, field_name)
        if value is not None:
            extra_properties[field_name] = value

    return (
        SelectedTask(
            project_id=project.project_id,
            task_id=task["id"],
            task_name=task.get("name") or task["id"],
            annotation_id=annotation["id"],
            annotation_status=status or "<missing>",
            annotation_updated_time=annotation.get("updated_time")
            or annotation.get("creation_time")
            or "",
            group=group,
            split=assign_split(group, task),
            leaves=label.leaves,
            new_label=label.new_label,
            raw_label=label.raw_label,
            confidence=metadata_value(annotation, "Confidence")
            or metadata_value(annotation, "confidence"),
            time_range=(start_time, end_time),
            geometry=geometry,
            legacy_group=legacy_group,
            legacy_window=legacy_window,
            levels=label.levels,
            extra_properties=extra_properties,
        ),
        "selected",
    )


def select_tasks(
    project: StudioProject,
    tasks: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> tuple[list[SelectedTask], Counter[str]]:
    """Select all usable tasks in a project.

    Returns:
        a tuple (selected tasks, counter of skip reasons).
    """
    annotations_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        annotations_by_task[annotation["task_id"]].append(annotation)

    selected: list[SelectedTask] = []
    reasons: Counter[str] = Counter()
    for task in tasks:
        record, reason = select_task(project, task, annotations_by_task[task["id"]])
        if record is None:
            reasons[reason] += 1
            continue
        selected.append(record)
    return selected, reasons


def get_window_geometry(
    geometry: shapely.Geometry,
) -> tuple[Projection, tuple[int, int, int, int], shapely.Geometry]:
    """Compute the UTM projection, window bounds, and projected polygon.

    Args:
        geometry: the forest loss polygon in WGS84.

    Returns:
        a tuple (projection, bounds, projected geometry) where the projection is the
            10 m/pixel UTM zone containing the polygon centroid and the bounds are a
            128x128 window centered on the centroid.
    """
    center = geometry.centroid
    projection = get_utm_ups_projection(
        center.x, center.y, WINDOW_RESOLUTION, -WINDOW_RESOLUTION
    )
    projected = (
        STGeometry(WGS84_PROJECTION, geometry, time_range=None)
        .to_projection(projection)
        .shp
    )
    projected_center = (
        STGeometry(WGS84_PROJECTION, center, time_range=None)
        .to_projection(projection)
        .shp
    )
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


def get_label_properties(record: SelectedTask) -> dict[str, Any]:
    """Get the properties to store in the label feature."""
    properties: dict[str, Any] = {
        "leaf_labels": [leaf for leaf in hierarchy.LEAVES if leaf in record.leaves],
        "hierarchy_nodes": hierarchy.covering_nodes(record.leaves),
        "raw_label": record.raw_label,
        "confidence": record.confidence,
        "studio_project_id": record.project_id,
        "studio_task_id": record.task_id,
        "studio_task_name": record.task_name,
        "studio_annotation_id": record.annotation_id,
        "annotation_status": record.annotation_status,
        "annotation_updated_time": record.annotation_updated_time,
        "legacy_group": record.legacy_group,
        "legacy_window": record.legacy_window,
    }
    # Omitted rather than null so that configs for the flat model skip the window.
    if record.new_label is not None:
        properties["new_label"] = record.new_label
    properties.update(record.levels)
    properties.update(record.extra_properties)
    return properties


def get_window_options(record: SelectedTask) -> dict[str, Any]:
    """Get the options to store in the window metadata."""
    return {
        "split": record.split,
        "studio_project_id": record.project_id,
        "studio_task_id": record.task_id,
        "studio_annotation_id": record.annotation_id,
        "annotation_updated_time": record.annotation_updated_time,
        "label_version": LABEL_VERSION,
        "legacy_group": record.legacy_group,
        "legacy_window": record.legacy_window,
        "event_time": record.time_range[0].isoformat(),
    }


def write_label(window: Window, dataset: Dataset, record: SelectedTask) -> None:
    """Write the label layer for the window and mark it completed."""
    projected_geometry = (
        STGeometry(WGS84_PROJECTION, record.geometry, time_range=None)
        .to_projection(window.projection)
        .shp
    )
    feature = Feature(
        STGeometry(window.projection, projected_geometry, time_range=None),
        get_label_properties(record),
    )
    vector_format = dataset.layers[LABEL_LAYER].instantiate_vector_format()
    with window.data.open_layer_writer(LABEL_LAYER) as writer:
        writer.write_vector(vector_format, [feature])
    window.mark_layer_completed(LABEL_LAYER)


def create_window(
    record: SelectedTask,
    dataset: Dataset,
    existing: Window | None = None,
) -> tuple[Window, str]:
    """Create the window for the record, or update an existing window's label.

    Args:
        record: the selected task.
        dataset: the rslearn dataset.
        existing: the existing window with the same group and name, if any.

    Returns:
        a tuple (window, outcome) where outcome is "created", "updated" (the label was
            rewritten because the annotation changed in Studio or the label format
            changed), or "existing".
    """
    if existing is not None:
        previous_updated_time = existing.options.get("annotation_updated_time") or ""
        if (
            existing.is_layer_completed(LABEL_LAYER)
            and existing.options.get("label_version") == LABEL_VERSION
            and record.annotation_updated_time <= previous_updated_time
        ):
            return existing, "existing"
        existing.options.update(get_window_options(record))
        existing.save()
        write_label(existing, dataset, record)
        return existing, "updated"

    projection, bounds, _ = get_window_geometry(record.geometry)
    window = Window(
        storage=dataset.storage,
        group=record.group,
        name=record.task_id,
        projection=projection,
        bounds=bounds,
        time_range=record.time_range,
        options=get_window_options(record),
        data_factory=dataset.window_data_storage_factory,
    )
    window.save()
    write_label(window, dataset, record)
    return window, "created"


def summarize(records: list[SelectedTask]) -> dict[str, Any]:
    """Summarize the selected tasks by group, label, and split."""
    by_group: Counter[str] = Counter()
    by_label: Counter[str] = Counter()
    by_leaf: Counter[str] = Counter()
    by_class: Counter[str] = Counter()
    by_split: Counter[str] = Counter()
    by_group_split: Counter[str] = Counter()
    for record in records:
        by_group[record.group] += 1
        by_label[" + ".join(hierarchy.covering_nodes(record.leaves))] += 1
        if len(record.leaves) == 1:
            by_leaf[next(iter(record.leaves))] += 1
        by_class[record.new_label or "<none>"] += 1
        by_split[record.split] += 1
        by_group_split[f"{record.group}/{record.split}"] += 1
    return {
        "selected": len(records),
        "selected_by_group": dict(sorted(by_group.items())),
        "selected_by_label": dict(sorted(by_label.items())),
        "selected_by_single_leaf": dict(sorted(by_leaf.items())),
        "selected_by_legacy_class": dict(sorted(by_class.items())),
        "selected_by_split": dict(sorted(by_split.items())),
        "selected_by_group_and_split": dict(sorted(by_group_split.items())),
    }


def create_dataset(
    ds_path: UPath,
    projects: list[StudioProject],
    client: StudioClient,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Fetch the Studio labels and create the dataset windows.

    Args:
        ds_path: the rslearn dataset path (must already contain config.json unless
            dry_run is set).
        projects: the Studio projects to include.
        client: the Studio client.
        dry_run: if set, only compute the summary and do not write anything.

    Returns:
        a JSON-serializable summary of the outcomes.
    """
    records: list[SelectedTask] = []
    skip_reasons: dict[str, dict[str, int]] = {}
    for project in projects:
        print(f"Fetching Studio project {project.project_id}")
        tasks = client.get_tasks(project.project_id)
        annotations = client.get_annotations(project.project_id)
        selected, reasons = select_tasks(project, tasks, annotations)
        print(
            f"Project {project.project_id}: {len(tasks)} tasks, "
            f"{len(selected)} selected, skipped {dict(reasons)}"
        )
        records.extend(selected)
        skip_reasons[project.project_id] = dict(sorted(reasons.items()))

    summary = summarize(records)
    summary["skipped"] = skip_reasons

    if dry_run:
        summary["outcomes"] = {"dry_run": len(records)}
        return summary

    dataset = Dataset(ds_path)
    groups = sorted({record.group for record in records})
    existing_windows: dict[tuple[str, str], Window] = {}
    if groups:
        for window in dataset.load_windows(groups=groups):
            existing_windows[(window.group, window.name)] = window
    outcomes: Counter[str] = Counter()
    for record in tqdm.tqdm(records, desc="Creating windows"):
        existing = existing_windows.get((record.group, record.task_id))
        _, outcome = create_window(record, dataset, existing)
        outcomes[outcome] += 1
    summary["outcomes"] = dict(sorted(outcomes.items()))
    return summary


def main() -> None:
    """Fetch Studio labels and create the rslearn dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ds-path", required=True, help="Path to the rslearn dataset to create."
    )
    parser.add_argument(
        "--project-id",
        action="append",
        dest="project_ids",
        help=(
            "Studio project ID to include (must be registered in PROJECTS); "
            "defaults to all registered projects."
        ),
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("data/forest_loss_driver/20261120_hierarchy/config.json"),
        help="Dataset config copied to DS_PATH/config.json when it does not exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the summary of what would be created.",
    )
    args = parser.parse_args()

    if args.project_ids:
        projects = []
        for project_id in args.project_ids:
            if project_id not in PROJECTS_BY_ID:
                raise ValueError(f"project {project_id} is not registered in PROJECTS")
            projects.append(PROJECTS_BY_ID[project_id])
    else:
        projects = list(PROJECTS)

    ds_path = UPath(args.ds_path)
    if not args.dry_run:
        ds_path.mkdir(parents=True, exist_ok=True)
        dst_config = ds_path / "config.json"
        if not dst_config.exists():
            with args.config_path.open("rb") as src, dst_config.open("wb") as dst:
                dst.write(src.read())

    summary = create_dataset(
        ds_path=ds_path,
        projects=projects,
        client=StudioClient(),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
