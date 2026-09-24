import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import shapely
from rslearn.dataset import Dataset
from upath import UPath

from rslp.forest_loss_driver.create_dataset import (
    ACA_LABEL_MAP,
    CLASSES,
    FLAT_LABEL_MAP,
    LABEL_LAYER,
    PERU_LEGACY_LABEL_MAP,
    PROJECTS,
    StudioProject,
    assign_split,
    create_window,
    get_window_geometry,
    select_task,
    select_tasks,
)

POLYGON = "POLYGON ((-76 -8, -75.999 -8, -75.999 -8.001, -76 -8.001, -76 -8))"
TASK_POLYGON = "POLYGON ((-76.005 -7.995, -75.994 -7.995, -75.994 -8.006, -76.005 -8.006, -76.005 -7.995))"
POINT = "POINT (-75.9995 -8.0005)"

LEGACY_PROJECT = StudioProject(
    project_id="legacy",
    label_field="tag_name",
    label_map=ACA_LABEL_MAP,
    group_map={"20250428_brazil_phase1": "20250428_brazil_phase1"},
)
FLAT_PROJECT = StudioProject(
    project_id="flat",
    label_field="validate",
    label_map=FLAT_LABEL_MAP,
    fixed_group="20260821_validatetest",
    extra_fields=("category", "country"),
)


def _task(
    *,
    task_id: str = "task",
    attributes: dict[str, Any] | None = None,
    geom_wkt: str = TASK_POLYGON,
) -> dict[str, Any]:
    if attributes is None:
        attributes = {"group": "20250428_brazil_phase1", "window": "feat_x_1"}
    return {
        "id": task_id,
        "name": "task name",
        "status": "reviewed",
        "start_time": "2024-01-01T00:00:00Z",
        "end_time": "2024-01-02T00:00:00Z",
        "attributes": attributes,
        "geom_wkt": geom_wkt,
    }


def _annotation(
    *,
    annotation_id: str = "annotation",
    task_id: str = "task",
    status: str = "pending",
    labels: dict[str, str | None] | None = None,
    values: dict[str, str] | None = None,
    geom_wkt: str | None = POLYGON,
    updated_time: str = "2024-02-01T00:00:00Z",
) -> dict[str, Any]:
    if labels is None:
        labels = {"tag_name": "Mining"}
    metadata_values = [
        {"name": name, "data_type": "labelset", "label_name": label, "value": None}
        for name, label in labels.items()
    ]
    for name, value in (values or {}).items():
        metadata_values.append(
            {"name": name, "data_type": "text", "label_name": None, "value": value}
        )
    return {
        "id": annotation_id,
        "task_id": task_id,
        "status": status,
        "geom_wkt": geom_wkt,
        "metadata_values": metadata_values,
        "updated_time": updated_time,
    }


def test_label_maps_only_produce_known_classes() -> None:
    for label_map in (PERU_LEGACY_LABEL_MAP, ACA_LABEL_MAP, FLAT_LABEL_MAP):
        assert set(label_map.values()) <= set(CLASSES)
    assert PERU_LEGACY_LABEL_MAP["agriculture-mennonite"] == "agriculture"
    assert PERU_LEGACY_LABEL_MAP["coca"] == "agriculture"
    assert PERU_LEGACY_LABEL_MAP["flood"] == "river"
    assert "unknown" not in PERU_LEGACY_LABEL_MAP
    assert ACA_LABEL_MAP["False_positives_not_forest_loss_an_alert_error"] == "none"
    assert ACA_LABEL_MAP["Logging_roads"] == "road"
    assert "Mining_in_River" not in ACA_LABEL_MAP
    assert "Natural_-_Unknown" not in ACA_LABEL_MAP


def test_registered_projects_have_split_policy() -> None:
    for project in PROJECTS:
        groups = (
            [project.fixed_group]
            if project.fixed_group is not None
            else list((project.group_map or {}).values())
        )
        for group in groups:
            assert group is not None
            # Should not raise.
            assign_split(group, "some_window")


def test_assign_split_hashes_legacy_window_name() -> None:
    for window_name in ["feat_x_1279385_2198383_98104_86556", "feat_276_1223626"]:
        prefix = hashlib.sha256(window_name.encode()).hexdigest()[0]
        expected = "val" if prefix in "0123" else "train"
        assert assign_split("20250428_brazil_phase1", window_name) == expected
        assert assign_split("peru_interesting", window_name) == expected
        assert assign_split("peru3", window_name) == "train"
        assert assign_split("20260821_validatetest", None) == "train"

    with pytest.raises(ValueError):
        assign_split("20250428_brazil_phase1", None)
    with pytest.raises(ValueError):
        assign_split("not_a_group", "window")


def test_select_task_maps_label_and_split() -> None:
    selected, reason = select_task(LEGACY_PROJECT, _task(), [_annotation()])
    assert reason == "selected"
    assert selected is not None
    assert selected.new_label == "mining"
    assert selected.raw_label == "Mining"
    assert selected.group == "20250428_brazil_phase1"
    assert selected.legacy_window == "feat_x_1"
    assert selected.split == assign_split("20250428_brazil_phase1", "feat_x_1")
    assert selected.time_range == (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert selected.geometry.geom_type == "Polygon"


def test_select_task_skips() -> None:
    selected, reason = select_task(LEGACY_PROJECT, _task(), [])
    assert selected is None and reason == "missing_annotation"

    selected, reason = select_task(
        LEGACY_PROJECT, _task(), [_annotation(status="rejected")]
    )
    assert selected is None and reason == "rejected_status"

    selected, reason = select_task(
        LEGACY_PROJECT, _task(), [_annotation(labels={"tag_name": None})]
    )
    assert selected is None and reason == "missing_label"

    selected, reason = select_task(
        LEGACY_PROJECT, _task(), [_annotation(labels={"tag_name": "Mining_in_River"})]
    )
    assert selected is None and reason == "unmapped_label:Mining_in_River"

    selected, reason = select_task(
        LEGACY_PROJECT,
        _task(attributes={"group": "other_group", "window": "w"}),
        [_annotation()],
    )
    assert selected is None and reason == "unknown_group"


def test_select_task_falls_back_to_task_geometry_for_points() -> None:
    selected, reason = select_task(
        LEGACY_PROJECT, _task(), [_annotation(geom_wkt=POINT)]
    )
    assert reason == "selected"
    assert selected is not None
    assert selected.geometry.equals(shapely.from_wkt(TASK_POLYGON))

    selected, reason = select_task(
        LEGACY_PROJECT,
        _task(geom_wkt=POINT),
        [_annotation(geom_wkt=POINT)],
    )
    assert selected is None and reason == "invalid_geometry"


def test_select_tasks_uses_latest_annotation() -> None:
    annotations = [
        _annotation(
            annotation_id="old",
            labels={"tag_name": "Roads"},
            updated_time="2024-02-01T00:00:00Z",
        ),
        _annotation(
            annotation_id="new",
            labels={"tag_name": "Mining"},
            updated_time="2024-03-01T00:00:00Z",
        ),
    ]
    selected, reasons = select_tasks(LEGACY_PROJECT, [_task()], annotations)
    assert len(selected) == 1
    assert selected[0].annotation_id == "new"
    assert selected[0].new_label == "mining"
    assert reasons == {}


def test_select_task_flat_project_copies_extra_fields() -> None:
    task = _task(attributes={})
    annotation = _annotation(
        status="approved",
        labels={"validate": "burned", "category": "agriculture", "country": "br"},
        values={"estrato": "BR_agriculture"},
    )
    selected, reason = select_task(FLAT_PROJECT, task, [annotation])
    assert reason == "selected"
    assert selected is not None
    assert selected.group == "20260821_validatetest"
    assert selected.split == "train"
    assert selected.new_label == "burned"
    assert selected.extra_properties == {"category": "agriculture", "country": "br"}
    assert selected.legacy_window is None


def test_get_window_geometry_is_utm_128() -> None:
    projection, bounds, projected = get_window_geometry(shapely.from_wkt(POLYGON))
    assert projection.crs.to_epsg() == 32718  # UTM zone 18S
    assert projection.x_resolution == 10
    assert projection.y_resolution == -10
    assert bounds[2] - bounds[0] == 128
    assert bounds[3] - bounds[1] == 128
    center = projected.centroid
    assert bounds[0] <= center.x <= bounds[2]
    assert bounds[1] <= center.y <= bounds[3]


def test_create_window_writes_label_and_updates(tmp_path: Path) -> None:
    config_path = Path("data/forest_loss_driver/20260924_utm/config.json")
    (tmp_path / "config.json").write_bytes(config_path.read_bytes())
    dataset = Dataset(UPath(tmp_path))

    selected, _ = select_task(LEGACY_PROJECT, _task(), [_annotation()])
    assert selected is not None

    window, outcome = create_window(selected, dataset)
    assert outcome == "created"
    assert window.group == "20250428_brazil_phase1"
    assert window.name == "task"
    assert window.projection.crs.to_epsg() == 32718
    assert window.bounds[2] - window.bounds[0] == 128
    assert window.options["split"] == selected.split
    assert window.options["legacy_window"] == "feat_x_1"
    assert window.is_layer_completed(LABEL_LAYER)

    vector_format = dataset.layers[LABEL_LAYER].instantiate_vector_format()
    features = window.data.read_vector(LABEL_LAYER, vector_format)
    assert len(features) == 1
    properties = features[0].properties
    assert properties["new_label"] == "mining"
    assert properties["raw_label"] == "Mining"
    assert properties["studio_task_id"] == "task"
    assert properties["studio_annotation_id"] == "annotation"

    # Re-running with the same annotation leaves the window alone.
    reloaded = dataset.load_windows(groups=[window.group], names=[window.name])[0]
    _, outcome = create_window(selected, dataset, existing=reloaded)
    assert outcome == "existing"

    # A newer annotation updates the label in place.
    updated, _ = select_task(
        LEGACY_PROJECT,
        _task(),
        [
            _annotation(
                annotation_id="annotation2",
                labels={"tag_name": "Roads"},
                updated_time="2024-05-01T00:00:00Z",
            )
        ],
    )
    assert updated is not None
    _, outcome = create_window(updated, dataset, existing=reloaded)
    assert outcome == "updated"
    features = reloaded.data.read_vector(LABEL_LAYER, vector_format)
    assert features[0].properties["new_label"] == "road"
    reloaded = dataset.load_windows(groups=[window.group], names=[window.name])[0]
    assert reloaded.options["studio_annotation_id"] == "annotation2"
