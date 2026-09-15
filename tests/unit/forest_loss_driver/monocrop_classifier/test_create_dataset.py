from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from rslearn.dataset import Dataset
from upath import UPath

from rslp.forest_loss_driver.monocrop_classifier.create_dataset import (
    CLASS_TO_ID,
    SelectedAnnotation,
    assign_split,
    create_window,
    rasterize_label,
    select_annotation,
)

POLYGON = "POLYGON ((-76 -8, -75.999 -8, -75.999 -8.001, -76 -8.001, -76 -8))"


def _project() -> dict[str, Any]:
    return {"id": "project", "name": "Monocrop - Peru"}


def _task() -> dict[str, Any]:
    return {
        "id": "task",
        "name": "task",
        "status": "reviewed",
        "start_time": "2024-01-01T00:00:00Z",
        "geom_wkt": POLYGON,
    }


def _annotation(
    *,
    status: str = "approved",
    target: str | None = "oil_palm",
    confidence: str | None = "low",
) -> dict[str, Any]:
    values = []
    if target is not None:
        values.append({"name": "monoculture_tag", "label_name": target})
    if confidence is not None:
        values.append({"name": "confidence", "label_name": confidence})
    return {
        "id": "annotation",
        "task_id": "task",
        "status": status,
        "geom_wkt": POLYGON,
        "metadata_values": values,
    }


def test_select_annotation_uses_low_confidence_and_pending() -> None:
    selected, reason = select_annotation(
        _project(), {"task": _task()}, _annotation(status="pending")
    )
    assert reason == "selected"
    assert selected is not None
    assert selected.confidence == "low"
    assert selected.class_id == CLASS_TO_ID["oil_palm"]


def test_select_annotation_rejects_rejected_or_missing_confidence() -> None:
    selected, reason = select_annotation(
        _project(), {"task": _task()}, _annotation(status="rejected")
    )
    assert selected is None
    assert reason == "rejected_status"

    selected, reason = select_annotation(
        _project(), {"task": _task()}, _annotation(confidence=None)
    )
    assert selected is None
    assert reason == "missing_confidence"


def test_rasterize_label_masks_background() -> None:
    geometry = shapely.box(60, 60, 68, 68)
    label = rasterize_label(geometry, (0, 0, 128, 128), class_id=3)
    assert label.shape == (128, 128)
    assert label.dtype == np.uint8
    assert set(np.unique(label)) == {0, 3}
    assert label[64, 64] == 3
    assert label[0, 0] == 0


def test_create_window_writes_label_and_23_month_range(tmp_path: Path) -> None:
    config_path = Path("data/forest_loss_driver/monocrop_classifier/config.json")
    (tmp_path / "config.json").write_bytes(config_path.read_bytes())
    dataset = Dataset(UPath(tmp_path))
    geometry = shapely.from_wkt(POLYGON)
    record = SelectedAnnotation(
        project_id="project",
        project_name="Monocrop - Peru",
        task_id="task",
        task_name="task",
        annotation_id="annotation",
        annotation_status="approved",
        confidence="low",
        class_name="oil_palm",
        class_id=CLASS_TO_ID["oil_palm"],
        event_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        geometry=geometry,
        source_identity="source",
    )

    window, outcome = create_window(record, dataset)

    assert outcome == "created"
    assert window.bounds[2] - window.bounds[0] == 128
    assert window.bounds[3] - window.bounds[1] == 128
    assert (window.time_range[1] - window.time_range[0]).days == 690
    assert window.options["split"] == assign_split("source")
    assert window.is_layer_completed("label")

    band_set = dataset.layers["label"].band_sets[0]
    raster = window.data.read_raster(
        "label",
        band_set.bands,
        band_set.instantiate_raster_format(),
        window.projection,
        window.bounds,
    ).get_chw_array()
    assert raster.shape == (1, 128, 128)
    assert set(np.unique(raster)) == {0, CLASS_TO_ID["oil_palm"]}

    existing, outcome = create_window(record, dataset)
    assert outcome == "existing"
    assert existing.options["annotation_id"] == "annotation"


def test_create_window_shortens_to_available_post_months(tmp_path: Path) -> None:
    config_path = Path("data/forest_loss_driver/monocrop_classifier/config.json")
    (tmp_path / "config.json").write_bytes(config_path.read_bytes())
    dataset = Dataset(UPath(tmp_path))
    record = SelectedAnnotation(
        project_id="project",
        project_name="Monocrop - Peru",
        task_id="task",
        task_name="task",
        annotation_id="short-annotation",
        annotation_status="approved",
        confidence="high",
        class_name="oil_palm",
        class_id=CLASS_TO_ID["oil_palm"],
        event_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        geometry=shapely.from_wkt(POLYGON),
        source_identity="short-source",
    )

    window, outcome = create_window(record, dataset, max_post_months=6)

    assert outcome == "created"
    assert (window.time_range[1] - window.time_range[0]).days == 510
    assert window.options["max_post_months"] == 6
