"""Unit tests for deduping multi-annotator tasks in the NISAR dataset creation script."""

from typing import Any

from rslp.nisar_vessels.scripts.create_dataset import dedupe_tasks


def make_task(
    task_id: str, crop_uid: str | None, geom_wkt: str = "POINT (0 0)"
) -> dict:
    """Make a minimal Studio task."""
    return {
        "id": task_id,
        "name": f"task {task_id}",
        "attributes": {"crop_uid": crop_uid} if crop_uid else {},
        "geom_wkt": geom_wkt,
        "start_time": "2026-06-01T00:00:00Z",
    }


def make_annotations(count: int) -> list[dict[str, Any]]:
    """Make the given number of placeholder vessel annotations."""
    return [{"id": str(i)} for i in range(count)]


def test_single_tasks_are_kept() -> None:
    """Tasks covering distinct crops are all kept, including ones with no vessels."""
    tasks = [make_task("a", "crop1"), make_task("b", "crop2")]
    kept = dedupe_tasks(tasks, {"a": make_annotations(2)})
    assert [task["id"] for task in kept] == ["a", "b"]


def test_agreeing_tasks_are_merged() -> None:
    """Tasks on the same crop with equal vessel counts are reduced to the first one."""
    tasks = [make_task("a", "crop1"), make_task("b", "crop1"), make_task("c", "crop2")]
    annotations = {"a": make_annotations(3), "b": make_annotations(3)}
    kept = dedupe_tasks(tasks, annotations)
    assert [task["id"] for task in kept] == ["a", "c"]


def test_agreeing_empty_tasks_are_merged() -> None:
    """Two annotators finding no vessels is an agreement."""
    tasks = [make_task("a", "crop1"), make_task("b", "crop1")]
    kept = dedupe_tasks(tasks, {})
    assert [task["id"] for task in kept] == ["a"]


def test_disagreeing_tasks_are_dropped() -> None:
    """Tasks on the same crop with different vessel counts are all dropped."""
    tasks = [make_task("a", "crop1"), make_task("b", "crop1"), make_task("c", "crop2")]
    annotations = {"a": make_annotations(13), "c": make_annotations(1)}
    kept = dedupe_tasks(tasks, annotations)
    assert [task["id"] for task in kept] == ["c"]


def test_tasks_without_crop_uid_group_by_geometry() -> None:
    """Without crop_uid, tasks with the same geometry and time cover the same crop."""
    tasks = [
        make_task("a", None, "POINT (1 1)"),
        make_task("b", None, "POINT (1 1)"),
        make_task("c", None, "POINT (2 2)"),
    ]
    annotations = {"a": make_annotations(1), "b": make_annotations(2)}
    kept = dedupe_tasks(tasks, annotations)
    assert [task["id"] for task in kept] == ["c"]
