"""Tests for the drain list, which is how a worker is retired without losing work."""

import json
import time
from pathlib import Path

import pytest

from rslp.common.worker import _DRAIN_STALE_SECONDS, _should_drain


def _publish(path: Path, workers: list[str], age: float = 0) -> str:
    with open(path, "w") as f:
        json.dump({"written": time.time() - age, "workers": workers}, f)
    return str(path)


@pytest.mark.parametrize(
    "worker_name, drains",
    [("worker_q_b", True), ("worker_q_c", False), (None, False)],
)
def test_only_a_named_worker_retires(
    tmp_path: Path, worker_name: str | None, drains: bool
) -> None:
    """A listed worker retires; an unlisted one, or one that does not know its name, keeps working."""
    path = _publish(tmp_path / "drain.json", ["worker_q_a", "worker_q_b"])
    assert _should_drain(path, worker_name) is drains


@pytest.mark.parametrize(
    "age, drains",
    [(_DRAIN_STALE_SECONDS - 60, True), (_DRAIN_STALE_SECONDS + 1, False)],
)
def test_a_stale_list_is_ignored(tmp_path: Path, age: float, drains: bool) -> None:
    """A list nobody has rewritten recently must not keep retiring workers."""
    path = _publish(tmp_path / "drain.json", ["worker_q_a"], age=age)
    assert _should_drain(path, "worker_q_a") is drains


@pytest.mark.parametrize("contents", [None, "{not json"])
def test_a_missing_or_unreadable_list_keeps_the_worker_working(
    tmp_path: Path, contents: str | None
) -> None:
    """Failing to shrink the pool is far cheaper than a storage blip emptying it."""
    path = tmp_path / "drain.json"
    if contents is not None:
        path.write_text(contents)
    assert _should_drain(str(path), "worker_q_a") is False
