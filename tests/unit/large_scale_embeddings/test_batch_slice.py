"""Unit tests for running one task's share of a render job list."""

import json
from pathlib import Path

import pytest

from rslp.large_scale_embeddings import batch_slice


def _jobs(tmp_path: Path, n: int) -> str:
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps({"function": "fake", "jobs": [{"i": i} for i in range(n)]})
    )
    return str(path)


def test_each_task_runs_every_nth_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tasks partition the list with no gaps and no overlaps."""
    seen: list[int] = []
    monkeypatch.setitem(batch_slice.FUNCTIONS, "fake", lambda i: seen.append(i))
    path = _jobs(tmp_path, 10)
    for task in range(3):
        batch_slice.run_batch_slice(path, task_index=task, task_count=3, workers=2)
    assert sorted(seen) == list(range(10))


def test_task_index_comes_from_the_batch_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Google Batch sets BATCH_TASK_INDEX and BATCH_TASK_COUNT for every task."""
    seen: list[int] = []
    monkeypatch.setitem(batch_slice.FUNCTIONS, "fake", lambda i: seen.append(i))
    monkeypatch.setenv("BATCH_TASK_INDEX", "1")
    monkeypatch.setenv("BATCH_TASK_COUNT", "4")
    batch_slice.run_batch_slice(_jobs(tmp_path, 10))
    assert sorted(seen) == [1, 5, 9]


def test_a_failed_job_fails_the_task_after_the_rest_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One bad job must not stop the others, but the task must fail so Batch retries it."""
    seen: list[int] = []

    def fake(i: int) -> None:
        if i == 2:
            raise OSError("transient")
        seen.append(i)

    monkeypatch.setitem(batch_slice.FUNCTIONS, "fake", fake)
    with pytest.raises(RuntimeError, match="1 of 5"):
        batch_slice.run_batch_slice(_jobs(tmp_path, 5), task_index=0, task_count=1)
    assert sorted(seen) == [0, 1, 3, 4]


def test_write_rejects_an_unknown_function(tmp_path: Path) -> None:
    """A typo in the driver should fail at write time, not on every task."""
    with pytest.raises(ValueError, match="unknown function"):
        batch_slice.write_batch_jobs(str(tmp_path / "j.json"), "nope", [])
