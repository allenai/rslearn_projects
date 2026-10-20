"""Tests for the drain list, which is how a worker is retired without losing work."""

import json
import time
from pathlib import Path

from rslp.common.worker import DRAIN_STALE_SECONDS, _should_drain


def _publish(path: Path, workers: list[str], written: float | None = None) -> str:
    with open(path, "w") as f:
        json.dump(
            {
                "written": time.time() if written is None else written,
                "workers": workers,
            },
            f,
        )
    return str(path)


def test_a_named_worker_retires(tmp_path: Path) -> None:
    """The whole mechanism: the supervisor names a worker and it stops itself."""
    path = _publish(tmp_path / "drain.json", ["worker_q_a", "worker_q_b"])
    assert _should_drain(path, "worker_q_b") is True


def test_an_unnamed_worker_keeps_working(tmp_path: Path) -> None:
    """One run's list must not retire another run's workers."""
    path = _publish(tmp_path / "drain.json", ["worker_q_a"])
    assert _should_drain(path, "worker_q_b") is False


def test_a_worker_that_does_not_know_its_name_keeps_working(tmp_path: Path) -> None:
    """Without the launch-time env var there is no way to tell who is named.

    A contract test: the guard in `_should_drain` is a short-circuit that saves a read
    per job, since an absent name cannot appear in the list either way. What must hold
    is the behaviour, that a worker of unknown name never retires itself.
    """
    path = _publish(tmp_path / "drain.json", ["worker_q_a"])
    assert _should_drain(path, None) is False


def test_a_stale_list_is_ignored(tmp_path: Path) -> None:
    """A frozen list means the supervisor died, not that the pool should empty.

    The supervisor rewrites the list every cycle, so an old one has no publisher behind
    it and would otherwise keep retiring workers until nothing was left.
    """
    path = _publish(
        tmp_path / "drain.json",
        ["worker_q_a"],
        written=time.time() - DRAIN_STALE_SECONDS - 1,
    )
    assert _should_drain(path, "worker_q_a") is False


def test_a_fresh_list_is_honoured(tmp_path: Path) -> None:
    """The staleness guard must not be so tight that it ignores live lists."""
    path = _publish(
        tmp_path / "drain.json",
        ["worker_q_a"],
        written=time.time() - DRAIN_STALE_SECONDS + 60,
    )
    assert _should_drain(path, "worker_q_a") is True


def test_a_missing_list_keeps_the_worker_working(tmp_path: Path) -> None:
    """Nothing published yet is the normal state for most of a run."""
    assert _should_drain(str(tmp_path / "absent.json"), "worker_q_a") is False


def test_an_unreadable_list_keeps_the_worker_working(tmp_path: Path) -> None:
    """Failing to shrink the pool is far cheaper than a storage blip emptying it."""
    path = tmp_path / "drain.json"
    path.write_text("{not json")
    assert _should_drain(str(path), "worker_q_a") is False
