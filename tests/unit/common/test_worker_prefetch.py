"""Prefetching the next queue entry while the current one runs.

The point is to keep the GPU busy: with two claims per worker, the next block's imagery
is materialized while the current block is on the GPU. These tests fake the queue and
the prefetch subprocess, and check the worker overlaps the two and still answers every
entry it claimed.
"""

import contextlib
import os
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from rslp.common import worker as worker_mod
from tests.unit.common.test_worker_error_handling import FakeTx, real_sleep

PREFETCH = {"args": ["--materialize_only", "true"], "scratch_arg": "--scratch_path"}


class _Input:
    """One entry as the worker sees it off the channel."""

    def __init__(self, entry_id: str, prefetch: dict | None) -> None:
        """Wrap an entry id.

        Args:
            entry_id: the id, also used as the job's only arg.
            prefetch: the entry's prefetch field, or None to omit it.
        """
        self.metadata = type("M", (), {"entry_id": entry_id})()
        self.input: dict[str, Any] = {
            "project": "test",
            "workflow": "noop",
            "args": [entry_id],
        }
        if prefetch is not None:
            self.input["prefetch"] = prefetch


class _Rx:
    """Hands out scripted batches, then behaves like an idle channel."""

    def __init__(self, batches: list[list[_Input]]) -> None:
        """Set up the batches to deliver.

        Args:
            batches: inputs to deliver, one list per batch.
        """
        self._batches = list(batches)
        self.rx = self

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Deliver the next batch, or wait out the timeout and signal idleness.

        Args:
            block: ignored.
            timeout: how long to wait when there is nothing left.

        Returns:
            the next batch.

        Raises:
            QueueEmpty: once the script is exhausted.
        """
        if not self._batches:
            real_sleep(timeout or 0)
            raise worker_mod.QueueEmpty
        return self._batches.pop(0)


class _Run:
    """What the faked worker did."""

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.lock = threading.Lock()
        self.events: list[tuple[str, str]] = []
        self.run_args: dict[str, list[str]] = {}
        self.prefetched: dict[str, threading.Event] = {}

    def record(self, kind: str, entry_id: str) -> None:
        """Append one event.

        Args:
            kind: what happened.
            entry_id: the entry it happened to.
        """
        with self.lock:
            self.events.append((kind, entry_id))


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run worker_pipeline against a faked queue and prefetch subprocess.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        a callable taking the batch script and returning (tx, run).
    """

    def go(
        batches: list[list[_Input]],
        prefetch_fails: frozenset[str] = frozenset(),
        **kwargs: Any,
    ) -> tuple[FakeTx, _Run]:
        tx, rx, run = FakeTx(), _Rx(batches), _Run()
        for batch in batches:
            for entry in batch:
                run.prefetched[entry.metadata.entry_id] = threading.Event()

        @contextlib.contextmanager
        def fake_channel(queue: Any, w: Any) -> Iterator[tuple[FakeTx, _Rx]]:
            yield tx, rx

        class FakeQueueClient:
            def get(self, name: str) -> object:
                return object()

            def create_worker(self, queue: Any) -> object:
                return object()

            worker_channel = staticmethod(fake_channel)

        class FakeBeaker:
            queue = FakeQueueClient()

            @classmethod
            def from_env(cls, **_: Any) -> Any:
                @contextlib.contextmanager
                def cm() -> Iterator[Any]:
                    yield cls()

                return cm()

        def fake_prefetch(cmd: list[str], stop: threading.Event) -> None:
            entry_id = cmd[cmd.index("noop") + 1]
            scratch = cmd[cmd.index("--scratch_path") + 1]
            assert "--materialize_only" in cmd
            assert os.path.isdir(os.path.dirname(scratch))
            assert not os.path.exists(scratch)
            if entry_id in prefetch_fails:
                raise RuntimeError(f"prefetch boom {entry_id}")
            os.mkdir(scratch)
            run.record("prefetched", entry_id)
            run.prefetched[entry_id].set()

        def fake_run_workflow(project: str, workflow: str, args: list[str]) -> None:
            entry_id = args[0]
            run.run_args[entry_id] = list(args)
            run.record("start", entry_id)
            # The next entry should be prefetched while this one runs. Waiting here
            # stands in for a long inference and proves the overlap if it arrives.
            later = [e for e in run.prefetched if e > entry_id]
            if later:
                run.prefetched[min(later)].wait(timeout=5)
            run.record("end", entry_id)

        monkeypatch.setattr(worker_mod, "Beaker", FakeBeaker)
        monkeypatch.setattr(worker_mod, "pb2_to_dict", lambda d: d)
        monkeypatch.setattr(worker_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(worker_mod, "PREFETCH_POLL_SECONDS", 0.05)
        monkeypatch.setattr(worker_mod, "_run_prefetch_subprocess", fake_prefetch)
        monkeypatch.setattr(worker_mod, "run_workflow", fake_run_workflow)
        worker_mod.worker_pipeline(queue_name="test/queue", idle_timeout=1, **kwargs)
        return tx, run

    return go


def test_the_next_entry_is_prefetched_while_the_current_one_runs(
    harness: Any,
) -> None:
    """The whole point: e2's prefetch finishes before e1's run does."""
    tx, run = harness([[_Input("e1", PREFETCH), _Input("e2", PREFETCH)]])
    assert run.events.index(("prefetched", "e2")) < run.events.index(("end", "e1"))
    assert tx.sent == [("e1", "done", None), ("e2", "done", None)]


def test_an_entry_runs_against_its_prefetched_scratch(harness: Any) -> None:
    """The real run gets the scratch arg but not the prefetch-only args."""
    _, run = harness([[_Input("e1", PREFETCH)]])
    args = run.run_args["e1"]
    assert args[:3] == ["e1", "--scratch_path", args[2]]
    assert len(args) == 3
    assert not os.path.exists(args[2]), "scratch must be removed after the run"


def test_an_entry_without_prefetch_runs_unchanged(harness: Any) -> None:
    """Entries from other projects carry no prefetch field and must run as before."""
    tx, run = harness([[_Input("e1", None)]])
    assert run.run_args["e1"] == ["e1"]
    assert tx.sent == [("e1", "done", None)]


def test_a_failed_prefetch_rejects_the_entry_without_running_it(
    harness: Any,
) -> None:
    """A prefetch failure is a job failure: reject it so the supervisor re-offers it."""
    tx, run = harness(
        [[_Input("e1", PREFETCH), _Input("e2", PREFETCH)]],
        prefetch_fails=frozenset({"e2"}),
        retries=99,
    )
    assert "e2" not in run.run_args
    assert tx.sent[0] == ("e1", "done", None)
    assert tx.sent[1][:2] == ("e2", "rejection")
    assert "prefetch boom e2" in (tx.sent[1][2] or "")


def test_draining_rejects_the_prefetched_entry(
    harness: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retiring worker must hand back the entry it claimed ahead, not strand it."""
    checks = iter([False, True])
    monkeypatch.setattr(worker_mod, "_should_drain", lambda p, n: next(checks))
    monkeypatch.setenv(worker_mod.WORKER_NAME_ENV_VAR, "w")
    tx, run = harness(
        [[_Input("e1", PREFETCH), _Input("e2", PREFETCH)]], drain_path="drain.json"
    )
    assert "e2" not in run.run_args
    assert tx.sent == [
        ("e1", "done", None),
        ("e2", "rejection", "worker exited before running it"),
    ]


def test_termination_releases_every_held_entry() -> None:
    """SIGTERM must release the prefetched entry as well as the running one."""
    import signal

    sent: list[str] = []

    class _Tx:
        def send(self, entry_id: str, **kwargs: Any) -> None:
            sent.append(entry_id)

    held = {"running", "prefetched"}
    handler = worker_mod._release_on_termination(_Tx(), held)
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)
    assert sorted(sent) == ["prefetched", "running"]
    assert held == set()
