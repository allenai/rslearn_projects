"""Tests for prefetching the next queue entry while the current one runs."""

import contextlib
import os
import threading
from collections.abc import Callable, Iterator
from queue import SimpleQueue
from typing import Protocol

import jsonargparse
import pytest

from rslp.common import worker as worker_mod
from rslp.large_scale_embeddings.predict_pipeline import PREFETCH


class _Input:
    """One entry as the worker sees it off the channel."""

    def __init__(self, entry_id: str, prefetch: dict | None) -> None:
        self.metadata = type("M", (), {"entry_id": entry_id})()
        self.input: dict[str, object] = {
            "project": "test",
            "workflow": "noop",
            "args": [entry_id],
        }
        if prefetch is not None:
            self.input["prefetch"] = prefetch


class _Rx:
    """Hands out scripted batches, then behaves like an idle channel."""

    def __init__(self, batches: list[list[_Input]]) -> None:
        self.rx: SimpleQueue[list[_Input] | None] = SimpleQueue()
        for batch in batches:
            self.rx.put(batch)


class _Tx:
    """Records what the worker answers."""

    def __init__(self, on_send: Callable[[], None] = lambda: None) -> None:
        self.sent: list[tuple[str, bool]] = []
        self.on_send = on_send

    def send(
        self, entry_id: str, done: bool = False, rejection: str | None = None
    ) -> None:
        self.sent.append((entry_id, done))
        self.on_send()


class _Run:
    """What the faked worker did."""

    def __init__(self, entry_ids: list[str]) -> None:
        self.lock = threading.Lock()
        self.events: list[tuple[str, str]] = []
        self.run_args: dict[str, list[str]] = {}
        self.prefetch_cmds: dict[str, list[str]] = {}
        self.prefetched = {e: threading.Event() for e in entry_ids}

    def record(self, kind: str, entry_id: str) -> None:
        with self.lock:
            self.events.append((kind, entry_id))


class Harness(Protocol):
    def __call__(
        self,
        batches: list[list[_Input]],
        prefetch_fails: frozenset[str] = frozenset(),
        worker_args: list[str] | None = None,
        serial: bool = False,
        run_seconds: float = 0,
    ) -> tuple[_Tx, _Run]: ...


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    """Run worker_pipeline against a faked queue and prefetch subprocess."""

    def go(
        batches: list[list[_Input]],
        prefetch_fails: frozenset[str] = frozenset(),
        worker_args: list[str] | None = None,
        serial: bool = False,
        run_seconds: float = 0,
    ) -> tuple[_Tx, _Run]:
        # Serial mimics max_claimed_entries 1: the next batch arrives only after the
        # current entry is answered.
        pending = list(batches[1:]) if serial else []
        rx = _Rx(batches[:1] if serial else batches)
        tx = _Tx(lambda: rx.rx.put(pending.pop(0)) if pending else None)
        run = _Run([e.metadata.entry_id for batch in batches for e in batch])

        class FakeQueueClient:
            def get(self, name: str) -> object:
                return object()

            def create_worker(self, queue: object) -> object:
                return object()

            @staticmethod
            @contextlib.contextmanager
            def worker_channel(
                queue: object, worker: object
            ) -> Iterator[tuple[_Tx, _Rx]]:
                yield tx, rx

        class FakeBeaker:
            queue = FakeQueueClient()

            @classmethod
            @contextlib.contextmanager
            def from_env(cls, **_: str) -> Iterator["FakeBeaker"]:
                yield cls()

        def fake_prefetch(cmd: list[str], stop: threading.Event) -> None:
            entry_id = cmd[cmd.index(PREFETCH["workflow"]) + 1]
            scratch = cmd[cmd.index("--scratch_path") + 1]
            run.prefetch_cmds[entry_id] = list(cmd)
            if entry_id in prefetch_fails:
                raise RuntimeError(f"prefetch boom {entry_id}")
            os.mkdir(scratch)
            run.record("prefetched", entry_id)
            run.prefetched[entry_id].set()

        def fake_run_workflow(project: str, workflow: str, args: list[str]) -> None:
            entry_id = args[0]
            run.run_args[entry_id] = list(args)
            run.record("start", entry_id)
            threading.Event().wait(run_seconds)
            # Stands in for a long inference; the next entry's prefetch should land.
            later = [e for e in run.prefetched if e > entry_id]
            if later:
                run.prefetched[min(later)].wait(timeout=5)
            run.record("end", entry_id)

        monkeypatch.setattr(worker_mod, "Beaker", FakeBeaker)
        monkeypatch.setattr(worker_mod, "pb2_to_dict", lambda d: d)
        monkeypatch.setattr(worker_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(worker_mod, "_PREFETCH_POLL_SECONDS", 0.05)
        monkeypatch.setattr(worker_mod, "_run_prefetch_subprocess", fake_prefetch)
        monkeypatch.setattr(worker_mod, "run_workflow", fake_run_workflow)
        worker_mod.worker_pipeline(
            queue_name="test/queue", idle_timeout=1, worker_args=worker_args
        )
        return tx, run

    return go


def test_the_next_entry_is_prefetched_while_the_current_one_runs(
    harness: Harness,
) -> None:
    """e2's prefetch finishes before e1's run does."""
    tx, run = harness([[_Input("e1", PREFETCH), _Input("e2", PREFETCH)]], frozenset())
    assert run.events.index(("prefetched", "e2")) < run.events.index(("end", "e1"))
    assert tx.sent == [("e1", True), ("e2", True)]


def test_an_entry_runs_against_its_prefetched_scratch(harness: Harness) -> None:
    """The real run gets the scratch arg but not the prefetch-only args."""
    _, run = harness([[_Input("e1", PREFETCH)]], frozenset())
    args = run.run_args["e1"]
    assert args == ["e1", "--scratch_path", args[2]]
    assert not os.path.exists(args[2]), "scratch must be removed after the run"


def test_an_entry_without_prefetch_runs_unchanged(harness: Harness) -> None:
    """Entries from other projects carry no prefetch field and run as before."""
    tx, run = harness([[_Input("e1", None)]], frozenset())
    assert run.run_args["e1"] == ["e1"]
    assert tx.sent == [("e1", True)]


def test_worker_args_reach_the_prefetch_and_the_run(harness: Harness) -> None:
    """A worker's own args follow the entry's in both steps, so they win."""
    _, run = harness(
        [[_Input("e1", PREFETCH), _Input("e2", None)]],
        worker_args=["--window_size", "384"],
    )
    cmd = run.prefetch_cmds["e1"]
    assert cmd[cmd.index("e1") + 1 :][:2] == ["--window_size", "384"]
    assert run.run_args["e1"][1:3] == ["--window_size", "384"]
    assert run.run_args["e2"] == ["e2", "--window_size", "384"]


def test_a_repeated_arg_takes_the_last_value() -> None:
    """Workflows parse args with jsonargparse, where the worker's later value wins."""

    def workflow(window_size: int = 480) -> int:
        return window_size

    args = ["--window_size", "480", "--window_size", "384"]
    assert jsonargparse.CLI(workflow, args=args, as_positional=False) == 384


def test_a_serial_queue_is_not_idle_while_an_entry_runs(harness: Harness) -> None:
    """With max_claimed_entries 1, a run longer than idle_timeout keeps the worker."""
    tx, _ = harness(
        [[_Input("e1", PREFETCH)], [_Input("e2", PREFETCH)]],
        serial=True,
        run_seconds=1.5,
    )
    assert tx.sent == [("e1", True), ("e2", True)]


def test_a_failed_prefetch_fails_the_entry_without_running_it(
    harness: Harness,
) -> None:
    """A prefetch failure rejects the entry and the next entry still runs."""
    tx, run = harness(
        [[_Input("e1", PREFETCH), _Input("e2", PREFETCH), _Input("e3", PREFETCH)]],
        frozenset({"e2"}),
    )
    assert "e2" not in run.run_args
    assert tx.sent == [("e1", True), ("e2", False), ("e3", True)]


def test_a_crashed_prefetch_thread_fails_the_worker(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker must not exit cleanly while the prefetch thread is dead."""

    def boom(self: worker_mod._Prefetcher, claimed: worker_mod._Claimed) -> None:
        raise ValueError("bad entry")

    monkeypatch.setattr(worker_mod._Prefetcher, "_prefetch", boom)
    with pytest.raises(RuntimeError, match="prefetch thread failed"):
        harness([[_Input("e1", PREFETCH)]], frozenset())
