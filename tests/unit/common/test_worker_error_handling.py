"""How a worker answers the queue when a job fails.

These are unit tests with a faked Beaker channel rather than integration tests,
because the behaviour under test is what the worker *says* about a failure, and the
existing integration tests can only observe what it does on success. Both bugs these
cover shipped and ran unnoticed on a real run: workers reported "succeeded" to Beaker
while every job they touched failed, and each failed job locked its queue entry for
ninety minutes.
"""

import contextlib
from collections.abc import Iterator
from queue import Empty as QueueEmpty
from typing import Any

import pytest

from rslp.common import worker as worker_mod


class FakeTx:
    """Records what the worker sends back for each entry."""

    def __init__(self) -> None:
        """Start with nothing sent."""
        self.sent: list[tuple[str, str, str | None]] = []

    def send(
        self,
        entry_id: str,
        *,
        output: dict | None = None,
        rejection: str | None = None,
        done: bool = False,
    ) -> None:
        """Record one reply.

        Args:
            entry_id: the entry being answered.
            output: worker response data, unused here.
            rejection: rejection reason, if rejecting.
            done: whether the entry is being marked done.
        """
        kind = "done" if done else "rejection" if rejection is not None else "output"
        self.sent.append((entry_id, kind, rejection))


class FakeRx:
    """Hands out scripted batches, then behaves like an idle queue."""

    def __init__(self, batches: list[list[str]]) -> None:
        """Set up the batches to deliver.

        Args:
            batches: entry ids to deliver, one list per batch.
        """
        self._batches = list(batches)
        self.rx = self

    def get(self, block: bool = True, timeout: float | None = None) -> Any:
        """Deliver the next batch, or signal idleness.

        Args:
            block: ignored.
            timeout: ignored.

        Returns:
            a list of fake worker inputs.

        Raises:
            QueueEmpty: once the script is exhausted, which is how the real worker
                learns the queue has gone quiet and exits.
        """
        if not self._batches:
            raise QueueEmpty
        return [_FakeInput(entry_id) for entry_id in self._batches.pop(0)]


class _FakeInput:
    """One entry as the worker sees it off the channel."""

    def __init__(self, entry_id: str) -> None:
        """Wrap an entry id.

        Args:
            entry_id: the id to expose through .metadata.entry_id.
        """
        self.metadata = type("M", (), {"entry_id": entry_id})()
        # Shaped like a real job payload, so process_message reaches run_workflow and
        # the test steers behaviour there rather than needing a seam in the worker.
        self.input = {"project": "test", "workflow": "noop", "args": [entry_id]}


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run worker_pipeline against a faked queue.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        a callable taking the batch script and a per-entry outcome map.
    """

    def run(
        batches: list[list[str]],
        fails: set[str],
        **kwargs: Any,
    ) -> tuple[FakeTx, list[float]]:
        tx, rx = FakeTx(), FakeRx(batches)
        slept: list[float] = []

        @contextlib.contextmanager
        def fake_channel(queue: Any, w: Any) -> Iterator[tuple[FakeTx, FakeRx]]:
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

        monkeypatch.setattr(worker_mod, "Beaker", FakeBeaker)
        monkeypatch.setattr(worker_mod, "pb2_to_dict", lambda d: d)
        monkeypatch.setattr(worker_mod.time, "sleep", lambda s: slept.append(s))

        def fake_run_workflow(project: str, workflow: str, args: Any) -> None:
            entry = args[0]
            if entry in fails:
                raise RuntimeError(f"boom {entry}")

        monkeypatch.setattr(worker_mod, "run_workflow", fake_run_workflow)
        worker_mod.worker_pipeline(queue_name="test/queue", idle_timeout=1, **kwargs)
        return tx, slept

    return run


def test_a_failed_entry_is_rejected_not_left_claimed(harness: Any) -> None:
    """A failure must answer the entry, or the job is stuck for claim_stale_seconds.

    Beaker never releases a claim on its own. An entry the worker never replies to sits
    in CLAIMED, and the supervisor counts its job as in flight until the claim goes
    stale, ninety minutes by default. That is what turned one bad batch size into a
    stalled run with sixteen locked entries.
    """
    tx, _ = harness([["e1"]], fails={"e1"}, retries=99)
    assert tx.sent == [("e1", "rejection", "RuntimeError: boom e1")]


def test_a_successful_entry_is_marked_done(harness: Any) -> None:
    """The success path is unchanged."""
    tx, _ = harness([["e1"]], fails=set())
    assert tx.sent == [("e1", "done", None)]


def test_the_error_streak_resets_on_success(harness: Any) -> None:
    """`retries` counts errors in a row, so a success in between must clear it.

    Without the reset the counter was a lifetime total, and a worker that did many jobs
    and hit `retries` scattered failures terminated on the last one.

    retries=2 is what makes this test bite. With two failures either side of a success,
    a lifetime counter reaches the limit and raises, while a streak counter is back to
    one and carries on. At retries=3 both versions pass and the test proves nothing,
    which is how it was first written.
    """
    tx, _ = harness([["f1"], ["ok"], ["f2"]], fails={"f1", "f2"}, retries=2)
    kinds = [(e, k) for e, k, _ in tx.sent]
    assert kinds == [("f1", "rejection"), ("ok", "done"), ("f2", "rejection")]


def test_consecutive_failures_terminate_the_worker(harness: Any) -> None:
    """At `retries` in a row the worker gives up, and does so loudly.

    Raising is what makes the process exit non-zero, which is the only reason Beaker
    ever shows a worker as failed. A worker that swallows everything reports success no
    matter how little it achieved.
    """
    with pytest.raises(RuntimeError):
        harness([["f1", "f2", "f3"]], fails={"f1", "f2", "f3"}, retries=3)


def test_the_backoff_doubles_and_is_capped(harness: Any) -> None:
    """A worker failing repeatedly should slow down, since the entry cannot hold a delay.

    The queue API has no per-entry delay, so the only place to back off is the worker.
    A worker failing for a reason that outlasts one entry (a bad GPU, an OOM at this
    batch size) would otherwise reject an entry every retry_sleep seconds forever.
    """
    _, slept = harness(
        [["f1", "f2", "f3", "f4", "f5"]],
        fails={"f1", "f2", "f3", "f4", "f5"},
        retries=99,
        retry_sleep=10,
        max_retry_sleep=40,
    )
    assert slept == [10, 20, 40, 40, 40]


def test_workers_request_an_allocated_min_runtime() -> None:
    """A worker must ask for more than five minutes, or its job is unallocated.

    The scheduler classifies a job as allocated only when its min_runtime exceeds five
    minutes, and unallocated jobs yield to any allocated work regardless of priority.
    Three France workers were preempted on jupiter with "allocated workloads are
    scheduled ahead of unallocated ones" while requesting a min_runtime of zero, which
    is what the retired `preemptible=True` produced.
    """
    import inspect
    from datetime import timedelta

    from rslp.common import worker

    default = inspect.signature(worker.launch_workers).parameters["min_runtime"].default
    assert default > timedelta(minutes=5), (
        f"launch_workers requests min_runtime={default}, at or under the five-minute "
        "threshold, so every worker it starts is an unallocated job"
    )
    assert default <= timedelta(hours=8), "eight hours is the scheduler's maximum"


def test_workers_are_replaced_when_preempted() -> None:
    """auto_resume must default on, or a preempted worker is simply gone."""
    import inspect

    from rslp.common import worker

    assert (
        inspect.signature(worker.launch_workers).parameters["auto_resume"].default
        is True
    )


def test_termination_releases_the_in_flight_entry() -> None:
    """SIGTERM must hand the claimed entry back, not leave it CLAIMED.

    Beaker gives about five minutes between SIGTERM and the kill. An entry left claimed
    is untouchable until it goes stale, which is `claim_stale_seconds` later, so the job
    sits idle for over an hour instead of being re-offered on the next cycle.
    """
    import signal

    from rslp.common.worker import _release_on_termination

    sent: list[dict] = []

    class _Tx:
        def send(self, entry_id, **kwargs):
            sent.append({"entry_id": entry_id, **kwargs})

    handler = _release_on_termination(_Tx(), {"entry_id": "entry-abc"})
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)

    assert len(sent) == 1, "the in-flight entry was not released"
    assert sent[0]["entry_id"] == "entry-abc"
    assert sent[0].get("rejection"), "must reject, not mark done: the work is unfinished"


def test_termination_with_no_entry_is_harmless() -> None:
    """A worker killed while idle has nothing to release and must not fail trying."""
    import signal

    from rslp.common.worker import _release_on_termination

    class _Tx:
        def send(self, entry_id, **kwargs):
            raise AssertionError("nothing should be sent when no entry is in flight")

    handler = _release_on_termination(_Tx(), {"entry_id": None})
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)
