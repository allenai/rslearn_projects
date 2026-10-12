"""What a worker does when the queue ends its entry stream.

Beaker's ProcessQueueEntries stream is closed by the server whenever it has nothing
to hand out, which the SDK reports as a closed channel rather than an error. Treating
that as the end of the worker made every idle worker exit within seconds; under a
container that restarts immediately it became a hot loop that claimed nothing, which
is how 198 GPUs spent a day restarting 1,200 times an hour. These pin the behaviour
that replaced it.
"""

import contextlib
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from rslp.common import worker as worker_mod
from tests.unit.common.test_worker_error_handling import FakeTx, real_sleep
from tests.unit.common.test_worker_prefetch import _Input


class _ClosingRx:
    """A receiver that closes the stream, then serves a script on later channels.

    One instance spans every channel the worker opens, so it can tell how many times
    the worker reconnected.
    """

    def __init__(self, scripts: list[list[list[_Input]] | None]) -> None:
        """Set up the per-channel scripts.

        Args:
            scripts: one entry per channel the worker is expected to open. None means
                the server closes that stream immediately; a list of batches means it
                serves them and then closes.
        """
        self.scripts = scripts
        self.channels = 0
        self._batches: list[list[_Input]] = []
        self._closed = False

    # A worker that reconnects forever would hang the test rather than fail it, so
    # the fake refuses past this many channels. The bug this guards is an idle clock
    # that restarts on every reconnect, which makes idle_timeout unreachable.
    MAX_CHANNELS = 100

    def open(self) -> None:
        """Start the next channel, loading its script.

        Raises:
            AssertionError: if the worker reopens more times than any correct
                implementation would.
        """
        script = self.scripts[min(self.channels, len(self.scripts) - 1)]
        self.channels += 1
        assert self.channels <= self.MAX_CHANNELS, (
            f"worker reopened the stream {self.channels} times; "
            "idle_timeout is not ending it"
        )
        self._batches = list(script or [])
        self._closed = script is None

    class _Inner:
        def __init__(self, outer: "_ClosingRx") -> None:
            self.outer = outer

        def get(self, block: bool = True, timeout: float | None = None) -> Any:
            """The next batch, or None once this channel's script is spent."""
            outer = self.outer
            if outer._batches:
                return outer._batches.pop(0)
            if not outer._closed:
                outer._closed = True
            # None is how the SDK signals that the server ended the stream.
            real_sleep(timeout or 0)
            return None

    @property
    def rx(self) -> "_ClosingRx._Inner":
        """The inner receiver the prefetcher reads from."""
        return _ClosingRx._Inner(self)


@pytest.fixture
def reconnecting(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Run worker_pipeline against a queue that closes the stream on cue.

    Args:
        monkeypatch: pytest's patcher.

    Returns:
        a callable taking per-channel scripts and returning (rx, ran, workers_created).
    """

    def go(
        scripts: list[list[list[_Input]] | None], **kwargs: Any
    ) -> tuple[Any, list, int]:
        rx = _ClosingRx(scripts)
        tx = FakeTx()
        ran: list[str] = []
        created = {"n": 0}

        @contextlib.contextmanager
        def fake_channel(queue: Any, w: Any) -> Iterator[tuple[FakeTx, Any]]:
            rx.open()
            yield tx, rx

        class FakeQueueClient:
            def get(self, name: str) -> object:
                return object()

            def create_worker(self, queue: Any) -> object:
                created["n"] += 1
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
            return None

        def fake_run_workflow(project: str, workflow: str, args: list[str]) -> None:
            ran.append(args[0])

        monkeypatch.setattr(worker_mod, "Beaker", FakeBeaker)
        monkeypatch.setattr(worker_mod, "pb2_to_dict", lambda d: d)
        monkeypatch.setattr(worker_mod.time, "sleep", lambda s: None)
        monkeypatch.setattr(worker_mod, "PREFETCH_POLL_SECONDS", 0.02)
        monkeypatch.setattr(worker_mod, "_run_prefetch_subprocess", fake_prefetch)
        monkeypatch.setattr(worker_mod, "run_workflow", fake_run_workflow)
        kwargs.setdefault("idle_timeout", 0.3)
        worker_mod.worker_pipeline(queue_name="test/queue", **kwargs)
        return rx, ran, created["n"]

    return go


def test_a_closed_stream_reopens_instead_of_ending_the_worker(
    reconnecting: Any,
) -> None:
    """A stream closed with nothing to hand out must not end the worker.

    This is the whole bug: the worker exited on the first close, so a worker that
    arrived while the queue was momentarily empty never did any work at all.
    """
    entry = _Input("e1", None)
    rx, ran, _ = reconnecting([None, [[entry]], None])

    assert ran == ["e1"], f"the entry on the second channel never ran: {ran}"
    assert rx.channels >= 2, "the worker did not reopen the stream"


def test_reconnecting_reuses_one_registration(reconnecting: Any) -> None:
    """Reopening must not register a new worker each time.

    Registrations are never removed from a queue, and the supervisor counts the fresh
    ones to report the fleet size. A registration per reconnect both inflates that
    count and accumulates: one queue reached 11,336 of them.
    """
    rx, _, created = reconnecting([None, None, None])

    assert rx.channels >= 2, "no reconnect happened, so this proves nothing"
    assert created == 1, f"expected one registration across reconnects, got {created}"


def test_the_idle_clock_is_not_reset_by_reconnecting(reconnecting: Any) -> None:
    """A stream that keeps closing must still let idle_timeout end the worker.

    Restarting the clock on each reconnect would make idle_timeout unreachable, and
    the worker would reconnect forever against an empty queue.
    """
    rx, ran, _ = reconnecting([None], idle_timeout=0)

    assert ran == [], "nothing should have run"
    # Bounded: it exits on the idle timeout rather than reopening indefinitely.
    assert rx.channels <= 3, f"reopened {rx.channels} times despite a zero idle timeout"
