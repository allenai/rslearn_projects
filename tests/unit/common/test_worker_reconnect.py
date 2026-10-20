"""What a worker does when the queue ends its entry stream.

Beaker's ProcessQueueEntries stream is closed by the server whenever it has nothing
to hand out, which the SDK reports as a closed channel rather than an error. Treating
that as the end of the worker made every idle worker exit within seconds; under a
container that restarts immediately it became a hot loop that claimed nothing, which
is how 198 GPUs spent a day restarting 1,200 times an hour. These pin the behaviour
that replaced it.
"""

import contextlib
import os
import pathlib
import tempfile
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
        scripts: list[list[list[_Input]] | None],
        on_open: Any = None,
        prefetch_hook: Any = None,
        run_hook: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, list, int]:
        rx = _ClosingRx(scripts)
        tx = FakeTx()
        ran: list[str] = []
        created = {"n": 0}

        @contextlib.contextmanager
        def fake_channel(queue: Any, w: Any) -> Iterator[tuple[FakeTx, Any]]:
            rx.open()
            if on_open is not None:
                on_open(rx.channels)
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
            if prefetch_hook is not None:
                prefetch_hook(cmd)

        def fake_run_workflow(project: str, workflow: str, args: list[str]) -> None:
            ran.append(args[0])
            if run_hook is not None:
                run_hook(args)

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


PREFETCH = {"args": ["--materialize_only", "true"], "scratch_arg": "--scratch_path"}


def test_a_reconnect_clears_prefetched_data_left_on_disk(
    reconnecting: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Prefetch data from a closed channel must not survive into the next one.

    A job prefetched but not run when the stream closes is dropped with its
    prefetcher. Its tens of GB stayed on disk for the life of the container, which
    filled 291 GB A100 disks in about a day.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    roots: list[str] = []
    seen: dict[int, list[str]] = {}

    def prefetch_hook(cmd: list[str]) -> None:
        scratch = cmd[cmd.index("--scratch_path") + 1]
        root = os.path.dirname(os.path.dirname(scratch))
        roots.append(root)
        # Stands in for a job prefetched on this channel and abandoned with it.
        os.makedirs(os.path.join(root, "abandoned", "scratch"))

    def on_open(channel: int) -> None:
        if roots:
            seen[channel] = sorted(os.listdir(roots[0]))

    entry = _Input("e1", PREFETCH)
    reconnecting([[[entry]], None, None], on_open=on_open, prefetch_hook=prefetch_hook)

    assert roots, "nothing was prefetched, so this proves nothing"
    assert seen.get(2) == [], f"left on disk into the next channel: {seen.get(2)}"


def test_stale_prefetch_roots_are_removed_at_startup(
    reconnecting: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """Roots left by a worker that exited abnormally are deleted when one starts."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    stale = tmp_path / f"{worker_mod.PREFETCH_ROOT_PREFIX}old"
    (stale / "entry" / "scratch").mkdir(parents=True)
    unrelated = tmp_path / "keep-me"
    unrelated.mkdir()
    seen: dict[str, bool] = {}

    def on_open(channel: int) -> None:
        seen.setdefault("stale", stale.exists())

    reconnecting([None], on_open=on_open, idle_timeout=0)

    assert seen["stale"] is False, "the stale root was still there once running"
    assert unrelated.exists(), "only prefetch roots may be removed"


def test_a_failed_prefetch_leaves_nothing_on_disk(
    reconnecting: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """A prefetch that fails after downloading most of a block must not keep it.

    Scenes missing on a provider fail a block's download about 45 minutes in, with
    most of its imagery already written. The stream stays open for hours while the
    queue is busy, so each failure left ~50 GB behind until the disk filled.
    """
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    failed: list[str] = []
    seen_while_running: dict[str, bool] = {}

    def prefetch_hook(cmd: list[str]) -> None:
        scratch = cmd[cmd.index("--scratch_path") + 1]
        os.makedirs(scratch)
        (pathlib.Path(scratch) / "imagery.tif").write_bytes(b"x" * 1024)
        if cmd[cmd.index("noop") + 1] == "e1":
            failed.append(scratch)
            raise RuntimeError("prefetch exited with code 1")

    def run_hook(args: list[str]) -> None:
        # Checked on the same channel, before any teardown could clean up.
        if args[0] == "e2" and failed:
            seen_while_running["e1_left"] = os.path.exists(os.path.dirname(failed[0]))

    reconnecting(
        [[[_Input("e1", PREFETCH), _Input("e2", PREFETCH)]], None],
        prefetch_hook=prefetch_hook,
        run_hook=run_hook,
    )

    assert failed, "nothing failed, so this proves nothing"
    assert seen_while_running == {
        "e1_left": False
    }, "the failed prefetch's data was still on disk while the next job ran"
