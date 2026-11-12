"""Worker to process jobs in a list of jobs."""

import json
import os
import shutil
import signal
import subprocess  # nosec
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from queue import Empty as QueueEmpty
from queue import Full as QueueFull
from queue import Queue
from types import FrameType
from typing import Any

import tqdm
from beaker import (
    Beaker,
    BeakerConstraints,
    BeakerEntryReceiver,
    BeakerEntrySender,
    BeakerEnvVar,
    BeakerExperimentSpec,
    BeakerJobPriority,
    BeakerQueue,
    BeakerQueueWorker,
    BeakerTaskResources,
)
from beaker.utils import pb2_to_dict
from upath import UPath

from rslp.log_utils import get_logger
from rslp.main import run_workflow
from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_MIN_RUNTIME,
    DEFAULT_WORKSPACE,
    WekaMount,
    create_gcp_credentials_mount,
    get_base_env_vars,
)

logger = get_logger(__name__)

# Maximum expected duration of a job in hours. We use this to limit how long we care
# about a pending claim that hasn't completed yet.
MAX_JOB_HOURS = 4

# How often the prefetch thread and the main loop recheck their state.
_PREFETCH_POLL_SECONDS = 1

# How long to wait for the prefetch thread to stop on exit.
_PREFETCH_JOIN_SECONDS = 60

# How much of a failure's text to keep in an entry's rejection reason.
_REJECTION_CHARS = 500

# How long to allow the queue channel to shut down before forcing the process out.
# beaker-py closes the channel by joining its streaming thread with no timeout, and that
# thread can get stuck, leaving a worker that holds its GPU but never claims again.
_SHUTDOWN_GRACE_SECONDS = 120

# Environment variable carrying a worker's own experiment name, set at launch, so the
# worker can tell whether the drain list names it.
_WORKER_NAME_ENV_VAR = "RSLP_WORKER_NAME"

# Ignore a drain list older than this. Its publisher rewrites it regularly, so a stale
# one means nothing is maintaining it, and it would otherwise retire every worker.
_DRAIN_STALE_SECONDS = 1800

# Backoff between reopening a closed entry stream, and its ceiling.
_RECONNECT_BACKOFF_SECONDS = 2
_MAX_RECONNECT_SLEEP = 30


def get_cleanup_signal_handler(
    tmp_dir: str,
) -> Callable[[int, FrameType | None], None]:
    """Make a signal handler that cleans up the specified directory before exiting.

    This should be passed as the handler to signal.signal.

    Args:
        tmp_dir: the directory to delete when the signal is received.
    """

    def cleanup_signal_handler(signo: int, stack_frame: FrameType | None) -> None:
        logger.error(f"cleanup_signal_handler: caught signal {signo}")
        shutil.rmtree(tmp_dir)
        sys.exit(1)

    return cleanup_signal_handler


def _stop_on_termination(
    stop: threading.Event, in_flight: set[str]
) -> Callable[[int, FrameType | None], None]:
    """Make a SIGTERM handler that stops claiming entries and exits.

    Beaker sends SIGTERM a few minutes before it kills a preempted job. Held entries
    are left claimed rather than rejected: Beaker refills a freed claim slot at once,
    so a rejection would only hand the exiting worker another entry to strand.

    Args:
        stop: set to stop the prefetcher taking entries off the channel.
        in_flight: ids of the entries this worker holds, for the log.

    Returns:
        a handler to pass to signal.signal.
    """

    def handler(signo: int, stack_frame: FrameType | None) -> None:
        stop.set()
        logger.error(
            "caught signal %d; exiting and leaving entries %s claimed",
            signo,
            sorted(in_flight),
        )
        sys.exit(1)

    return handler


def _force_exit_after(seconds: float) -> threading.Timer:
    """Start a daemon timer that hard-exits the process after `seconds`.

    os._exit rather than sys.exit, which would only raise in the timer's thread and not
    in the thread stuck in the channel's shutdown.

    Args:
        seconds: how long to wait before forcing the exit.

    Returns:
        the started timer, to cancel once the shutdown returns.
    """

    def bail() -> None:
        logger.error(
            "queue channel shutdown exceeded %s seconds; forcing exit", seconds
        )
        os._exit(1)

    timer = threading.Timer(seconds, bail)
    timer.daemon = True
    timer.start()
    return timer


def _should_drain(drain_path: str, worker_name: str | None) -> bool:
    """Whether the drain list names this worker, asking it to retire.

    The list is JSON, {"written": <unix time>, "workers": [<names>]}. It is checked
    between entries, the only point a worker can stop without losing work. Never
    raises: a worker that cannot read the list keeps working.

    Args:
        drain_path: the drain list's path.
        worker_name: this worker's experiment name, or None if it was not told.

    Returns:
        whether this worker should exit.
    """
    if worker_name is None:
        return False
    try:
        upath = UPath(drain_path)
        if not upath.exists():
            return False
        with upath.open() as f:
            published = json.load(f)
        age = time.time() - float(published["written"])
        if age > _DRAIN_STALE_SECONDS:
            logger.warning("ignoring drain list written %.0fs ago", age)
            return False
        return worker_name in published["workers"]
    except Exception:
        logger.exception("could not read drain list at %s; continuing", drain_path)
        return False


def _empty_dir(path: str) -> None:
    """Delete everything inside a directory, keeping the directory."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


@dataclass
class _Claimed:
    """One claimed entry, handed from the prefetch thread to the main thread."""

    entry_id: str
    entry_input: dict[str, Any]
    # Appended to the entry's args when it runs, pointing it at its prefetched data.
    extra_args: list[str] = field(default_factory=list)
    entry_dir: str | None = None
    # Set when prefetching failed, so the main thread fails the entry.
    error: Exception | None = None


def _run_prefetch_subprocess(cmd: list[str], stop: threading.Event) -> None:
    """Run a prefetch command, terminating it early if `stop` is set.

    A subprocess because the main process holds a CUDA context after its first job,
    and the data pipelines fork, which is unsafe once CUDA is initialized.

    Args:
        cmd: the command to run.
        stop: set when the worker is exiting.

    Raises:
        RuntimeError: if the command fails or is stopped.
    """
    proc = subprocess.Popen(cmd)  # nosec
    while proc.poll() is None:
        if stop.wait(_PREFETCH_POLL_SECONDS):
            proc.terminate()
            proc.wait()
            raise RuntimeError("prefetch stopped because the worker is exiting")
    if proc.returncode != 0:
        raise RuntimeError(f"prefetch exited with code {proc.returncode}")


class _Prefetcher(threading.Thread):
    """Takes entries off the channel and prefetches each one ahead of its turn.

    An entry opts in with a "prefetch" key in its queue input, set by write_jobs:
    {"workflow": "...", "scratch_arg": "--x"}. That workflow runs first, in a
    subprocess, with the entry's args plus `scratch_arg <dir>`, and writes what the
    main workflow needs to `<dir>`. The main workflow then gets `scratch_arg <dir>` too.

    Prefetching needs max_claimed_entries at 2; with 1, entries run serially. The
    thread stops after idle_timeout with no entry arriving, waiting, or running, or
    when the server closes the channel.
    """

    def __init__(
        self,
        rx: BeakerEntryReceiver,
        in_flight: set[str],
        scratch_root: str,
        flush_messages: bool,
        worker_args: list[str],
        idle_timeout: float,
        idle_since: float,
    ) -> None:
        """Set up the prefetcher.

        Args:
            rx: the queue channel's receiver.
            in_flight: ids of entries this worker holds, shared with the main thread.
            scratch_root: directory to create each entry's prefetch directory in.
            flush_messages: skip prefetching, since nothing will run.
            worker_args: the worker's own args, appended as for the real run.
            idle_timeout: seconds of idleness before the thread stops.
            idle_since: when the worker last had work, carried over from the previous
                channel so reconnecting does not restart the idle clock.
        """
        super().__init__(name="prefetcher", daemon=True)
        self.rx = rx
        self.in_flight = in_flight
        self.scratch_root = scratch_root
        self.flush_messages = flush_messages
        self.worker_args = worker_args
        self.idle_timeout = idle_timeout
        self.idle_since = idle_since
        # At most one entry is prefetched ahead of the running one.
        self.ready: Queue[_Claimed] = Queue(maxsize=1)
        # Set by the main thread while it processes an entry.
        self.running = threading.Event()
        self.stop = threading.Event()
        # Set when the server closed the channel, as opposed to the worker stopping.
        self.channel_closed = threading.Event()
        self.error: BaseException | None = None

    def run(self) -> None:
        """Prefetch entries until stopped or the channel closes."""
        try:
            self._run()
        except BaseException as e:
            logger.exception("prefetch thread failed")
            self.error = e

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                batch = self.rx.rx.get(block=True, timeout=_PREFETCH_POLL_SECONDS)
            except QueueEmpty:
                if self.running.is_set() or not self.ready.empty():
                    self.idle_since = time.monotonic()
                elif time.monotonic() - self.idle_since >= self.idle_timeout:
                    logger.info("no entries for %s seconds", self.idle_timeout)
                    return
                continue
            # beaker-py sends None when the server ends the stream, which happens
            # routinely, so the worker reconnects rather than exiting.
            if batch is None:
                self.channel_closed.set()
                return
            for worker_input in batch:
                if self.stop.is_set():
                    # Left claimed; the queue re-offers it once the claim goes stale.
                    return
                claimed = _Claimed(
                    entry_id=worker_input.metadata.entry_id,
                    entry_input=pb2_to_dict(worker_input.input),
                )
                self.in_flight.add(claimed.entry_id)
                if not self.flush_messages:
                    self._prefetch(claimed)
                while not self.stop.is_set():
                    try:
                        self.ready.put(claimed, timeout=_PREFETCH_POLL_SECONDS)
                        break
                    except QueueFull:
                        continue
            self.idle_since = time.monotonic()

    def _prefetch(self, claimed: _Claimed) -> None:
        """Run the entry's prefetch step, if it has one.

        Args:
            claimed: the entry, updated with the args to run it with or the error.
        """
        prefetch = claimed.entry_input.get("prefetch")
        if prefetch is None:
            return
        claimed.entry_dir = tempfile.mkdtemp(dir=self.scratch_root)
        scratch_args = [
            prefetch["scratch_arg"],
            os.path.join(claimed.entry_dir, "scratch"),
        ]
        cmd = [
            sys.executable,
            "-m",
            "rslp.main",
            claimed.entry_input["project"],
            prefetch["workflow"],
            *claimed.entry_input["args"],
            *self.worker_args,
            *scratch_args,
        ]
        logger.info("prefetching entry %s", claimed.entry_id)
        try:
            _run_prefetch_subprocess(cmd, self.stop)
            claimed.extra_args = scratch_args
        except Exception as e:
            logger.exception("prefetch failed for entry %s", claimed.entry_id)
            claimed.error = e


class _QueueWorker:
    """Runs entries from one queue registration, reopening the channel as needed."""

    def __init__(
        self,
        retries: int,
        retry_sleep: int,
        max_retry_sleep: int,
        idle_timeout: float,
        flush_messages: bool,
        worker_args: list[str],
        drain_path: str | None,
        scratch_root: str,
    ) -> None:
        """Set up the worker; see worker_pipeline for the arguments."""
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.max_retry_sleep = max_retry_sleep
        self.idle_timeout = idle_timeout
        self.flush_messages = flush_messages
        self.worker_args = worker_args
        self.drain_path = drain_path
        self.scratch_root = scratch_root
        self.worker_name = os.environ.get(_WORKER_NAME_ENV_VAR)
        self.on_main_thread = threading.current_thread() is threading.main_thread()
        # Ids of the entries this worker holds, shared with the prefetch thread.
        self.in_flight: set[str] = set()
        self.consecutive_errors = 0
        self.idle_since = time.monotonic()

    def run(
        self, beaker: Beaker, queue: BeakerQueue, worker: BeakerQueueWorker
    ) -> None:
        """Process entries until idle, drained, or failing, reconnecting as needed."""
        reconnects = 0
        while self._run_channel(beaker, queue, worker):
            reconnects += 1
            idle_for = time.monotonic() - self.idle_since
            logger.info(
                "queue closed the entry stream (reconnect %d, idle %.0fs of %.0fs)",
                reconnects,
                idle_for,
                self.idle_timeout,
            )
            if idle_for >= self.idle_timeout:
                return
            time.sleep(
                min(_RECONNECT_BACKOFF_SECONDS * reconnects, _MAX_RECONNECT_SLEEP)
            )

    def _run_channel(
        self, beaker: Beaker, queue: BeakerQueue, worker: BeakerQueueWorker
    ) -> bool:
        """Process entries on one channel.

        Returns:
            whether the server closed the channel and the worker should reconnect.
        """
        watchdog: threading.Timer | None = None
        try:
            with beaker.queue.worker_channel(queue, worker) as (tx, rx):
                prefetcher = _Prefetcher(
                    rx,
                    self.in_flight,
                    self.scratch_root,
                    self.flush_messages,
                    self.worker_args,
                    self.idle_timeout,
                    self.idle_since,
                )
                if self.on_main_thread:
                    signal.signal(
                        signal.SIGTERM,
                        _stop_on_termination(prefetcher.stop, self.in_flight),
                    )
                prefetcher.start()
                reconnect = False
                try:
                    reconnect = self._process_entries(prefetcher, tx)
                finally:
                    prefetcher.stop.set()
                    prefetcher.join(timeout=_PREFETCH_JOIN_SECONDS)
                    self.idle_since = prefetcher.idle_since
                    if reconnect:
                        # Drop data prefetched for entries that never ran.
                        _empty_dir(self.scratch_root)
                        self.in_flight.clear()
                    else:
                        if self.in_flight:
                            logger.info(
                                "exiting; leaving entries %s claimed",
                                sorted(self.in_flight),
                            )
                        watchdog = _force_exit_after(_SHUTDOWN_GRACE_SECONDS)
            return reconnect
        finally:
            # Reached only once the channel's shutdown returned.
            if watchdog is not None:
                watchdog.cancel()

    def _process_entries(self, prefetcher: _Prefetcher, tx: BeakerEntrySender) -> bool:
        """Run entries as the prefetcher hands them over.

        Returns:
            whether the loop ended because the server closed the channel.
        """
        while True:
            # Before taking the next entry, the only point the worker can stop
            # without losing work.
            if self.drain_path is not None and _should_drain(
                self.drain_path, self.worker_name
            ):
                logger.info("worker %s is on the drain list; exiting", self.worker_name)
                return False
            claimed = self._next_ready(prefetcher)
            if claimed is None:
                return prefetcher.channel_closed.is_set()
            prefetcher.running.set()
            try:
                self._run_entry(claimed, tx)
            finally:
                self.in_flight.discard(claimed.entry_id)
                prefetcher.running.clear()
                if claimed.entry_dir is not None:
                    shutil.rmtree(claimed.entry_dir, ignore_errors=True)

    def _next_ready(self, prefetcher: _Prefetcher) -> _Claimed | None:
        """The next entry, or None once the prefetch thread stops."""
        while True:
            try:
                return prefetcher.ready.get(timeout=_PREFETCH_POLL_SECONDS)
            except QueueEmpty:
                pass
            if not prefetcher.is_alive():
                if prefetcher.channel_closed.is_set():
                    # A waiting entry cannot be answered on the closed channel; it is
                    # left claimed and re-offered.
                    return None
                try:
                    return prefetcher.ready.get_nowait()
                except QueueEmpty:
                    pass
                if prefetcher.error is not None:
                    raise RuntimeError("prefetch thread failed") from prefetcher.error
                return None

    def _run_entry(self, claimed: _Claimed, tx: BeakerEntrySender) -> None:
        """Run one entry and answer it: done on success, rejected on failure."""
        logger.info("processing entry %s", claimed.entry_id)
        try:
            if claimed.error is not None:
                raise claimed.error
            if not self.flush_messages:
                entry = claimed.entry_input
                logger.debug("worker received message %s", entry)
                run_workflow(
                    entry["project"],
                    entry["workflow"],
                    entry["args"] + self.worker_args + claimed.extra_args,
                )
            tx.send(claimed.entry_id, done=True)
            self.consecutive_errors = 0
        except Exception as e:
            self.consecutive_errors += 1
            logger.exception(
                "encountered error while processing message %s (%d/%d consecutive errors)",
                claimed.entry_input,
                self.consecutive_errors,
                self.retries,
            )
            # Beaker never releases a claim on its own, so an unanswered entry stays
            # claimed until it expires.
            try:
                tx.send(
                    claimed.entry_id,
                    rejection=f"{type(e).__name__}: {e}"[:_REJECTION_CHARS],
                )
            except Exception:
                logger.exception("could not reject entry %s", claimed.entry_id)
            if self.consecutive_errors >= self.retries:
                raise
            time.sleep(
                min(
                    self.retry_sleep * 2 ** (self.consecutive_errors - 1),
                    self.max_retry_sleep,
                )
            )


def worker_pipeline(
    queue_name: str,
    retries: int = 3,
    retry_sleep: int = 60,
    max_retry_sleep: int = 600,
    idle_timeout: float = 10,
    flush_messages: bool = False,
    worker_args: list[str] | None = None,
    drain_path: str | None = None,
) -> None:
    """Start a worker to run jobs from a Beaker queue.

    The job dict including rslp project, workflow, and arguments to pass must be
    written to the queue. It may also carry a "prefetch" key, see _Prefetcher.

    A failed entry is rejected, so it does not stay claimed. If the server closes the
    entry stream, the worker reopens it until idle_timeout passes without work.

    Args:
        queue_name: the name of the Beaker queue.
        retries: retry for this many consecutive errors before terminating. A "retry"
            may run a different job than the one that originally caused failure. This
            ensures workers will complete most of the jobs before they terminate due to
            errors. The count resets on every success.
        retry_sleep: seconds to sleep after an error, doubled for each consecutive
            error. Sleeping helps in case there is an error due to rate limiting.
        max_retry_sleep: cap on that doubling.
        idle_timeout: seconds before we terminate if there is no activity.
        flush_messages: whether to just flesh messages without actually running the
            requested workflows. This is to just delete all the messages in a topic.
        worker_args: args appended to every entry this worker runs, overriding the
            entry's own, e.g. a smaller --window_size on GPUs with less memory.
        drain_path: a drain list (see _should_drain) checked between entries, so the
            worker can be retired without interrupting a job, or None.
    """
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        queue = beaker.queue.get(queue_name)
        worker = beaker.queue.create_worker(queue)
        logger.info("listening for messages on %s", queue_name)

        queue_worker = _QueueWorker(
            retries=retries,
            retry_sleep=retry_sleep,
            max_retry_sleep=max_retry_sleep,
            idle_timeout=idle_timeout,
            flush_messages=flush_messages,
            worker_args=worker_args or [],
            drain_path=drain_path,
            scratch_root=tempfile.mkdtemp(prefix="rslp-prefetch-"),
        )
        previous_sigterm = signal.getsignal(signal.SIGTERM)
        try:
            queue_worker.run(beaker, queue, worker)
        finally:
            if queue_worker.on_main_thread:
                signal.signal(signal.SIGTERM, previous_sigterm)
            shutil.rmtree(queue_worker.scratch_root, ignore_errors=True)


def launch_workers(
    image_name: str,
    queue_name: str,
    num_workers: int,
    cluster: list[str],
    gpus: int = 0,
    shared_memory: str | None = None,
    priority: BeakerJobPriority = BeakerJobPriority.low,
    weka_mounts: list[WekaMount] = [],
    extra_env_vars: dict[str, str] | None = None,
    extra_env_secrets: dict[str, str] | None = None,
    min_runtime: timedelta = DEFAULT_MIN_RUNTIME,
    gcp_credentials_secret: str | None = None,
    worker_args: list[str] | None = None,
    idle_timeout: int | None = None,
    drain_path: str | None = None,
    name_prefix: str = "worker",
    auto_resume: bool | None = None,
) -> None:
    """Start workers for the prediction jobs.

    Args:
        image_name: the Beaker image name to use for the jobs.
        queue_name: the Beaker queue name.
        num_workers: number of workers to launch
        cluster: clusters to target.
        gpus: number of GPUs to request per worker.
        shared_memory: shared memory string like "256GiB".
        priority: priority to assign the Beaker jobs.
        weka_mounts: list of weka mounts for Beaker job.
        extra_env_vars: additional environment variables to set on each worker, beyond
            the base env vars, mapping environment variable name to its plain value.
        extra_env_secrets: additional environment variables to set on each worker from
            Beaker secrets, mapping environment variable name to the name of the Beaker
            secret (in the target workspace) to read its value from.
        min_runtime: how long to protect each (preemptible) worker job from
            preemption.
        gcp_credentials_secret: Beaker secret holding the GCP service account key, or
            None for the default.
        worker_args: args each worker appends to every entry it runs (see
            worker_pipeline).
        idle_timeout: seconds a worker waits for new work before exiting, or None for
            the worker's default.
        drain_path: the drain list each worker checks between entries (see
            worker_pipeline), or None.
        name_prefix: prefix for each worker's experiment name, which is also the name
            the drain list uses.
        auto_resume: whether Beaker replaces a preempted worker, or None for Beaker's
            default.
    """
    if extra_env_vars is None:
        extra_env_vars = {}
    if extra_env_secrets is None:
        extra_env_secrets = {}
    extra_beaker_env_vars = [
        BeakerEnvVar(name=env_name, value=env_value)
        for env_name, env_value in extra_env_vars.items()
    ]
    extra_beaker_env_vars += [
        BeakerEnvVar(name=env_name, secret=secret_name)
        for env_name, secret_name in extra_env_secrets.items()
    ]
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        for _ in tqdm.tqdm(range(num_workers)):
            worker_name = f"{name_prefix}_{str(uuid.uuid4())[0:8]}"
            env_vars = get_base_env_vars(use_weka_prefix=False)
            env_vars += extra_beaker_env_vars
            env_vars.append(BeakerEnvVar(name=_WORKER_NAME_ENV_VAR, value=worker_name))

            datasets = [
                create_gcp_credentials_mount(gcp_credentials_secret)
                if gcp_credentials_secret
                else create_gcp_credentials_mount()
            ]
            datasets += [weka_mount.to_data_mount() for weka_mount in weka_mounts]

            spec = BeakerExperimentSpec.new(
                budget=DEFAULT_BUDGET,
                description="worker",
                beaker_image=image_name,
                priority=priority,
                command=["python", "-m", "rslp.main"],
                arguments=[
                    "common",
                    "worker",
                    "--queue_name",
                    queue_name,
                    *(
                        ["--worker_args", json.dumps(worker_args)]
                        if worker_args
                        else []
                    ),
                    *(
                        ["--idle_timeout", str(idle_timeout)]
                        if idle_timeout is not None
                        else []
                    ),
                    *(["--drain_path", drain_path] if drain_path else []),
                ],
                constraints=BeakerConstraints(
                    cluster=cluster,
                ),
                min_runtime=min_runtime,
                auto_resume=auto_resume,
                datasets=datasets,
                env_vars=env_vars,
                resources=BeakerTaskResources(
                    gpu_count=gpus, shared_memory=shared_memory
                ),
            )
            beaker.experiment.create(name=worker_name, spec=spec)


def write_jobs(
    queue_name: str,
    rslp_project: str,
    rslp_workflow: str,
    args_list: list[list[str]],
    expires_in_sec: int = 7 * 24 * 3600,
    prefetch: dict[str, Any] | None = None,
) -> None:
    """Write tasks to the Beaker queue.

    Args:
        queue_name: the Beaker queue to write to.
        rslp_project: the rslp project to run.
        rslp_workflow: the workflow in the project to run.
        args_list: list of arguments fo reach task.
        expires_in_sec: how long until the queue entries should expire
        prefetch: how a worker may prefetch each task ahead of its turn (see
            _Prefetcher), or None to run it in one step.
    """
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        queue = beaker.queue.get(queue_name)

        for args in tqdm.tqdm(args_list, desc="Writing jobs to Beaker queue"):
            json_data = dict(
                project=rslp_project,
                workflow=rslp_workflow,
                args=args,
            )
            if prefetch is not None:
                json_data["prefetch"] = prefetch
            beaker.queue.create_entry_async(
                queue, input=json_data, expires_in_sec=expires_in_sec
            )
