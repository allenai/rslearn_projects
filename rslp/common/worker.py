"""Worker to process jobs in a list of jobs."""

import json
import os
import shlex
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
from typing import Any

import tqdm
from beaker import (
    Beaker,
    BeakerConstraints,
    BeakerEnvVar,
    BeakerExperimentSpec,
    BeakerJobPriority,
    BeakerTaskResources,
)
from beaker.utils import pb2_to_dict
from upath import UPath

from rslp.log_utils import get_logger
from rslp.main import run_workflow
from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_WORKSPACE,
    WekaMount,
    create_gcp_credentials_mount,
    get_base_env_vars,
)

logger = get_logger(__name__)

# Maximum expected duration of a job in hours. We use this to limit how long we care
# about a pending claim that hasn't completed yet.
MAX_JOB_HOURS = 4

# How much of a failure's text to keep in an entry's rejection reason.
REJECTION_CHARS = 500

# How long to allow the queue channel to shut down before forcing the process out.
#
# The Beaker SDK's worker_channel closes by joining its streaming thread with no
# timeout. That thread can get stuck in the bidirectional stream it manages (a state
# its own source notes: "we stop sending or receiving new streaming messages"), and
# then the join never returns. The worker has finished its job and written its marker
# by that point, so it looks alive to Beaker, holds its GPU, and never claims again:
# 50 of 192 workers ended up that way over 37 hours of a global run. Exiting hard is
# safe here because everything durable is already written.
SHUTDOWN_GRACE_SECONDS = 120

# Minimum runtime to request for a worker, which is what makes its job *allocated*
# rather than unallocated. The scheduler treats anything at or under five minutes as
# unallocated, and unallocated jobs only run when no allocated job wants the slot, so a
# worker without this is preempted by allocated work whatever its priority. Set it to
# roughly one job: long enough to finish a unit of work, short enough to be placed
# quickly, since a shorter request fits the allocation grid sooner.
DEFAULT_WORKER_MIN_RUNTIME = timedelta(minutes=45)

# Arguments this worker appends to every entry it runs, as a shell-quoted string.
#
# Appended last, so they beat the values the supervisor baked into the entry. That is
# what lets one queue feed pools with different hardware: a GPU-memory knob like
# --batch_size has to follow the worker, not the job, once H100s and A100s share a
# queue. Kept general rather than a batch_size field, since the next such knob will
# want the same treatment.
WORKER_EXTRA_ARGS_ENV = "RSLP_WORKER_EXTRA_ARGS"


# Environment variable carrying a worker's own experiment name, set at launch. Beaker
# does not inject the experiment name, and the drain list names workers, so the worker
# has to be told who it is.
WORKER_NAME_ENV_VAR = "RSLP_WORKER_NAME"

# Ignore a drain list older than this. The supervisor rewrites the list every cycle, so
# a stale one means the supervisor is gone; without this the last list it wrote would
# keep retiring workers until the pool emptied.
DRAIN_STALE_SECONDS = 1800

# How often the prefetch thread and the main loop recheck their stop and idle state.
PREFETCH_POLL_SECONDS = 1

# How long to wait for the prefetch thread to stop on exit. It terminates its
# subprocess when stopped, so this only needs to cover that.
PREFETCH_JOIN_SECONDS = 60


def get_cleanup_signal_handler(tmp_dir: str) -> Callable[[int, Any], None]:
    """Make a signal handler that cleans up the specified directory before exiting.

    This should be passed as the handler to signal.signal.

    Args:
        tmp_dir: the directory to delete when the signal is received.
    """

    def cleanup_signal_handler(signo: int, stack_frame: Any) -> None:
        logger.error(f"cleanup_signal_handler: caught signal {signo}")
        shutil.rmtree(tmp_dir)
        sys.exit(1)

    return cleanup_signal_handler


def _release_on_termination(tx: Any, in_flight: set[str]) -> Callable[[int, Any], None]:
    """Make a SIGTERM handler that hands the in-flight entries back to the queue.

    Beaker sends SIGTERM about five minutes before it kills a preempted job. Without
    this the entries stay CLAIMED and nothing may touch them until the claims go stale,
    which is `claim_stale_seconds` later; rejecting them means the supervisor re-offers
    the jobs on its next cycle instead. The work itself is still lost, since a job is
    only marked complete once every window in it is written.

    Args:
        tx: the queue worker channel to send the rejections on.
        in_flight: ids of the entries this worker holds, the running one and any
            prefetched one.

    Returns:
        a handler to pass to signal.signal.
    """

    def handler(signo: int, stack_frame: Any) -> None:
        entry_ids = list(in_flight)
        logger.error(
            "caught signal %d; releasing entries %s back to the queue",
            signo,
            entry_ids,
        )
        for entry_id in entry_ids:
            try:
                tx.send(entry_id, rejection=f"worker terminated by signal {signo}")
            except Exception:
                # The job is going away regardless; the entry just goes stale instead.
                logger.exception("could not release entry %s", entry_id)
        in_flight.clear()
        sys.exit(1)

    return handler


def _force_exit_after(seconds: int) -> threading.Timer:
    """Start a daemon timer that hard-exits the process after `seconds`.

    Used to bound the queue channel's shutdown, which can block forever. os._exit is
    deliberate: sys.exit only raises in the calling thread, which is exactly the thread
    already stuck inside the join we are trying to escape.

    Args:
        seconds: how long to wait before forcing the exit.

    Returns:
        the started timer, so a caller that shuts down cleanly can cancel it.
    """

    def bail() -> None:
        logger.error(
            "queue channel shutdown exceeded %d seconds; forcing exit so the worker "
            "does not hold its GPU while unable to claim work",
            seconds,
        )
        os._exit(1)

    timer = threading.Timer(seconds, bail)
    timer.daemon = True
    timer.start()
    return timer


def _should_drain(drain_path: str, worker_name: str | None) -> bool:
    """Whether this worker has been asked to retire.

    Checked between jobs, never during one. A worker that is over the capacity target
    still owns a claimed job and there is no intra-job checkpointing, so the only free
    moment to stop is after one job is done and before the next is claimed. Retiring
    there costs nothing and hands the GPU back within one job.

    Never raises: a worker that cannot read the list keeps working. Failing to shrink
    the pool is a much smaller problem than a transient storage error emptying it.

    A worker that was never told its name cannot be named in the list, so it returns
    early rather than reading the list once per job to learn nothing.

    Args:
        drain_path: the drain list the supervisor publishes.
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
        if age > DRAIN_STALE_SECONDS:
            logger.warning(
                "ignoring drain list written %.0fs ago (over the %ds staleness "
                "limit); assuming no supervisor is publishing it",
                age,
                DRAIN_STALE_SECONDS,
            )
            return False
        return worker_name in published["workers"]
    except Exception:
        logger.exception("could not read drain list at %s; continuing", drain_path)
        return False


@dataclass
class _Claimed:
    """One claimed entry on its way from the channel to the main thread."""

    entry_id: str
    entry_input: dict[str, Any]
    # Appended to the entry's args when it runs, pointing it at its prefetched data.
    extra_args: list[str] = field(default_factory=list)
    # Set when prefetching failed, so the main thread rejects the entry instead.
    error: Exception | None = None


def _run_prefetch_subprocess(cmd: list[str], stop: threading.Event) -> None:
    """Run a prefetch command, terminating it early if `stop` is set.

    A subprocess rather than a thread because the main process holds a CUDA context
    once it has run one job, and the data pipelines fork worker pools, which is not
    safe after CUDA is initialized.

    Args:
        cmd: the command to run.
        stop: set when the worker is exiting and the prefetch is no longer wanted.

    Raises:
        RuntimeError: if the command fails or is stopped.
    """
    proc = subprocess.Popen(cmd)  # nosec
    while proc.poll() is None:
        if stop.wait(PREFETCH_POLL_SECONDS):
            proc.terminate()
            proc.wait()
            raise RuntimeError("prefetch stopped because the worker is exiting")
    if proc.returncode != 0:
        raise RuntimeError(f"prefetch exited with code {proc.returncode}")


class _Prefetcher(threading.Thread):
    """Takes entries off the channel and prefetches each one ahead of its turn.

    An entry opts in with a "prefetch" field, {"args": [...], "scratch_arg": "--x"}:
    the worker runs the entry's workflow in a subprocess with `args` and
    `scratch_arg <dir>` appended, then runs it for real with only `scratch_arg <dir>`
    appended. `<dir>` does not exist yet but its parent does.

    Overlap needs the queue's max_claimed_entries at 2, so that Beaker hands over the
    next entry while the current one is still running. With one claim, or entries
    without the field, the worker runs in the same serial order as before.
    """

    def __init__(
        self, rx: Any, in_flight: set[str], scratch_root: str, flush_messages: bool
    ) -> None:
        """Set up the prefetcher.

        Args:
            rx: the queue channel's receiver.
            in_flight: ids of entries this worker holds, shared with the main thread.
            scratch_root: directory to create each entry's prefetch directory in.
            flush_messages: skip prefetching, since nothing will run.
        """
        super().__init__(name="prefetcher", daemon=True)
        self.rx = rx
        self.in_flight = in_flight
        self.scratch_root = scratch_root
        self.flush_messages = flush_messages
        # Size one, so at most one entry is prefetched ahead of the running one.
        self.ready: Queue[_Claimed] = Queue(maxsize=1)
        # Set while an entry has been taken off the channel but not yet handed over.
        self.busy = threading.Event()
        self.stop = threading.Event()

    def run(self) -> None:
        """Prefetch entries until stopped or the channel closes."""
        while not self.stop.is_set():
            try:
                batch = self.rx.rx.get(block=True, timeout=PREFETCH_POLL_SECONDS)
            except QueueEmpty:
                continue
            # The SDK sends None when the channel closes.
            if batch is None:
                return
            for worker_input in batch:
                self.busy.set()
                claimed = _Claimed(
                    entry_id=worker_input.metadata.entry_id,
                    entry_input=pb2_to_dict(worker_input.input),
                )
                self.in_flight.add(claimed.entry_id)
                if not self.flush_messages:
                    self._prefetch(claimed)
                while not self.stop.is_set():
                    try:
                        self.ready.put(claimed, timeout=PREFETCH_POLL_SECONDS)
                        break
                    except QueueFull:
                        continue
                # If stopped, the entry stays in in_flight and the main thread
                # rejects it on the way out.
                self.busy.clear()

    def _prefetch(self, claimed: _Claimed) -> None:
        """Run the entry's prefetch step, if it has one.

        Args:
            claimed: the entry, updated with the args to run it with or the error.
        """
        prefetch = claimed.entry_input.get("prefetch")
        if prefetch is None:
            return
        entry_dir = tempfile.mkdtemp(dir=self.scratch_root)
        scratch_args = [prefetch["scratch_arg"], os.path.join(entry_dir, "scratch")]
        cmd = [
            sys.executable,
            "-m",
            "rslp.main",
            claimed.entry_input["project"],
            claimed.entry_input["workflow"],
            *claimed.entry_input["args"],
            *prefetch["args"],
            *scratch_args,
        ]
        logger.info("prefetching entry %s", claimed.entry_id)
        try:
            _run_prefetch_subprocess(cmd, self.stop)
            claimed.extra_args = scratch_args
            logger.info("prefetched entry %s", claimed.entry_id)
        except Exception as e:
            logger.exception("prefetch failed for entry %s", claimed.entry_id)
            claimed.error = e


def worker_pipeline(
    queue_name: str,
    retries: int = 3,
    retry_sleep: int = 60,
    max_retry_sleep: int = 600,
    idle_timeout: int = 10,
    flush_messages: bool = False,
    drain_path: str | None = None,
) -> None:
    """Start a worker to run jobs from a Beaker queue.

    The job dict including rslp project, workflow, and arguments to pass must be
    written to the queue. It may also carry a "prefetch" field, see _Prefetcher.

    Args:
        queue_name: the name of the Beaker queue.
        retries: terminate after this many errors in a row, so a worker gives up when
            it is failing systematically but not on a few scattered bad jobs. A "retry"
            may run a different job than the one that failed. The count resets on every
            success.
        retry_sleep: base seconds to sleep after an error, doubled per consecutive
            error, since a worker failing repeatedly is usually failing for a reason
            that outlasts one entry.
        max_retry_sleep: cap on that doubling.
        idle_timeout: seconds before we terminate if there is no activity.
        flush_messages: whether to just flesh messages without actually running the
            requested workflows. This is to just delete all the messages in a topic.
        drain_path: a list of worker names the supervisor wants to retire, checked
            between jobs. Pass None to never retire early.
    """
    # Read once: it cannot change while the worker runs, and a bad value should be
    # reported at startup rather than on whichever entry happens to be next.
    worker_args = shlex.split(os.environ.get(WORKER_EXTRA_ARGS_ENV, ""))
    if worker_args:
        logger.info("appending worker args to every entry: %s", worker_args)

    def process_message(json_data: dict[str, Any], extra_args: list[str]) -> None:
        logger.debug("worker received message %s", json_data)
        rslp_project = json_data["project"]
        rslp_workflow = json_data["workflow"]
        # Last wins in jsonargparse, so the worker's own args override the entry's.
        workflow_args = json_data["args"] + extra_args + worker_args
        run_workflow(rslp_project, rslp_workflow, workflow_args)

    def next_ready(prefetcher: _Prefetcher) -> _Claimed | None:
        # Idle only counts while nothing is being prefetched, since a prefetch can
        # take minutes.
        idle_since = time.monotonic()
        while True:
            try:
                return prefetcher.ready.get(timeout=PREFETCH_POLL_SECONDS)
            except QueueEmpty:
                pass
            if not prefetcher.is_alive():
                return None
            if prefetcher.busy.is_set():
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= idle_timeout:
                return None

    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        queue = beaker.queue.get(queue_name)
        worker = beaker.queue.create_worker(queue)
        logger.info("listening for messages on %s", queue_name)

        consecutive_errors = 0
        watchdog: threading.Timer | None = None
        worker_name = os.environ.get(WORKER_NAME_ENV_VAR)
        scratch_root = tempfile.mkdtemp(prefix="rslp-prefetch-")
        with beaker.queue.worker_channel(queue, worker) as (tx, rx):
            in_flight: set[str] = set()
            signal.signal(signal.SIGTERM, _release_on_termination(tx, in_flight))
            prefetcher = _Prefetcher(rx, in_flight, scratch_root, flush_messages)
            prefetcher.start()
            try:
                while True:
                    # Before taking the next entry, not after. With prefetching the
                    # next entry may already be claimed, and it is rejected on the
                    # way out so the supervisor re-offers it.
                    if drain_path is not None and _should_drain(
                        drain_path, worker_name
                    ):
                        logger.info(
                            "worker %s is on the drain list; exiting to hand back its "
                            "slot",
                            worker_name,
                        )
                        break

                    claimed = next_ready(prefetcher)
                    if claimed is None:
                        break
                    entry_id = claimed.entry_id
                    entry_input = claimed.entry_input
                    logger.info("processing entry %s", entry_id)

                    try:
                        if claimed.error is not None:
                            raise claimed.error
                        if not flush_messages:
                            process_message(entry_input, claimed.extra_args)
                        tx.send(entry_id, done=True)
                        in_flight.discard(entry_id)
                        consecutive_errors = 0
                    except Exception as e:
                        consecutive_errors += 1
                        # exc_info so the traceback survives: without it only the
                        # exception's message reaches the logs, which is rarely
                        # enough to locate a failure inside the model or dataset.
                        logger.exception(
                            "encountered error while processing message %s (%d/%d consecutive errors)",
                            entry_input,
                            consecutive_errors,
                            retries,
                        )
                        # Release the claim: Beaker never releases one on its own, so an
                        # unanswered entry stays CLAIMED and its job counts as in flight
                        # until the claim goes stale. REJECTED does not, so the
                        # supervisor re-enqueues on its next cycle.
                        try:
                            tx.send(
                                entry_id,
                                rejection=f"{type(e).__name__}: {e}"[:REJECTION_CHARS],
                            )
                        except Exception:
                            # Not worth losing the run over: the entry just goes stale.
                            logger.exception("could not reject entry %s", entry_id)
                        in_flight.discard(entry_id)
                        if consecutive_errors >= retries:
                            raise
                        time.sleep(
                            min(
                                retry_sleep * 2 ** (consecutive_errors - 1),
                                max_retry_sleep,
                            )
                        )
                    finally:
                        # The prefetched data is only needed for this one run.
                        if claimed.extra_args:
                            shutil.rmtree(
                                os.path.dirname(claimed.extra_args[-1]),
                                ignore_errors=True,
                            )
            finally:
                prefetcher.stop.set()
                prefetcher.join(timeout=PREFETCH_JOIN_SECONDS)
                # Anything still held was claimed but never run. Hand it back now
                # rather than leaving it for the claim to go stale.
                for entry_id in list(in_flight):
                    try:
                        tx.send(entry_id, rejection="worker exited before running it")
                    except Exception:
                        logger.exception("could not reject entry %s", entry_id)
                in_flight.clear()
                shutil.rmtree(scratch_root, ignore_errors=True)
                # The channel teardown that follows joins a thread that can be stuck
                # in the SDK's bidirectional stream, which would hang the worker with
                # its GPU held. Everything durable is written by now, so bound it.
                watchdog = _force_exit_after(SHUTDOWN_GRACE_SECONDS)

        # Reaching here means the teardown returned, so the channel closed cleanly and
        # the watchdog has nothing left to guard.
        if watchdog is not None:
            watchdog.cancel()


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
    drain_path: str | None = None,
    idle_timeout: int | None = None,
    name_prefix: str = "worker",
    min_runtime: timedelta = DEFAULT_WORKER_MIN_RUNTIME,
    auto_resume: bool = True,
    gcp_credentials_secret: str | None = None,
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
        idle_timeout: seconds a worker waits for new work before exiting. Left unset,
            the worker's own default applies. Raise it when a supervisor refills the
            queue on a cycle, so a worker does not quit the moment the queue drains and
            have to pay container start again.
        drain_path: the drain list to pass to each worker, so the launcher can
            retire workers by name without interrupting a job.
        name_prefix: prefix for each worker's experiment name. Pass a value unique to
            the run so its launcher can count its own workers by name; the default
            makes every run's workers indistinguishable.
        min_runtime: how long the scheduler should let a worker run before it may be
            preempted. Above five minutes the job counts as allocated; at or below it
            the job is unallocated and yields to any allocated work.
        gcp_credentials_secret: Beaker secret holding the GCP service account key,
            or None for the shared default. One run can use its own identity this
            way without moving every other rslp job onto it.
        auto_resume: whether Beaker replaces the job when it is preempted. Without it a
            preempted worker is simply gone.
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
    base_env_vars = get_base_env_vars(use_weka_prefix=False)
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        for _ in tqdm.tqdm(range(num_workers)):
            # Named up front so the worker can be told its own name, which is how the
            # drain list addresses it.
            unique_id = str(uuid.uuid4())[0:8]
            worker_name = f"{name_prefix}_{unique_id}"
            env_vars = (
                base_env_vars
                + extra_beaker_env_vars
                + [BeakerEnvVar(name=WORKER_NAME_ENV_VAR, value=worker_name)]
            )

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
                        []
                        if idle_timeout is None
                        else ["--idle_timeout", str(idle_timeout)]
                    ),
                    *([] if drain_path is None else ["--drain_path", drain_path]),
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
            _Prefetcher), or None to always run it in one step.
    """
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        queue = beaker.queue.get(queue_name)

        for args in tqdm.tqdm(args_list, desc="Writing jobs to Beaker queue"):
            json_data: dict[str, Any] = {
                "project": rslp_project,
                "workflow": rslp_workflow,
                "args": args,
            }
            if prefetch is not None:
                json_data["prefetch"] = prefetch
            beaker.queue.create_entry_async(
                queue, input=json_data, expires_in_sec=expires_in_sec
            )
