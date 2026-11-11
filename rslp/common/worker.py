"""Worker to process jobs in a list of jobs."""

import glob
import json
import os
import shutil
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
    BeakerEntryReceiver,
    BeakerEnvVar,
    BeakerExperimentSpec,
    BeakerJobPriority,
    BeakerTaskResources,
)
from beaker.utils import pb2_to_dict

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
PREFETCH_POLL_SECONDS = 1

# How long to wait for the prefetch thread to stop on exit.
PREFETCH_JOIN_SECONDS = 60

# Prefix of the temp directory a worker keeps prefetched data in.
PREFETCH_ROOT_PREFIX = "rslp-prefetch-"


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
        if stop.wait(PREFETCH_POLL_SECONDS):
            proc.terminate()
            proc.wait()
            raise RuntimeError("prefetch stopped because the worker is exiting")
    if proc.returncode != 0:
        raise RuntimeError(f"prefetch exited with code {proc.returncode}")


class _Prefetcher(threading.Thread):
    """Takes entries off the channel and prefetches each one ahead of its turn.

    An entry opts in with a "prefetch" field, {"args": [...], "scratch_arg": "--x"}.
    Its workflow first runs in a subprocess with `args` and `scratch_arg <dir>`
    appended, then runs for real with only `scratch_arg <dir>` appended.

    Overlap needs the queue's max_claimed_entries at 2, so the next entry arrives while
    the current one runs. Otherwise the worker runs entries serially as before.
    """

    def __init__(
        self,
        rx: BeakerEntryReceiver,
        scratch_root: str,
        flush_messages: bool,
        worker_args: list[str],
    ) -> None:
        """Set up the prefetcher.

        Args:
            rx: the queue channel's receiver.
            scratch_root: directory to create each entry's prefetch directory in.
            flush_messages: skip prefetching, since nothing will run.
            worker_args: the worker's own args, appended as for the real run.
        """
        super().__init__(name="prefetcher", daemon=True)
        self.rx = rx
        self.scratch_root = scratch_root
        self.flush_messages = flush_messages
        self.worker_args = worker_args
        # At most one entry is prefetched ahead of the running one.
        self.ready: Queue[_Claimed] = Queue(maxsize=1)
        # Set while an entry is taken off the channel but not yet handed over.
        self.busy = threading.Event()
        self.stop = threading.Event()
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
                batch = self.rx.rx.get(block=True, timeout=PREFETCH_POLL_SECONDS)
            except QueueEmpty:
                continue
            if batch is None:
                return
            for worker_input in batch:
                if self.stop.is_set():
                    # Left claimed; the queue re-offers it once the claim goes stale.
                    return
                self.busy.set()
                claimed = _Claimed(
                    entry_id=worker_input.metadata.entry_id,
                    entry_input=pb2_to_dict(worker_input.input),
                )
                if not self.flush_messages:
                    self._prefetch(claimed)
                while not self.stop.is_set():
                    try:
                        self.ready.put(claimed, timeout=PREFETCH_POLL_SECONDS)
                        break
                    except QueueFull:
                        continue
                self.busy.clear()

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
            claimed.entry_input["workflow"],
            *claimed.entry_input["args"],
            *self.worker_args,
            *prefetch["args"],
            *scratch_args,
        ]
        logger.info("prefetching entry %s", claimed.entry_id)
        try:
            _run_prefetch_subprocess(cmd, self.stop)
            claimed.extra_args = scratch_args
        except Exception as e:
            logger.exception("prefetch failed for entry %s", claimed.entry_id)
            claimed.error = e


def _remove_stale_scratch_roots() -> None:
    """Delete prefetch directories left by an earlier worker in this container."""
    pattern = os.path.join(tempfile.gettempdir(), PREFETCH_ROOT_PREFIX + "*")
    for root in glob.glob(pattern):
        logger.info("removing stale prefetch directory %s", root)
        shutil.rmtree(root, ignore_errors=True)


def worker_pipeline(
    queue_name: str,
    retries: int = 3,
    retry_sleep: int = 60,
    idle_timeout: int = 10,
    flush_messages: bool = False,
    worker_args: list[str] | None = None,
) -> None:
    """Start a worker to run jobs from a Pub/Sub subscription.

    The job dict including rslp project, workflow, and arguments to pass must be
    written to the topic. It may also carry a "prefetch" field, see _Prefetcher.

    Args:
        queue_name: the name of the Beaker queue.
        retries: retry for this many consecutive errors before terminating. A "retry"
            may run a different job than the one that originally caused failure. This
            ensures workers will complete most of the jobs before they terminate due to
            errors.
        retry_sleep: sleep for this many seconds between retries. Sleeping helps in
            case there is an error due to rate limiting.
        idle_timeout: seconds before we terminate if there is no activity.
        flush_messages: whether to just flesh messages without actually running the
            requested workflows. This is to just delete all the messages in a topic.
        worker_args: args appended to every entry this worker runs, overriding the
            entry's own, e.g. a smaller --window_size on GPUs with less memory.
    """
    if worker_args is None:
        worker_args = []

    # Callback to run the workflow indicated in the message.
    def process_message(json_data: dict[str, Any], extra_args: list[str]) -> None:
        logger.debug("worker received message %s", json_data)
        rslp_project = json_data["project"]
        rslp_workflow = json_data["workflow"]
        workflow_args = json_data["args"] + worker_args + extra_args
        run_workflow(rslp_project, rslp_workflow, workflow_args)

    def next_ready(prefetcher: _Prefetcher) -> _Claimed | None:
        """The next entry, or None once idle. Time spent prefetching is not idle."""
        idle_since = time.monotonic()
        while True:
            try:
                return prefetcher.ready.get(timeout=PREFETCH_POLL_SECONDS)
            except QueueEmpty:
                pass
            if not prefetcher.is_alive():
                try:
                    return prefetcher.ready.get_nowait()
                except QueueEmpty:
                    pass
                if prefetcher.error is not None:
                    raise RuntimeError("prefetch thread failed") from prefetcher.error
                return None
            if prefetcher.busy.is_set():
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= idle_timeout:
                return None

    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        queue = beaker.queue.get(queue_name)
        worker = beaker.queue.create_worker(queue)
        logger.info("listening for messages on %s", queue_name)

        _remove_stale_scratch_roots()
        scratch_root = tempfile.mkdtemp(prefix=PREFETCH_ROOT_PREFIX)
        consecutive_errors = 0
        try:
            with beaker.queue.worker_channel(queue, worker) as (tx, rx):
                prefetcher = _Prefetcher(rx, scratch_root, flush_messages, worker_args)
                prefetcher.start()
                try:
                    while True:
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
                        except Exception as e:
                            # TODO: does the entry go back in queue if we didn't reply to it with tx.send?
                            # This code below is pretty much copied from the Pub/Sub processing code.
                            # Maybe it's fine if it goes away to be honest, some tasks just don't work.
                            logger.error(
                                "encountered error while processing message %s: %s (%d/%d consecutive errors)",
                                entry_input,
                                e,
                                consecutive_errors,
                                retries,
                            )
                            consecutive_errors += 1
                            if consecutive_errors >= retries:
                                raise
                            time.sleep(retry_sleep)
                        finally:
                            if claimed.entry_dir is not None:
                                shutil.rmtree(claimed.entry_dir, ignore_errors=True)
                finally:
                    prefetcher.stop.set()
                    prefetcher.join(timeout=PREFETCH_JOIN_SECONDS)
        finally:
            shutil.rmtree(scratch_root, ignore_errors=True)


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
            env_vars = get_base_env_vars(use_weka_prefix=False)
            env_vars += extra_beaker_env_vars

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
                ],
                constraints=BeakerConstraints(
                    cluster=cluster,
                ),
                min_runtime=min_runtime,
                datasets=datasets,
                env_vars=env_vars,
                resources=BeakerTaskResources(
                    gpu_count=gpus, shared_memory=shared_memory
                ),
            )
            unique_id = str(uuid.uuid4())[0:8]
            beaker.experiment.create(name=f"worker_{unique_id}", spec=spec)


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
