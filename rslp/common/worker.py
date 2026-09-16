"""Worker to process jobs in a list of jobs."""

import json
import os
import shutil
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from queue import Empty as QueueEmpty
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

# Environment variable carrying a worker's own experiment name, set at launch. Beaker
# does not inject the experiment name, and the drain list names workers, so the worker
# has to be told who it is.
WORKER_NAME_ENV_VAR = "RSLP_WORKER_NAME"

# Ignore a drain list older than this. The supervisor rewrites the list every cycle, so
# a stale one means the supervisor is gone; without this the last list it wrote would
# keep retiring workers until the pool emptied.
DRAIN_STALE_SECONDS = 1800


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


def _release_on_termination(
    tx: Any, current: dict[str, str | None]
) -> Callable[[int, Any], None]:
    """Make a SIGTERM handler that hands the in-flight entry back to the queue.

    Beaker sends SIGTERM about five minutes before it kills a preempted job. Without
    this the entry stays CLAIMED and nothing may touch it until the claim goes stale,
    which is `claim_stale_seconds` later; rejecting it means the supervisor re-offers
    the job on its next cycle instead. The work itself is still lost, since a job is
    only marked complete once every window in it is written.

    Args:
        tx: the queue worker channel to send the rejection on.
        current: single-key dict holding the entry id being processed, or None.

    Returns:
        a handler to pass to signal.signal.
    """

    def handler(signo: int, stack_frame: Any) -> None:
        entry_id = current.get("entry_id")
        logger.error(
            "caught signal %d; releasing entry %s back to the queue", signo, entry_id
        )
        if entry_id is not None:
            try:
                tx.send(entry_id, rejection=f"worker terminated by signal {signo}")
            except Exception:
                # The job is going away regardless; the entry just goes stale instead.
                logger.exception("could not release entry %s", entry_id)
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


def worker_pipeline(
    queue_name: str,
    retries: int = 3,
    retry_sleep: int = 60,
    max_retry_sleep: int = 600,
    idle_timeout: int = 10,
    flush_messages: bool = False,
    drain_path: str | None = None,
) -> None:
    """Start a worker to run jobs from a Pub/Sub subscription.

    The job dict including rslp project, workflow, and arguments to pass must be
    written to the topic.

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

    def process_message(json_data: dict[str, Any]) -> None:
        logger.debug("worker received message %s", json_data)
        rslp_project = json_data["project"]
        rslp_workflow = json_data["workflow"]
        workflow_args = json_data["args"]
        run_workflow(rslp_project, rslp_workflow, workflow_args)

    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        queue = beaker.queue.get(queue_name)
        worker = beaker.queue.create_worker(queue)
        logger.info("listening for messages on %s", queue_name)

        consecutive_errors = 0
        watchdog: threading.Timer | None = None
        worker_name = os.environ.get(WORKER_NAME_ENV_VAR)
        with beaker.queue.worker_channel(queue, worker) as (tx, rx):
            in_flight: dict[str, str | None] = {"entry_id": None}
            signal.signal(signal.SIGTERM, _release_on_termination(tx, in_flight))
            try:
                while True:
                    # Before claiming, not after: holding no entry is what makes
                    # stopping free.
                    if drain_path is not None and _should_drain(
                        drain_path, worker_name
                    ):
                        logger.info(
                            "worker %s is on the drain list; exiting to hand back its "
                            "slot",
                            worker_name,
                        )
                        break

                    try:
                        batch = rx.rx.get(block=True, timeout=idle_timeout)
                    except QueueEmpty:
                        break

                    for worker_input in batch:
                        entry_id = worker_input.metadata.entry_id
                        entry_input = pb2_to_dict(worker_input.input)
                        in_flight["entry_id"] = entry_id
                        logger.info("processing entry %s", entry_id)

                        try:
                            if not flush_messages:
                                process_message(entry_input)
                            tx.send(entry_id, done=True)
                            in_flight["entry_id"] = None
                            consecutive_errors = 0
                        except Exception as e:
                            consecutive_errors += 1
                            logger.error(
                                "encountered error while processing message %s: %s (%d/%d consecutive errors)",
                                entry_input,
                                e,
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
                                    rejection=f"{type(e).__name__}: {e}"[
                                        :REJECTION_CHARS
                                    ],
                                )
                            except Exception:
                                # Not worth losing the run over: the entry just goes stale.
                                logger.exception("could not reject entry %s", entry_id)
                            if consecutive_errors >= retries:
                                raise
                            time.sleep(
                                min(
                                    retry_sleep * 2 ** (consecutive_errors - 1),
                                    max_retry_sleep,
                                )
                            )
            finally:
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

            datasets = [create_gcp_credentials_mount()]
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
) -> None:
    """Write tasks to the Beaker queue.

    Args:
        queue_name: the Beaker queue to write to.
        rslp_project: the rslp project to run.
        rslp_workflow: the workflow in the project to run.
        args_list: list of arguments fo reach task.
        expires_in_sec: how long until the queue entries should expire
    """
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        queue = beaker.queue.get(queue_name)

        for args in tqdm.tqdm(args_list, desc="Writing jobs to Beaker queue"):
            json_data = dict(
                project=rslp_project,
                workflow=rslp_workflow,
                args=args,
            )
            beaker.queue.create_entry_async(
                queue, input=json_data, expires_in_sec=expires_in_sec
            )
