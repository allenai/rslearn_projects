"""Worker to process jobs in a list of jobs."""

import shutil
import signal
import sys
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

# Minimum runtime to request for a worker, which is what makes its job *allocated*
# rather than unallocated. The scheduler treats anything at or under five minutes as
# unallocated, and unallocated jobs only run when no allocated job wants the slot, so a
# worker without this is preempted by allocated work whatever its priority. Set it to
# roughly one job: long enough to finish a unit of work, short enough to be placed
# quickly, since a shorter request fits the allocation grid sooner.
DEFAULT_WORKER_MIN_RUNTIME = timedelta(minutes=45)


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


def worker_pipeline(
    queue_name: str,
    retries: int = 3,
    retry_sleep: int = 60,
    max_retry_sleep: int = 600,
    idle_timeout: int = 10,
    flush_messages: bool = False,
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
        with beaker.queue.worker_channel(queue, worker) as (tx, rx):
            in_flight: dict[str, str | None] = {"entry_id": None}
            signal.signal(signal.SIGTERM, _release_on_termination(tx, in_flight))
            while True:
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
                                rejection=f"{type(e).__name__}: {e}"[:REJECTION_CHARS],
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
            env_vars = base_env_vars + extra_beaker_env_vars

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
            unique_id = str(uuid.uuid4())[0:8]
            beaker.experiment.create(name=f"{name_prefix}_{unique_id}", spec=spec)


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
