"""Worker to process jobs in a list of jobs."""

import shutil
import sys
import time
import uuid
from collections.abc import Callable
from queue import Empty as QueueEmpty
from typing import Any

import tqdm
from beaker import (
    Beaker,
    BeakerConstraints,
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


def worker_pipeline(
    queue_name: str,
    retries: int = 3,
    retry_sleep: int = 60,
    idle_timeout: int = 10,
    flush_messages: bool = False,
) -> None:
    """Start a worker to run jobs from a Pub/Sub subscription.

    The job dict including rslp project, workflow, and arguments to pass must be
    written to the topic.

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
    """

    # Callback to run the workflow indicated in the message.
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
            while True:
                try:
                    batch = rx.rx.get(block=True, timeout=idle_timeout)
                except QueueEmpty:
                    break

                for worker_input in batch:
                    entry_id = worker_input.metadata.entry_id
                    entry_input = pb2_to_dict(worker_input.input)
                    logger.info("processing entry %s", entry_id)

                    try:
                        if not flush_messages:
                            process_message(entry_input)
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


def launch_workers(
    image_name: str,
    queue_name: str,
    num_workers: int,
    cluster: list[str],
    gpus: int = 0,
    shared_memory: str | None = None,
    priority: BeakerJobPriority = BeakerJobPriority.low,
    weka_mounts: list[WekaMount] = [],
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
    """
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        for _ in tqdm.tqdm(range(num_workers)):
            env_vars = get_base_env_vars(use_weka_prefix=False)

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
                ],
                constraints=BeakerConstraints(
                    cluster=cluster,
                ),
                preemptible=True,
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
