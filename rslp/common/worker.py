"""Worker to process jobs in a list of jobs."""

import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent import futures
from typing import Any

import tqdm
from beaker import (
    Beaker,
    Constraints,
    DataMount,
    DataSource,
    ExperimentSpec,
    Priority,
    TaskResources,
)
from google.cloud import pubsub_v1

from rslp.launch_beaker import DEFAULT_WORKSPACE
from rslp.log_utils import get_logger
from rslp.main import run_workflow
from rslp.utils.beaker import DEFAULT_BUDGET, get_base_env_vars

logger = get_logger(__name__)

# Maximum expected duration of a job in hours. We use this to limit how long we care
# about a pending claim that hasn't completed yet.
MAX_JOB_HOURS = 4

# Scratch directory that jobs can use and it will be managed by this module.
SCRATCH_DIRECTORY = "/tmp/scratch"

# Directory to store SCRATCH_DIRECTORY (via symlink) in case
# manage_scratch_dir_on_data_disk is used.
# This is because some Beaker machines have much bigger /data disk than what's
# available for ephemeral storage within the Docker container, so we need to use that
# for disk-intensive tasks to avoid running out of disk space. But we also need to make
# sure we delete everything we wrote, so worker.py manages the folder.
DATA_DISK = "/data/rslearn_projects"


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
    project_id: str,
    subscription_id: str,
    retries: int = 3,
    retry_sleep: int = 60,
    idle_timeout: int = 10,
    manage_scratch_dir_on_data_disk: bool = False,
) -> None:
    """Start a worker to run jobs from a Pub/Sub subscription.

    The job dict including rslp project, workflow, and arguments to pass must be
    written to the topic.

    Args:
        project_id: the Google Cloud project ID.
        subscription_id: the Pub/Sub subscription ID.
        retries: retry for this many consecutive errors before terminating. A "retry"
            may run a different job than the one that originally caused failure. This
            ensures workers will complete most of the jobs before they terminate due to
            errors.
        retry_sleep: sleep for this many seconds between retries. Sleeping helps in
            case there is an error due to rate limiting.
        idle_timeout: seconds before we terminate if there is no activity.
        manage_scratch_dir_on_data_disk: whether to create SCRATCH_DIRECTORY on the
            /data disk and manage it to ensure it is deleted in case of SIGTERM.
    """
    if manage_scratch_dir_on_data_disk:
        # Some tasks use SCRATCH_DIRECTORY, and if management is enabled, it means we
        # should put the SCRATCH_DIRECTORY on the /data/ disk (via symlink), and that
        # we must ensure it is deleted in case SIGTERM is received (i.e. if the Beaker
        # job is cancelled or pre-empted.
        os.makedirs(DATA_DISK, exist_ok=True)
        tmp_dir_on_data_disk = tempfile.TemporaryDirectory(dir=DATA_DISK)
        os.symlink(tmp_dir_on_data_disk.name, SCRATCH_DIRECTORY)
        signal.signal(
            signal.SIGTERM, get_cleanup_signal_handler(tmp_dir_on_data_disk.name)
        )

    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(project_id, subscription_id)

    # Callback to run the workflow indicated in the message.
    def process_message(message: pubsub_v1.subscriber.message.Message) -> None:
        logger.debug("worker received message %s", message)
        json_data = json.loads(message.data.decode())
        rslp_project = json_data["project"]
        rslp_workflow = json_data["workflow"]
        workflow_args = json_data["args"]
        run_workflow(rslp_project, rslp_workflow, workflow_args)

    # Callback that wraps process_message to keep track of:
    # 1. Whether a message is currently being processed.
    # 2. The last time that a message finished processing.
    # 3. The number of consecutive errors. If there is an error in process_message, it
    #    will sleep for retry_sleep unless it exceeds retries in which case we exit.
    lock = threading.Lock()
    is_processing = False
    last_message_time = time.time()
    consecutive_errors = 0

    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        nonlocal is_processing, last_message_time, consecutive_errors
        try:
            with lock:
                is_processing = True

            process_message(message)
            message.ack()

            with lock:
                consecutive_errors = 0
        except Exception as e:
            logger.error(
                "encountered error while processing message %s: %s (%d/%d consecutive errors)",
                message,
                e,
                consecutive_errors,
                retries,
            )
            with lock:
                consecutive_errors += 1
            time.sleep(retry_sleep)
            # Pub/Sub will catch this error and print it so we just re-raise it here.
            # But in our monitoring loop below we will check for more errors than
            # retries and cancel the subscription if so.
            raise
        finally:
            with lock:
                is_processing = False
                last_message_time = time.time()

    # We limit to a single message at a time and a single worker since tasks should use
    # all of the available CPU/GPU resources.
    flow_control = pubsub_v1.types.FlowControl(
        max_messages=1,
        # Tasks may take several hours so we allow extending the lease for up to a day.
        max_lease_duration=24 * 3600,
    )
    executor = futures.ThreadPoolExecutor(max_workers=1)
    scheduler = pubsub_v1.subscriber.scheduler.ThreadScheduler(executor)
    streaming_pull_future = subscriber.subscribe(
        subscription_path,
        callback=callback,
        flow_control=flow_control,
        scheduler=scheduler,
    )
    logger.info("worker listening for messages on %s", subscription_path)
    # We use the loop below to make the worker exit if there are no more messages (by
    # guessing based on the provided idle timeout). We also need to exit if there have
    # been too many consecutive errors.
    try:
        while True:
            time.sleep(idle_timeout)

            with lock:
                if consecutive_errors > retries:
                    logger.info(
                        "worker exiting due to %d consecutive errors",
                        consecutive_errors,
                    )
                    break

                if is_processing:
                    logger.debug(
                        "worker continuing since a message is currently being processed"
                    )
                    continue

                time_since_last_activity = time.time() - last_message_time
                if time_since_last_activity < idle_timeout:
                    logger.debug(
                        "worker continuing since time since last activity %d is less than idle timeout %d",
                        time_since_last_activity,
                        idle_timeout,
                    )
                    continue

            logger.info("worker exiting due to idle timeout")
            break
    finally:
        # Exit the worker process.
        streaming_pull_future.cancel()
        streaming_pull_future.result()


def launch_workers(
    image_name: str,
    project_id: str,
    subscription_id: str,
    num_workers: int,
    gpus: int = 0,
    shared_memory: str | None = None,
    priority: Priority = Priority.low,
    cluster: list[str] = ["ai2/augusta-google-1"],
    manage_scratch_dir_on_data_disk: bool = False,
) -> None:
    """Start workers for the prediction jobs.

    Args:
        image_name: the Beaker image name to use for the jobs.
        project_id: the Google Cloud project ID.
        subscription_id: the Pub/Sub subscription ID.
        num_workers: number of workers to launch
        gpus: number of GPUs to request per worker.
        shared_memory: shared memory string like "256GiB".
        priority: priority to assign the Beaker jobs.
        cluster: clusters to target.
        manage_scratch_dir_on_data_disk: see worker_pipeline.
    """
    beaker = Beaker.from_env(default_workspace=DEFAULT_WORKSPACE)

    with beaker.session():
        for _ in tqdm.tqdm(range(num_workers)):
            env_vars = get_base_env_vars(use_weka_prefix=False)

            spec = ExperimentSpec.new(
                budget=DEFAULT_BUDGET,
                description="worker",
                beaker_image=image_name,
                priority=priority,
                command=["python", "-m", "rslp.main"],
                arguments=[
                    "common",
                    "worker",
                    project_id,
                    subscription_id,
                    "--manage_scratch_dir_on_data_disk",
                    str(manage_scratch_dir_on_data_disk),
                ],
                constraints=Constraints(
                    cluster=cluster,
                ),
                preemptible=True,
                datasets=[
                    DataMount(
                        source=DataSource(secret="RSLEARN_GCP_CREDENTIALS"),  # nosec
                        mount_path="/etc/credentials/gcp_credentials.json",  # nosec
                    ),
                    DataMount(
                        source=DataSource(host_path="/data"),
                        mount_path="/data",
                    ),
                ],
                env_vars=env_vars,
                resources=TaskResources(gpu_count=gpus, shared_memory=shared_memory),
            )
            unique_id = str(uuid.uuid4())[0:8]
            beaker.experiment.create(f"worker_{unique_id}", spec)


def write_jobs(
    project_id: str,
    topic_id: str,
    rslp_project: str,
    rslp_workflow: str,
    args_list: list[list[str]],
) -> None:
    """Write tasks to the PubSub topic.

    Args:
        project_id: the project ID for the PubSub topic.
        topic_id: the topic ID for the PubSub topic.
        rslp_project: the rslp project to run.
        rslp_workflow: the workflow in the project to run.
        args_list: list of arguments fo reach task.
    """
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    for args in tqdm.tqdm(args_list, desc="Writing jobs to Pub/Sub topic"):
        json_data = dict(
            project=rslp_project,
            workflow=rslp_workflow,
            args=args,
        )
        data = json.dumps(json_data).encode()
        publisher.publish(topic_path, data).result()
