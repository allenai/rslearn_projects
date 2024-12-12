"""Worker to process jobs in a list of jobs."""

import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

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
from google.cloud import pubsub_v1, storage

from rslp.launch_beaker import BUDGET, DEFAULT_WORKSPACE, IMAGE_NAME, get_base_env_vars
from rslp.log_utils import get_logger
from rslp.main import run_workflow

logger = get_logger(__name__)

# Maximum expected duration of a job in hours. We use this to limit how long we care
# about a pending claim that hasn't completed yet.
MAX_JOB_HOURS = 4


def _get_pending_jobs(
    jobs: list[list[str]], claim_bucket: storage.Bucket, claim_dir: str
) -> list[int]:
    """Get the indices of jobs that haven't been claimed yet.

    Args:
        jobs: the full list of jobs.
        claim_bucket: bucket where files indicating completed jobs are written.
        claim_dir: path within bucket.
    """
    claimed = set()
    # Pending claims are only valid for a few hours.
    for blob in claim_bucket.list_blobs(prefix=f"{claim_dir}pending/"):
        if datetime.now(timezone.utc) - blob.time_created > timedelta(
            hours=MAX_JOB_HOURS
        ):
            # This is a stale pending claim (the job may have completed, but if so we
            # will see its completed blob below).
            continue
        claimed.add(int(blob.name.split("/")[-1]))
    # While completed files indicate that the job is done permanently.
    for blob in claim_bucket.list_blobs(prefix=f"{claim_dir}completed/"):
        claimed.add(int(blob.name.split("/")[-1]))

    pending = []
    for idx in range(len(jobs)):
        if idx in claimed:
            continue
        pending.append(idx)

    return pending


def worker_pipeline(
    project_id: str,
    subscription_id: str,
    retries: int = 3,
    retry_sleep: int = 60,
    idle_timeout: int = 10,
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
    """
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

    flow_control = pubsub_v1.types.FlowControl(
        max_messages=1,
    )
    streaming_pull_future = subscriber.subscribe(
        subscription_path, callback=callback, flow_control=flow_control
    )
    logger.info("worker listening for messages on %s", subscription_path)
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
        streaming_pull_future.cancel()
        streaming_pull_future.result()


def launch_worker(project_id: str, subscription_id: str) -> None:
    """Launch a worker job.

    Args:
        project_id: the Google Cloud project ID.
        subscription_id: the Pub/Sub subscription ID.
    """


def launch_workers(
    project_id: str,
    subscription_id: str,
    num_workers: int,
    gpus: int = 0,
    shared_memory: str | None = None,
    priority: Priority = Priority.low,
    cluster: list[str] = ["ai2/augusta-google-1"],
) -> None:
    """Start workers for the prediction jobs.

    Args:
        project_id: the Google Cloud project ID.
        subscription_id: the Pub/Sub subscription ID.
        num_workers: number of workers to launch
        gpus: number of GPUs to request per worker.
        shared_memory: shared memory string like "256GiB".
        priority: priority to assign the Beaker jobs.
        cluster: clusters to target.
    """
    beaker = Beaker.from_env(default_workspace=DEFAULT_WORKSPACE)

    with beaker.session():
        for _ in tqdm.tqdm(range(num_workers)):
            env_vars = get_base_env_vars(use_weka_prefix=False)

            spec = ExperimentSpec.new(
                budget=BUDGET,
                description="worker",
                beaker_image=IMAGE_NAME,
                priority=priority,
                command=["python", "-m", "rslp.main"],
                arguments=[
                    "common",
                    "worker",
                    project_id,
                    subscription_id,
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
                ],
                env_vars=env_vars,
                resources=TaskResources(gpu_count=gpus, shared_memory=shared_memory),
            )
            unique_id = str(uuid.uuid4())[0:8]
            beaker.experiment.create(f"worker_{unique_id}", spec)
