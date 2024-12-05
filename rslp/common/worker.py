"""Worker to process jobs in a list of jobs."""

import json
import random
from datetime import datetime, timedelta, timezone

from google.api_core.exceptions import PreconditionFailed
from google.cloud import storage
from upath import UPath

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
    project: str,
    workflow: str,
    job_fname: str,
    claim_bucket_name: str,
    claim_dir: str,
) -> None:
    """Start a worker to run the specified jobs.

    Args:
        project: the project that the workflow to run is in.
        workflow: the workflow to run.
        job_fname: file containing the full list of jobs (arguments to the workflow
            function) that need to be run.
        claim_bucket_name: the GCS bucket to use for claiming jobs.
        claim_dir: the path within claim_bucket_name to use for claiming jobs.
    """
    job_upath = UPath(job_fname)
    client = storage.Client()
    claim_bucket = client.bucket(claim_bucket_name)

    with job_upath.open("r") as f:
        jobs: list[list[str]] = json.load(f)

    # Get the currently pending jobs.
    # Our strategy will be to sample a job and attempt to claim it.
    # And then if the claim fails then we refresh the pending jobs.
    # This works for up to ~10000 jobs.
    pending = _get_pending_jobs(jobs, claim_bucket, claim_dir)

    while len(pending) > 0:
        job_idx = random.choice(pending)
        pending.remove(job_idx)
        pending_blob = claim_bucket.blob(f"{claim_dir}pending/{job_idx}")
        completed_blob = claim_bucket.blob(f"{claim_dir}completed/{job_idx}")

        # Determine the generation of pending_blob so we can create a newer one if
        # applicable. If it doesn't exist, we use 0 so that it will throw error if the
        # file exists at all (the actual generation should never be 0).
        pending_blob_generation = 0
        is_pending = False
        if pending_blob.exists():
            pending_blob.reload()
            pending_blob_generation = pending_blob.generation
            if datetime.now(timezone.utc) - pending_blob.time_created < timedelta(
                hours=MAX_JOB_HOURS
            ):
                is_pending = True

        if is_pending or completed_blob.exists():
            pending = _get_pending_jobs(jobs, claim_bucket, claim_dir)
            continue

        try:
            # Use generation so that it throws error if generation doesn't match.
            pending_blob.upload_from_string(
                "", if_generation_match=pending_blob_generation
            )
        except PreconditionFailed:
            # This means another worker claimed the job in between when we confirmed
            # the blob doesn't exist already and when we tried to claim it. In this
            # case we just try again.
            continue

        logger.info("claimed job %d and running it now", job_idx)
        run_workflow(project, workflow, jobs[job_idx])

        completed_blob.upload_from_string("")
