"""Utility functions for launchers/entrypoints to call."""

import io
import os
import shutil
import tempfile

import yaml
from google.cloud import storage

CODE_BLOB_PATH = "projects/{project_id}/{experiment_id}/code.zip"
WANDB_ID_BLOB_PATH = "projects/{project_id}/{experiment_id}/wandb_id"

bucket = None


def _get_bucket() -> storage.Bucket:
    global bucket
    if bucket is None:
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.environ["RSLP_BUCKET"])
    return bucket


def get_project_and_experiment(config_path: str) -> tuple[str, str]:
    """Get the project and experiment IDs from the configuration file.

    Args:
        config_path: the configuration file.

    Returns:
        a tuple (project_id, experiment_id)
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)
    project_id = data["rslp_project"]
    experiment_id = data["rslp_experiment"]
    return project_id, experiment_id


def upload_code(project_id: str, experiment_id: str) -> None:
    """Upload code to GCS that entrypoint should retrieve.

    Called by the launcher.

    Args:
        project_id: the project ID.
        experiment_id: the experiment ID.
    """
    bucket = _get_bucket()
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("creating archive of current code state")
        zip_fname = shutil.make_archive(
            os.path.join(tmpdirname, "archive"), "zip", root_dir="."
        )
        print("uploading archive")
        blob_path = CODE_BLOB_PATH.format(
            project_id=project_id, experiment_id=experiment_id
        )
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(zip_fname)
        print("upload complete")


def download_code(project_id: str, experiment_id: str) -> None:
    """Download code from GCS for this experiment.

    Called by the entrypoint.

    Args:
        project_id: the project ID.
        experiment_id: the experiment ID.
    """
    bucket = _get_bucket()
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("downloading code acrhive")
        blob_path = CODE_BLOB_PATH.format(
            project_id=project_id, experiment_id=experiment_id
        )
        blob = bucket.blob(blob_path)
        zip_fname = os.path.join(tmpdirname, "archive.zip")
        blob.download_to_filename(zip_fname)
        print("extracting archive")
        shutil.unpack_archive(zip_fname, ".", "zip")
        print("extraction complete", flush=True)


def upload_wandb_id(project_id: str, experiment_id: str, wandb_id: str) -> None:
    """Save a W&B run ID to GCS.

    Args:
        project_id: the project ID.
        experiment_id: the experiment ID.
        wandb_id: the W&B run ID.
    """
    bucket = _get_bucket()
    blob_path = WANDB_ID_BLOB_PATH.format(
        project_id=project_id, experiment_id=experiment_id
    )
    blob = bucket.blob(blob_path)
    buf = io.BytesIO()
    buf.write(wandb_id.encode())
    buf.seek(0)
    blob.upload_from_file(buf)


def download_wandb_id(project_id: str, experiment_id: str) -> str | None:
    """Retrieve W&B run ID from GCS.

    Args:
        project_id: the project ID.
        experiment_id: the experiment ID.

    Returns:
        the W&B run ID, or None if it wasn't saved on GCS.
    """
    bucket = _get_bucket()
    blob_path = WANDB_ID_BLOB_PATH.format(
        project_id=project_id, experiment_id=experiment_id
    )
    blob = bucket.blob(blob_path)
    if not blob.exists():
        return None
    buf = io.BytesIO()
    blob.download_to_file(buf)
    return buf.getvalue().decode()
