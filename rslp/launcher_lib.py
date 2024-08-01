import os
import shutil
import tempfile

from google.cloud import storage
import yaml

BUCKET_NAME = "rslearn-data"
CODE_BLOB_PATH = "projects/{project_id}/{experiment_id}/code.zip"


def get_project_and_experiment(config_path: str) -> tuple[str, str]:
    with open(config_path) as f:
        data = yaml.safe_load(f)
    project_id = data["rslp_project"]
    experiment_id = data["rslp_experiment"]
    return project_id, experiment_id


def upload_code(project_id: str, experiment_id: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    with tempfile.TemporaryDirectory() as tmpdirname:
        print("creating archive of current code state")
        zip_fname = shutil.make_archive(os.path.join(tmpdirname, "archive"), "zip", root_dir=".")
        print("uploading archive")
        blob_path = CODE_BLOB_PATH.format(project_id=project_id, experiment_id=experiment_id)
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(zip_fname)
        print("upload complete")


def download_code(project_id: str, experiment_id: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)

    with tempfile.TemporaryDirectory() as tmpdirname:
        print("downloading code acrhive")
        blob_path = CODE_BLOB_PATH.format(project_id=project_id, experiment_id=experiment_id)
        blob = bucket.blob(blob_path)
        zip_fname = os.path.join(tmpdirname, "archive.zip")
        blob.download_to_filename(zip_fname)
        print("extracting archive")
        shutil.unpack_archive(zip_fname, ".", "zip")
        print("extraction complete", flush=True)
