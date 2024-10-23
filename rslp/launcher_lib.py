"""Utility functions for launchers/entrypoints to call."""

import copy
import io
import os
import shutil
import tempfile
import zipfile
from itertools import product
from typing import Any

import yaml
from google.cloud import storage

CODE_BLOB_PATH = "projects/{project_id}/{experiment_id}/code.zip"
WANDB_ID_BLOB_PATH = "projects/{project_id}/{experiment_id}/wandb_id"
CODE_EXCLUDES = [".env", "wandb", "rslp/__pycache__"]

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


def make_archive(
    zip_filename: str, root_dir: str, exclude_prefixes: list[str] = []
) -> None:
    """Create a zip archive of the contents of root_dir.

    The paths in the zip archive will be relative to root_dir.

    This is similar to shutil.make_archive but it allows specifying a list of prefixes
    that should not be added to the zip archive.

    Args:
        zip_filename: the filename to save archive under.
        root_dir: the directory to create archive of.
        exclude_prefixes: a list of prefixes to exclude from the archive. If the
            relative path of a file from root_dir starts with one of the prefixes, then
            it will not be added to the resulting archive.
    """

    def should_exclude(rel_path: str) -> bool:
        for prefix in exclude_prefixes:
            if rel_path.startswith(prefix):
                return True
        return False

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(root_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, start=root_dir)
                if should_exclude(rel_path):
                    continue
                zipf.write(full_path, arcname=rel_path)


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
        zip_fname = os.path.join(tmpdirname, "archive.zip")
        make_archive(
            zip_fname,
            root_dir=".",
            exclude_prefixes=CODE_EXCLUDES,
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
        print("downloading code archive")
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


def extract_parameters(
    config: dict, path: list[str] | None = None
) -> list[tuple[list[str], list]]:
    """Recursively extract parameters that have list values.

    Args:
        config: the configuration dictionary.
        path: the current path in the configuration dictionary.

    Returns:
        a list of tuples: (path, list_of_values)
    """
    if path is None:
        path = []
    params = []
    for key, value in config.items():
        current_path = path + [key]
        if isinstance(value, dict):
            params.extend(extract_parameters(value, current_path))
        elif isinstance(value, list):
            params.append((current_path, value))
    return params


def set_in_dict(config: dict, path: list[str], value: Any) -> None:
    """Set a value in a nested configuration dictionary given a path.

    Args:
        config: the configuration dictionary to set the value in.
        path: the path to the value.
        value: the value to set.
    """
    for key in path[:-1]:
        config = config.setdefault(key, {})
    config[path[-1]] = value


def generate_combinations(base_config: dict, hparams_config: dict) -> list[dict]:
    """Generate all combinations of hyperparameters.

    Args:
        base_config: the base configuration dictionary.
        hparams_config: the hyperparameters configuration dictionary.

    Returns:
        a list of dictionaries, each represents a configuration with different hyperparameter values.
    """
    # Extract parameters with list values
    params = extract_parameters(hparams_config)
    if not params:
        return [base_config]
    # Generate all combinations of hyperparameters
    paths, lists = zip(*params)
    combinations = list(product(*lists))
    # Create a new config for each combination
    config_dicts = []
    for combo in combinations:
        new_config = copy.deepcopy(base_config)
        for path, value in zip(paths, combo):
            set_in_dict(new_config, path, value)
        config_dicts.append(new_config)

    return config_dicts


def create_custom_configs(
    config_path: str, hparams_config_path: str, custom_dir: str
) -> list[str]:
    """Create custom configs with different hyperparameter combinations.

    Args:
        config_path: the path to the base config.
        hparams_config_path: the path to the hyperparameters config.
        custom_dir: the directory to save the custom configs to.

    Returns:
        a list of paths to the custom configs.
    """
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    with open(hparams_config_path) as f:
        hparams_config = yaml.safe_load(f)
    custom_configs = generate_combinations(base_config, hparams_config)
    configs_paths = []
    for idx, config in enumerate(custom_configs):
        # Update experiment ID by appending the index as a suffix
        # Not sure if it's better to add the hyperparameters to the experiment ID
        custom_experiment_id = f"{config['rslp_experiment']}_{idx}"
        config["rslp_experiment"] = custom_experiment_id
        config_filename = os.path.join(custom_dir, f"{custom_experiment_id}.yaml")
        with open(config_filename, "w") as f:
            yaml.dump(config, f)
        configs_paths.append(config_filename)
    return configs_paths
