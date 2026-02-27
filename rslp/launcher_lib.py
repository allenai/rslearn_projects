"""Utility functions for training job launcher."""

import copy
import os
import shutil
import tempfile
import zipfile
from itertools import product
from typing import Any

import yaml
from upath import UPath

from rslp.log_utils import get_logger

CODE_BLOB_PATH = "projects/{project_id}/{experiment_id}/code.zip"
CODE_EXCLUDES = [
    ".git",
    "rslp/__pycache__",
    ".env",
    ".mypy_cache",
    "lightning_logs",
    "test_data",
    "wandb",
    "project_data",
    ".venv",
]

logger = get_logger(__name__)


def get_project_and_experiment(config_path: str) -> tuple[str, str]:
    """Get the project and experiment IDs from the configuration file.

    Args:
        config_path: the configuration file.

    Returns:
        a tuple (project_id, experiment_id)
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)
    project_id = data["project_name"]
    experiment_id = data["run_name"]
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
                if os.path.exists(full_path):
                    zipf.write(full_path, arcname=rel_path)


def upload_code(project_id: str, experiment_id: str) -> None:
    """Upload code to RSLP_PREFIX that entrypoint should retrieve.

    Called by the launcher.

    Args:
        project_id: the project ID.
        experiment_id: the experiment ID.
    """
    rslp_prefix = UPath(os.environ["RSLP_PREFIX"])
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.info("creating archive of current code state")
        zip_fname = os.path.join(tmpdirname, "archive.zip")
        make_archive(
            zip_fname,
            root_dir=".",
            exclude_prefixes=CODE_EXCLUDES,
        )
        logger.info("uploading archive")
        project_code_fname = rslp_prefix / CODE_BLOB_PATH.format(
            project_id=project_id, experiment_id=experiment_id
        )
        project_code_fname.parent.mkdir(parents=True, exist_ok=True)
        with open(zip_fname, "rb") as src:
            with project_code_fname.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        logger.info("upload complete")


def download_code(project_id: str, experiment_id: str) -> None:
    """Download code from RSLP_PREFIX for this experiment.

    Called by the entrypoint.

    Args:
        project_id: the project ID.
        experiment_id: the experiment ID.
    """
    rslp_prefix = UPath(os.environ["RSLP_PREFIX"])
    with tempfile.TemporaryDirectory() as tmpdirname:
        logger.info("downloading code archive")
        project_code_fname = rslp_prefix / CODE_BLOB_PATH.format(
            project_id=project_id, experiment_id=experiment_id
        )
        zip_fname = os.path.join(tmpdirname, "archive.zip")
        with project_code_fname.open("rb") as src:
            with open(zip_fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
        logger.info("extracting archive")
        shutil.unpack_archive(zip_fname, ".", "zip")
        logger.info("extraction complete")


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
) -> dict[str, list[str]]:
    """Create custom configs with different hyperparameter combinations.

    Args:
        config_path: the path to the base config.
        hparams_config_path: the path to the hyperparameters config.
        custom_dir: the directory to save the custom configs to.

    Returns:
        a dictionary mapping run IDs to paths to the custom configs.
    """
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    with open(hparams_config_path) as f:
        hparams_config = yaml.safe_load(f)
    custom_configs = generate_combinations(base_config, hparams_config)
    configs_paths = {}
    for idx, config in enumerate(custom_configs):
        # Not sure if it's better to add the hyperparameters to the filename
        experiment_id = base_config["run_name"]
        config_filename = os.path.join(custom_dir, f"{experiment_id}_{idx}.yaml")
        with open(config_filename, "w") as f:
            yaml.dump(config, f)
        configs_paths[f"run_{idx}"] = [config_filename]
    return configs_paths
