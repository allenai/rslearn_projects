"""Utility functions for launchers/entrypoints to call."""

import copy
import os
import shutil
import tempfile
import zipfile
from itertools import product
from typing import Any

import yaml
from beaker import DataMount, DataSource, EnvVar, ImageSource
from beaker.client import Beaker
from upath import UPath

CODE_BLOB_PATH = "projects/{project_id}/{experiment_id}/code.zip"
WANDB_ID_BLOB_PATH = "projects/{project_id}/{experiment_id}/{run_id}wandb_id"
CODE_EXCLUDES = [
    ".git",
    "rslp/__pycache__",
    ".env",
    ".mypy_cache",
    "lightning_logs",
    "wandb",
]


def get_base_env_vars(use_weka_prefix: bool = False) -> list[EnvVar]:
    """Get basic environment variables that should be common across all Beaker jobs.

    Args:
        use_weka_prefix: set RSLP_PREFIX to RSLP_WEKA_PREFIX which should be set up to
            point to Weka. Otherwise it is set to RSLP_PREFIX which could be GCS or
            Weka.
    """
    env_vars = [
        EnvVar(
            name="WANDB_API_KEY",  # nosec
            secret="RSLEARN_WANDB_API_KEY",  # nosec
        ),
        EnvVar(
            name="GOOGLE_APPLICATION_CREDENTIALS",  # nosec
            value="/etc/credentials/gcp_credentials.json",  # nosec
        ),
        EnvVar(
            name="GCLOUD_PROJECT",  # nosec
            value="prior-satlas",  # nosec
        ),
        EnvVar(
            name="WEKA_ACCESS_KEY_ID",  # nosec
            secret="RSLEARN_WEKA_KEY",  # nosec
        ),
        EnvVar(
            name="WEKA_SECRET_ACCESS_KEY",  # nosec
            secret="RSLEARN_WEKA_SECRET",  # nosec
        ),
        EnvVar(
            name="WEKA_ENDPOINT_URL",  # nosec
            value="https://weka-aus.beaker.org:9000",  # nosec
        ),
        EnvVar(
            name="MKL_THREADING_LAYER",
            value="GNU",
        ),
    ]

    if use_weka_prefix:
        env_vars.append(
            EnvVar(
                name="RSLP_PREFIX",
                value=os.environ["RSLP_WEKA_PREFIX"],
            )
        )
    else:
        env_vars.append(
            EnvVar(
                name="RSLP_PREFIX",
                value=os.environ["RSLP_PREFIX"],
            )
        )
    return env_vars


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
    rslp_prefix = UPath(os.environ["RSLP_PREFIX"])
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
        with open(zip_fname, "rb") as src:
            with (rslp_prefix / blob_path).open("wb") as dst:
                shutil.copyfileobj(src, dst)
        print("upload complete")


def download_code(project_id: str, experiment_id: str) -> None:
    """Download code from GCS for this experiment.

    Called by the entrypoint.

    Args:
        project_id: the project ID.
        experiment_id: the experiment ID.
    """
    rslp_prefix = UPath(os.environ["RSLP_PREFIX"])
    with tempfile.TemporaryDirectory() as tmpdirname:
        print("downloading code archive")
        blob_path = CODE_BLOB_PATH.format(
            project_id=project_id, experiment_id=experiment_id
        )
        zip_fname = os.path.join(tmpdirname, "archive.zip")
        with (rslp_prefix / blob_path).open("rb") as src:
            with open(zip_fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
        print("extracting archive")
        shutil.unpack_archive(zip_fname, ".", "zip")
        print("extraction complete", flush=True)


def upload_image(image_name: str, workspace: str, beaker_client: Beaker) -> ImageSource:
    """Upload an image to Beaker.

    This function handles uploading a Docker image to Beaker's image registry. It creates
    a new image entry in the specified Beaker workspace and returns an ImageSource that
    can be used to reference this image in Beaker experiments.

    The image must already exist locally in the Docker daemon. The image_name parameter
    should match the name of the local Docker image.

    Args:
        image_name: The name of the local Docker image to upload. This should be in the
            format "repository/image:tag" or just "image:tag".
        workspace: The Beaker workspace where the image should be uploaded. The workspace
            must already exist and the authenticated user must have write permissions.
        beaker_client: An authenticated Beaker client instance that will be used to
            make the API calls.

    Returns:
        ImageSource: A Beaker ImageSource object containing the full Beaker image name.
            This can be used as a source in experiment specifications.

    Example:
        >>> client = Beaker(token="...")
        >>> image_source = upload_image("myimage:latest", "my-workspace", client)
        >>> print(image_source.beaker)
        'beaker://my-workspace/myimage'
    """
    image = beaker_client.image.create(image_name, image_name, workspace=workspace)
    image_source = ImageSource(beaker=image.full_name)
    return image_source


def create_gcp_credentials_mount() -> DataMount:
    """Create a mount for the GCP credentials."""
    return DataMount(
        source=DataSource(secret="RSLEARN_GCP_CREDENTIALS"),  # nosec
        mount_path="/etc/credentials/gcp_credentials.json",  # nosec
    )


def upload_wandb_id(
    project_id: str, experiment_id: str, run_id: str | None, wandb_id: str
) -> None:
    """Save a W&B run ID to GCS.

    Args:
        project_id: the project ID.
        experiment_id: the experiment ID.
        run_id: optional run ID (for hyperparameter experiments)
        wandb_id: the W&B run ID.
    """
    rslp_prefix = UPath(os.environ["RSLP_PREFIX"])
    run_id_path = f"{run_id}/" if run_id else ""
    blob_path = WANDB_ID_BLOB_PATH.format(
        project_id=project_id, experiment_id=experiment_id, run_id=run_id_path
    )
    with (rslp_prefix / blob_path).open("w") as f:
        f.write(wandb_id)


def download_wandb_id(
    project_id: str, experiment_id: str, run_id: str | None
) -> str | None:
    """Retrieve W&B run ID from GCS.

    Args:
        project_id: the project ID.
        experiment_id: the experiment ID.
        run_id: the run ID (for hyperparameter experiments)

    Returns:
        the W&B run ID, or None if it wasn't saved on GCS.
    """
    rslp_prefix = UPath(os.environ["RSLP_PREFIX"])
    run_id_path = f"{run_id}/" if run_id else ""
    blob_path = WANDB_ID_BLOB_PATH.format(
        project_id=project_id, experiment_id=experiment_id, run_id=run_id_path
    )
    fname = rslp_prefix / blob_path
    if not fname.exists():
        return None
    with fname.open() as f:
        return f.read().strip()


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
) -> dict[str, str]:
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
        experiment_id = base_config["rslp_experiment"]
        config_filename = os.path.join(custom_dir, f"{experiment_id}_{idx}.yaml")
        with open(config_filename, "w") as f:
            yaml.dump(config, f)
        configs_paths[f"run_{idx}"] = config_filename
    return configs_paths
