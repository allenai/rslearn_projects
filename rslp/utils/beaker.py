"""Utilities relating to Beaker jobs."""

import os

from beaker import DataMount, DataSource, EnvVar, ImageSource
from beaker.client import Beaker

DEFAULT_WORKSPACE = "ai2/earth-systems"
DEFAULT_BUDGET = "ai2/d5"


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
            value="earthsystem-dev-c3po",  # nosec
        ),
        EnvVar(
            name="GOOGLE_CLOUD_PROJECT",  # nosec
            value="earthsystem-dev-c3po",  # nosec
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


def create_gcp_credentials_mount(
    secret: str = "RSLEARN_GCP_CREDENTIALS",
    mount_path: str = "/etc/credentials/gcp_credentials.json",
) -> DataMount:
    """Create a mount for the GCP credentials.

    Args:
        secret: the beaker secret containing the GCP credentials.
        mount_path: the path to mount the GCP credentials to.

    Returns:
        DataMount: A Beaker DataMount object that can be used in an experiment specification.
    """
    return DataMount(
        source=DataSource(secret=secret),  # nosec
        mount_path=mount_path,  # nosec
    )
