"""Utilities relating to Beaker jobs."""

import os
from dataclasses import dataclass

from beaker import BeakerDataMount, BeakerDataSource, BeakerEnvVar, BeakerImageSource
from beaker.client import Beaker

DEFAULT_WORKSPACE = "ai2/earth-systems"
DEFAULT_BUDGET = "ai2/d5"


@dataclass
class WekaMount:
    """Specification of a Weka mount within a Beaker job."""

    bucket_name: str
    mount_path: str
    sub_path: str | None = None

    def to_data_mount(self) -> BeakerDataMount:
        """Convert this WekaMount to a Beaker DataMount object."""
        return BeakerDataMount(
            source=BeakerDataSource(weka=self.bucket_name),
            mount_path=self.mount_path,
            sub_path=self.sub_path,
        )


def get_base_env_vars(use_weka_prefix: bool = False) -> list[BeakerEnvVar]:
    """Get basic environment variables that should be common across all Beaker jobs.

    Args:
        use_weka_prefix: set RSLP_PREFIX to RSLP_WEKA_PREFIX which should be set up to
            point to Weka. Otherwise it is set to RSLP_PREFIX which could be GCS or
            Weka.
    """
    env_vars = [
        BeakerEnvVar(
            name="WANDB_API_KEY",  # nosec
            secret="RSLEARN_WANDB_API_KEY",  # nosec
        ),
        BeakerEnvVar(
            name="GOOGLE_APPLICATION_CREDENTIALS",  # nosec
            value="/etc/credentials/gcp_credentials.json",  # nosec
        ),
        BeakerEnvVar(
            name="GCLOUD_PROJECT",  # nosec
            value="earthsystem-dev-c3po",  # nosec
        ),
        BeakerEnvVar(
            name="GOOGLE_CLOUD_PROJECT",  # nosec
            value="earthsystem-dev-c3po",  # nosec
        ),
        BeakerEnvVar(
            name="WEKA_ACCESS_KEY_ID",  # nosec
            secret="RSLEARN_WEKA_KEY",  # nosec
        ),
        BeakerEnvVar(
            name="WEKA_SECRET_ACCESS_KEY",  # nosec
            secret="RSLEARN_WEKA_SECRET",  # nosec
        ),
        BeakerEnvVar(
            name="WEKA_ENDPOINT_URL",  # nosec
            value="https://weka-aus.beaker.org:9000",  # nosec
        ),
        BeakerEnvVar(
            name="MKL_THREADING_LAYER",
            value="GNU",
        ),
        BeakerEnvVar(
            name="BEAKER_TOKEN",  # nosec
            secret="RSLP_BEAKER_TOKEN",  # nosec
        ),
    ]

    if use_weka_prefix:
        env_vars.append(
            BeakerEnvVar(
                name="RSLP_PREFIX",
                value=os.environ["RSLP_WEKA_PREFIX"],
            )
        )
    else:
        env_vars.append(
            BeakerEnvVar(
                name="RSLP_PREFIX",
                value=os.environ["RSLP_PREFIX"],
            )
        )
    return env_vars


def upload_image(
    image_name: str, workspace: str, beaker_client: Beaker
) -> BeakerImageSource:
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
    image_source = BeakerImageSource(beaker=image.full_name)
    return image_source


def create_gcp_credentials_mount(
    secret: str = "RSLEARN_GCP_CREDENTIALS",
    mount_path: str = "/etc/credentials/gcp_credentials.json",
) -> BeakerDataMount:
    """Create a mount for the GCP credentials.

    Args:
        secret: the beaker secret containing the GCP credentials.
        mount_path: the path to mount the GCP credentials to.

    Returns:
        DataMount: A Beaker DataMount object that can be used in an experiment specification.
    """
    return BeakerDataMount(
        source=BeakerDataSource(secret=secret),  # nosec
        mount_path=mount_path,  # nosec
    )
