"""Utilities relating to Beaker jobs."""

import os
from dataclasses import dataclass
from functools import cache

from beaker import BeakerDataMount, BeakerDataSource, BeakerEnvVar, BeakerImageSource
from beaker.client import Beaker
from beaker.exceptions import BeakerSecretNotFound

from rslp.log_utils import get_logger

logger = get_logger(__name__)

DEFAULT_WORKSPACE = "ai2/earth-systems"
DEFAULT_BUDGET = "ai2/atec-olmoearth"

# Fallback secret holding a Beaker token shared across the project.
SHARED_BEAKER_TOKEN_SECRET = "RSLP_BEAKER_TOKEN"  # nosec


# Cached because callers invoke this once per worker inside a launch loop, and neither
# the username nor the secret's existence changes within a process. Without this, a
# 128-worker launch would open 128 Beaker clients to ask the same question.
@cache
def resolve_beaker_token_secret(workspace: str = DEFAULT_WORKSPACE) -> str:
    """Choose which Beaker secret to mount as BEAKER_TOKEN.

    Beaker attributes anything a job creates to the owner of the token that job carries.
    A launcher that mounts one shared token therefore produces jobs owned by whoever owns
    that token, not by the person who started the run. That is wrong for attribution and
    for quota, and it means you cannot cancel your own jobs. It matters most for
    ``supervise``, which launches further jobs from inside a job.

    Prefers ``<username>_BEAKER_TOKEN``, the convention already used across this
    workspace and in olmoearth_pretrain. Falls back to the shared secret when the caller
    has not written their own, so this cannot break a launcher that worked before; write
    yours with::

        beaker secret write "$(beaker account whoami --format json | jq -r '.[0].name')_BEAKER_TOKEN" <token>

    Args:
        workspace: the workspace whose secrets to look in.

    Returns:
        the name of the Beaker secret to mount as BEAKER_TOKEN.
    """
    try:
        with Beaker.from_env(default_workspace=workspace) as beaker:
            username = beaker.user_name
            if not username:
                logger.warning(
                    "could not determine the Beaker username; mounting %s, so jobs this "
                    "run creates will be attributed to that token's owner",
                    SHARED_BEAKER_TOKEN_SECRET,
                )
                return SHARED_BEAKER_TOKEN_SECRET
            name = f"{username}_BEAKER_TOKEN"
            try:
                beaker.secret.get(name)
            except BeakerSecretNotFound:
                logger.warning(
                    "no %s secret in %s, so mounting %s instead; jobs this run creates "
                    "will be attributed to that token's owner rather than to %s. Write "
                    "your own with: beaker secret write %s <token>",
                    name,
                    workspace,
                    SHARED_BEAKER_TOKEN_SECRET,
                    username,
                    name,
                )
                return SHARED_BEAKER_TOKEN_SECRET
            logger.info("mounting %s as BEAKER_TOKEN", name)
            return name
    except Exception:
        # Never let attribution break a launch: fall back to the behaviour that has
        # always worked, and say why.
        logger.warning(
            "could not resolve a per-user Beaker token secret; falling back to %s",
            SHARED_BEAKER_TOKEN_SECRET,
            exc_info=True,
        )
        return SHARED_BEAKER_TOKEN_SECRET


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


def get_base_env_vars(
    use_weka_prefix: bool = False, token_secret: str | None = None
) -> list[BeakerEnvVar]:
    """Get basic environment variables that should be common across all Beaker jobs.

    Args:
        use_weka_prefix: set RSLP_PREFIX to RSLP_WEKA_PREFIX which should be set up to
            point to Weka. Otherwise it is set to RSLP_PREFIX which could be GCS or
            Weka.
        token_secret: the Beaker secret to mount as BEAKER_TOKEN. Defaults to the
            launching user's own secret, falling back to the shared one; see
            :func:`resolve_beaker_token_secret`. Pass a name explicitly to pin it.
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
            secret=token_secret or resolve_beaker_token_secret(),  # nosec
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


def create_gee_credentials_mount(
    secret: str | None = None,
    mount_path: str = "/etc/credentials/gee_credentials.json",
) -> BeakerDataMount:
    """Create a mount for the Google Earth Engine credentials.

    If the secret is not specified, it defaults to "GCP_HELIOS_SERVICE_ACCOUNT", unless
    the GEE_CREDENTIALS_MOUNT_SECRET environment variable is set.
    """
    if secret is None:
        secret = os.environ.get(
            "GEE_CREDENTIALS_MOUNT_SECRET", "GCP_HELIOS_SERVICE_ACCOUNT"
        )
    return create_gcp_credentials_mount(secret, mount_path)
