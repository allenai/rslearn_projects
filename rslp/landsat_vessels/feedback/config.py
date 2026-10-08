"""Shared constants for the Landsat vessel feedback loop (see README.md)."""

from upath import UPath

# Skylight
SKYLIGHT_ENVIRONMENTS: dict[str, dict[str, str]] = {
    "integration": {
        "admin_feedback_url": "https://app-int.skylight.earth/admin?selected-tab=in-app-feedback",
        "graphql_url": "https://api-int.skylight.earth/graphql",
    },
    "production": {
        "admin_feedback_url": "https://app.skylight.earth/admin?selected-tab=in-app-feedback",
        "graphql_url": "https://api.skylight.earth/graphql",
    },
}
DEFAULT_ENVIRONMENT = "integration"

# Sat-service detection JSONs, at <bucket>/<SENSOR_DETECTIONS_SUBPATH>/YYYY/MM/DD/<scene>.json.
SKYLIGHT_DETECTION_BUCKETS: dict[str, str | None] = {
    "integration": "gs://skylight-data-sky-int-a-wxbc/sat-service",
    "production": None,
}
SENSOR_DETECTIONS_SUBPATH = "landsat_8_9/detections"

DEPLOYED_MODEL_VERSION = "landsat_vessels_v1.0.0"

# Feedback is only kept from these users (email suffix or exact address).
TRUSTED_DOMAINS = [
    "@allenai.org",
    "@mpi.govt.nz",
    "@marinemanagement.org.uk",
    "@c4ads.org",
]
TRUSTED_EXACT = [
    "loureirorius@gmail.com",
    "jess.williams@akashinga.org",
    "paigem.roberts@gmail.com",
    "namratakolla@yahoo.com",
    "gregg@exulans.net",
    "david.pearl@noaa.gov",
    "max.schofield@globalfishingwatch.org",
]

# Other values (UNSURE, empty) are dropped.
FEEDBACK_VALUE_TO_LABEL = {
    "GOOD": "correct",
    "BAD": "incorrect",
}

# Classifier dataset
DATASET_ROOT = UPath(
    "/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624"
)
LANDSAT_LAYER = "landsat"
LABEL_LAYER = "label"
WINDOW_SIZE = 64
WINDOW_RESOLUTION = 15
# Event timestamps are only minute-accurate.
TIME_BUFFER_MINUTES = 20

# Training and publishing. Paths are relative to the repo root.
DATA_DIR_REL = "data/landsat_vessels"
BASE_CLASSIFIER_CONFIG = f"{DATA_DIR_REL}/config_classifier_20260908.yaml"
CLASSIFIER_PROJECT_NAME = "landsat_vessel_classification_v2"
DEPLOYED_RUN_NAME = "olmoearth_base_layerdecay_20260908d"
RUN_NAME_STEM = "olmoearth_base_layerdecay"
DOCKERFILE_REL = "rslp/landsat_vessels/Dockerfile"
CONFIG_PY_REL = "rslp/landsat_vessels/config.py"


def gcs_checkpoint_url(project_name: str, run_name: str) -> str:
    """Public URL the Dockerfile downloads the classifier checkpoint from."""
    return (
        "https://storage.googleapis.com/ai2-rslearn-projects-data/projects/"
        f"{project_name}/{run_name}/best.ckpt"
    )


def gcs_checkpoint_uri(project_name: str, run_name: str) -> str:
    """gs:// URI the checkpoint is uploaded to."""
    return (
        f"gs://ai2-rslearn-projects-data/projects/{project_name}/{run_name}/best.ckpt"
    )
