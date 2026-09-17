"""Shared configuration for the Landsat vessel feedback loop.

The feedback loop turns false positives flagged in the Skylight platform into new
training samples for the classifier, retrains, and republishes the served Docker image.
Every stage (``pull`` -> ``create_windows`` -> ``add_to_training`` -> train ->
``publish``) reads its constants from here so the pieces stay in sync.

See ``rslp/landsat_vessels/feedback/README.md`` for the end-to-end workflow.
"""

from upath import UPath

# ---------------------------------------------------------------------------
# Skylight platform
# ---------------------------------------------------------------------------
# The v1.0.0 model is deployed in the integration environment; production is kept here
# too so the same tooling can pull either. ``pull.py`` selects one with --environment.
SKYLIGHT_ENVIRONMENTS: dict[str, dict[str, str]] = {
    "integration": {
        "app_url": "https://app-int.skylight.earth",
        "admin_feedback_url": "https://app-int.skylight.earth/admin?selected-tab=in-app-feedback",
        "graphql_url": "https://api-int.skylight.earth/graphql",
    },
    "production": {
        "app_url": "https://app.skylight.earth",
        "admin_feedback_url": "https://app.skylight.earth/admin?selected-tab=in-app-feedback",
        "graphql_url": "https://api.skylight.earth/graphql",
    },
}
DEFAULT_ENVIRONMENT = "integration"

# The Skylight sat-service detection outputs on GCS. Each scene is one JSON at
# ``<bucket>/<sensor>/detections/<YYYY>/<MM>/<DD>/<scene_id>.json`` whose ``detections``
# list carries ``longitude``/``latitude``/``score`` and a crop index (the ``<n>_rgb.png``
# in ``crop_fnames.rgb``) that matches the trailing index of the feedback ``event_id``.
# This is the authoritative source of detection coordinates and is used to enrich feedback
# offline -- the Skylight GraphQL API is IP-restricted and unreachable from compute.
SKYLIGHT_DETECTION_BUCKETS: dict[str, str | None] = {
    "integration": "gs://skylight-data-sky-int-a-wxbc/sat-service",
    "production": None,  # fill in when the production detections bucket is known
}
SENSOR_DETECTIONS_SUBPATH = "landsat_8_9/detections"

# The v1.0.0 model deployed to integration, recorded on every feedback label so a window
# always knows which model's mistake it is correcting.
DEPLOYED_MODEL_VERSION = "landsat_vessels_v1.0.0"

# Only feedback from these users is trusted enough to train on. Domains match on the
# email suffix; exacts match the whole address. Mirrors the round-0 allowlist in
# /weka/dfive-default/yawenz/landsat/README.md.
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

# In-app feedback values we keep. GOOD -> the detection was a real vessel (positive);
# BAD -> a false positive (the hard negative this loop is built to collect). Anything
# else (UNSURE, empty) is dropped.
FEEDBACK_VALUE_TO_LABEL = {
    "GOOD": "correct",
    "BAD": "incorrect",
}

# ---------------------------------------------------------------------------
# rslearn dataset (classifier)
# ---------------------------------------------------------------------------
# The classifier's dataset. New feedback windows are written as a dated group inside it,
# alongside selected_copy / phase2a_completed / round1_20260803 / feedback_20260325.
DATASET_ROOT = UPath(
    "/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624"
)
LANDSAT_LAYER = "landsat"
LABEL_LAYER = "label"

# Window geometry. 64 px @ 15 m matches the original feedback_20260325 group; the
# classifier reads the centre 32 px via CenterCrop at train time, so 64 px is ample.
# (round1_20260803 uses 512 px only because it doubles as an annotation-context view;
# feedback labels arrive already decided in Skylight, so no context view is needed.)
WINDOW_SIZE = 64
WINDOW_RESOLUTION = 15
# Widen the item-match time range around the reported event so a re-prepare still finds
# the scene even when the timestamp is only minute-accurate.
TIME_BUFFER_MINUTES = 20

# ---------------------------------------------------------------------------
# Training / publishing
# ---------------------------------------------------------------------------
# The deployed classifier config that new dated configs are derived from, and its
# project/run coordinates. Paths are relative to the repo root (two levels above the
# landsat_vessels package), matching how rslp is normally invoked.
DATA_DIR_REL = "data/landsat_vessels"
BASE_CLASSIFIER_CONFIG = f"{DATA_DIR_REL}/config_classifier_20260908d.yaml"
CLASSIFIER_PROJECT_NAME = "landsat_vessel_classification_v2"
# Currently deployed run (Run d from the 20260908 experiments). New feedback retrains
# drop the a/b/c/d experiment letter and are named purely by date:
#   config:   data/landsat_vessels/config_classifier_<date>.yaml
#   run_name: <RUN_NAME_STEM>_<date>            e.g. olmoearth_base_layerdecay_20260911
#   ckpt:     gs://.../landsat_vessel_classification_v2/<run_name>/best.ckpt (date in path)
DEPLOYED_RUN_NAME = "olmoearth_base_layerdecay_20260908d"
RUN_NAME_STEM = "olmoearth_base_layerdecay"

# Where trained checkpoints live on weka (RSLP_PREFIX/projects/...) and where the served
# Dockerfile downloads them from. publish.py copies between the two.
GCS_PROJECTS_BUCKET = "ai2-rslearn-projects-data"
GCS_PROJECTS_PREFIX = "projects"  # gs://<bucket>/<prefix>/<project>/<run>/best.ckpt

DOCKERFILE_REL = "rslp/landsat_vessels/Dockerfile"
CONFIG_PY_REL = "rslp/landsat_vessels/config.py"


def gcs_checkpoint_url(project_name: str, run_name: str) -> str:
    """Public https URL the Dockerfile wgets the classifier checkpoint from."""
    return (
        f"https://storage.googleapis.com/{GCS_PROJECTS_BUCKET}/"
        f"{GCS_PROJECTS_PREFIX}/{project_name}/{run_name}/best.ckpt"
    )


def gcs_checkpoint_uri(project_name: str, run_name: str) -> str:
    """gs:// URI used when copying the checkpoint up with gsutil."""
    return (
        f"gs://{GCS_PROJECTS_BUCKET}/{GCS_PROJECTS_PREFIX}/"
        f"{project_name}/{run_name}/best.ckpt"
    )
