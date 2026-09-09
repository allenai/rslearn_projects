"""This module contains the configuration for the Landsat Vessel Detection pipeline."""

import json
from pathlib import Path

# Config paths are anchored to the repository root rather than the process cwd. This
# module reads one of them at import time, and importing it is no longer only something
# the pipeline does from the repo root: a training config that references
# rslp.landsat_vessels.transforms pulls this module in through the package __init__ while
# jsonargparse has chdir'd into the config file's own directory, which made the relative
# read fail.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Landsat config
LANDSAT_LAYER_NAME = "landsat"
LANDSAT_ALLBANDS_LAYER_NAME = "landsat_allbands"
OUTPUT_LAYER_NAME = "output"
LANDSAT_RESOLUTION = 15

# Data config
LOCAL_FILES_DATASET_CONFIG = str(
    _REPO_ROOT / "data/landsat_vessels/predict_dataset_config.json"
)
AWS_DATASET_CONFIG = str(
    _REPO_ROOT / "data/landsat_vessels/predict_dataset_config_aws.json"
)

# Extract Landsat bands from local config file.
# LANDSAT_BANDS covers the 7-band `landsat` layer used by the detector.
# LANDSAT_ALL_BAND_NAMES is the full set needed by the classifier and attribute models
# (via `landsat_allbands`), and is also used for scene zip extraction so all bands are
# available to every pipeline stage.
with open(LOCAL_FILES_DATASET_CONFIG) as f:
    json_data = json.load(f)
LANDSAT_BANDS = [
    band["bands"][0] for band in json_data["layers"][LANDSAT_LAYER_NAME]["band_sets"]
]
LANDSAT_ALL_BAND_NAMES = json_data["layers"][LANDSAT_ALLBANDS_LAYER_NAME]["band_sets"][
    0
]["bands"]

# Model config
# Detector: config_detector.yaml (score_threshold=0.7).
# Classifier: Run-d layer-decay model (olmoearth_base_layerdecay_20260908d), deployed at
# positive_class_threshold=0.99. Together this is the det0.7 / cls0.99 operating point.
DETECT_MODEL_CONFIG = str(_REPO_ROOT / "data/landsat_vessels/config_detector.yaml")
CLASSIFY_MODEL_CONFIG = str(
    _REPO_ROOT / "data/landsat_vessels/config_classifier_20260908d.yaml"
)
CLASSIFY_WINDOW_SIZE = 64
ATTRIBUTE_MODEL_CONFIG = str(_REPO_ROOT / "data/landsat_vessel_attribute/config.yaml")
ATTRIBUTE_WINDOW_SIZE = 128

# Filter config
INFRA_THRESHOLD_KM = 0.03  # max-distance between marine infra and prediction

# Evaluation config
MATCH_THRESHOLD_KM = 0.1  # max-distance between ground-truth and prediction

# We make sure the windows we create for Landsat scenes are multiples of this amount
# because we store some bands at 1/2 of the input resolution, so the window size needs
# be a multiple of 2.
WINDOW_MIN_MULTIPLE = 2
