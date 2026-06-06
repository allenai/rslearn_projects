"""This module contains the configuration for the Landsat Vessel Detection pipeline."""

import json

# Landsat config
LANDSAT_LAYER_NAME = "landsat"
LANDSAT_ALLBANDS_LAYER_NAME = "landsat_allbands"
OUTPUT_LAYER_NAME = "output"
LANDSAT_RESOLUTION = 15

# Data config
LOCAL_FILES_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config.json"
AWS_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config_aws.json"

# All Landsat bands required by the prediction pipeline. The detector and classifier
# only use a subset, but the attribute model reads the full band stack via the
# landsat_allbands layer, so every band must be provided.
with open(LOCAL_FILES_DATASET_CONFIG) as f:
    json_data = json.load(f)
LANDSAT_ALLBANDS = json_data["layers"][LANDSAT_ALLBANDS_LAYER_NAME]["band_sets"][0][
    "bands"
]

# Model config
DETECT_MODEL_CONFIG = "data/landsat_vessels/config_detector.yaml"
CLASSIFY_MODEL_CONFIG = "data/landsat_vessels/config_classifier.yaml"
CLASSIFY_WINDOW_SIZE = 128
ATTRIBUTE_MODEL_CONFIG = "data/landsat_vessel_attribute/config.yaml"
ATTRIBUTE_WINDOW_SIZE = 128

# Filter config
INFRA_THRESHOLD_KM = 0.03  # max-distance between marine infra and prediction

# Evaluation config
MATCH_THRESHOLD_KM = 0.1  # max-distance between ground-truth and prediction

# We make sure the windows we create for Landsat scenes are multiples of this amount
# because we store some bands at 1/2 of the input resolution, so the window size needs
# be a multiple of 2.
WINDOW_MIN_MULTIPLE = 2
