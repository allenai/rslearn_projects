"""This module contains the configuration for the Landsat Vessel Detection pipeline."""

import json

# Landsat config
LANDSAT_LAYER_NAME = "landsat"
OUTPUT_LAYER_NAME = "output"
LANDSAT_RESOLUTION = 15

# Data config
LOCAL_FILES_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config.json"
AWS_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config_aws.json"

# Extract Landsat bands from local config file
with open(LOCAL_FILES_DATASET_CONFIG) as f:
    json_data = json.load(f)
LANDSAT_BANDS = [
    band["bands"][0] for band in json_data["layers"][LANDSAT_LAYER_NAME]["band_sets"]
]

# Model config
DETECT_MODEL_CONFIG = "data/landsat_vessels/config_detector.yaml"
CLASSIFY_MODEL_CONFIG = "data/landsat_vessels/config_classifier.yaml"
CLASSIFY_WINDOW_SIZE = 128

# Filter config
INFRA_THRESHOLD_KM = 0.03  # max-distance between marine infra and prediction

# Evaluation config
MATCH_THRESHOLD_KM = 0.1  # max-distance between ground-truth and prediction
