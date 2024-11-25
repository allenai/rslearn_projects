"""This module contains the configuration for the Landsat Vessel Detection pipeline."""

import json

# Landsat configuration
LANDSAT_LAYER_NAME = "landsat"
LANDSAT_RESOLUTION = 15

# Detector configuration
LOCAL_FILES_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config.json"
AWS_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config_aws.json"
DETECT_MODEL_CONFIG = "data/landsat_vessels/config.yaml"
DETECT_MODEL_EVAL_CONFIG = (
    "data/landsat_vessels/config_eval.yaml"  # config for evaluation
)

# Extract Landsat bands from local config file
with open(LOCAL_FILES_DATASET_CONFIG) as f:
    json_data = json.load(f)
LANDSAT_BANDS = [
    band["bands"][0] for band in json_data["layers"][LANDSAT_LAYER_NAME]["band_sets"]
]

# Classifier configuration
CLASSIFY_MODEL_CONFIG = "landsat/recheck_landsat_labels/phase123_config.yaml"
CLASSIFY_WINDOW_SIZE = 64

# Filter configuration
INFRA_DISTANCE_THRESHOLD = 0.1  # unit: km, 100 meters
