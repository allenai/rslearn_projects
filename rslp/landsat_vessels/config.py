"""This module contains the configuration for the Landsat Vessel Detection pipeline."""

# Landsat configuration
LANDSAT_LAYER_NAME = "landsat"
LANDSAT_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]
LANDSAT_RESOLUTION = 15

# Detector configuration
LOCAL_FILES_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config.json"
AWS_DATASET_CONFIG = "data/landsat_vessels/predict_dataset_config_aws.json"
DETECT_MODEL_CONFIG = "data/landsat_vessels/config.yaml"

# Classifier configuration
CLASSIFY_MODEL_CONFIG = "landsat/recheck_landsat_labels/phase123_config.yaml"
CLASSIFY_WINDOW_SIZE = 64

# Filter configuration
INFRA_DISTANCE_THRESHOLD = 0.1  # unit: km, 100 meters
