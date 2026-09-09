"""This module contains the configuration for the Landsat Vessel Detection pipeline."""

import json
import os
from pathlib import Path

# Config paths are anchored to the directory that actually holds the `data/` tree rather
# than the process cwd, because this module reads one of those files at import time and it
# is no longer only imported from the repo root. The correct root differs by how rslp is
# being run, so we probe a list of candidates and use the first one that contains the
# dataset config:
#   1. $RSLP_DATA_ROOT, if set (explicit override).
#   2. Two levels above this file. For a source checkout or an editable install this is the
#      repo root next to data/. This also survives jsonargparse chdir'ing into a training
#      config's own directory at import time (the path is absolute), which is the case the
#      __file__ anchor was originally added to fix.
#   3. The process cwd. For a non-editable install (e.g. CI, where rslp lives under
#      site-packages and is no longer next to data/) the working tree that holds data/ is
#      the cwd, so this is what makes the installed-package layout work.
_DATASET_CONFIG_MARKER = Path("data/landsat_vessels/predict_dataset_config.json")


def _find_repo_root() -> Path:
    """Return the directory containing the `data/` config tree."""
    candidates = []
    env_root = os.environ.get("RSLP_DATA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path(__file__).resolve().parents[2])
    candidates.append(Path.cwd())
    for root in candidates:
        if (root / _DATASET_CONFIG_MARKER).exists():
            return root
    # Fall back to the file-anchored root so any resulting error points somewhere sensible.
    return Path(__file__).resolve().parents[2]


_REPO_ROOT = _find_repo_root()

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

# All Landsat bands required by the prediction pipeline. The detector and classifier
# only use a subset, but the attribute model reads the full band stack via the
# landsat_allbands layer, so every band must be provided. It is also used for scene zip
# extraction so all bands are available to every pipeline stage.
with open(LOCAL_FILES_DATASET_CONFIG) as f:
    json_data = json.load(f)
LANDSAT_ALLBANDS = json_data["layers"][LANDSAT_ALLBANDS_LAYER_NAME]["band_sets"][0][
    "bands"
]

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
