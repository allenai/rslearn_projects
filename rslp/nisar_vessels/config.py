"""Environment-configurable settings for the NISAR vessel detection program.

Single source of truth for every value the NISAR API and prediction pipeline read from
the environment. Import the constants from here rather than calling os.getenv directly,
so all configurable knobs live in one place.
"""

import os

from dotenv import load_dotenv

from rslp.utils.filter import DEFAULT_INFRA_PATH

# Load environment variables from the .env file before reading any of them below.
load_dotenv()

# Host and port the FastAPI server binds to.
NISAR_HOST = os.getenv("NISAR_HOST", "0.0.0.0")
NISAR_PORT = int(os.getenv("NISAR_PORT", "5555"))

# Default detector score threshold, overridable per request via the API.
NISAR_SCORE_THRESHOLD = float(os.getenv("NISAR_SCORE_THRESHOLD", "0.7"))

# Distance threshold for the near marine infrastructure filter, in km. Measured on the
# 20260828 predict set: detections nearest the infra layer cluster 10-20 m out, with
# nothing between 70 and 100 m.
INFRA_DISTANCE_THRESHOLD_KM = float(os.getenv("NISAR_INFRA_DISTANCE_KM", "0.05"))

# GeoJSON of marine infrastructure that detections are filtered against. rslp.utils.filter
# reads MARINE_INFRA_PATH from the environment, falling back to a public URL.
MARINE_INFRA_PATH = DEFAULT_INFRA_PATH

# Number of workers the rslearn data loader uses during prediction.
NUM_DATA_LOADER_WORKERS = int(os.getenv("RSLEARN_NUM_DATA_LOADER_WORKERS", "4"))

# Number of workers used to prepare and materialize the rslearn dataset. A request
# carries a single granule, so this only parallelizes across the bands of one window.
NUM_MATERIALIZE_WORKERS = int(os.getenv("NISAR_MATERIALIZE_WORKERS", "32"))

# Side length, in pixels, of the tiles a scene is split into for detection. Materializing
# a window holds it in memory, so this, not the granule size, sets peak usage.
SCENE_TILE_SIZE = int(os.getenv("NISAR_SCENE_TILE_SIZE", "4096"))

# Overlap between adjacent scene tiles, so a vessel on a seam falls fully inside one of
# them. Only has to exceed a vessel's ~15 pixel footprint.
SCENE_TILE_OVERLAP = int(os.getenv("NISAR_SCENE_TILE_OVERLAP", "64"))

# How the detector crops each window at inference time, matching what it trained on.
PREDICT_CROP_SIZE = int(os.getenv("NISAR_PREDICT_CROP_SIZE", "128"))
PREDICT_OVERLAP_PIXELS = int(os.getenv("NISAR_PREDICT_OVERLAP_PIXELS", "16"))
