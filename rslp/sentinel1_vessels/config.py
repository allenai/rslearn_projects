"""Environment-configurable settings for the Sentinel-1 vessel detection program.

Single source of truth for every value the Sentinel-1 API and prediction pipeline
read from the environment. Import the constants from here rather than calling
os.getenv directly, so all configurable knobs live in one place.
"""

import os

from dotenv import load_dotenv

# Load environment variables from the .env file before reading any of them below.
load_dotenv()

# Host and port the FastAPI server binds to.
SENTINEL1_HOST = os.getenv("SENTINEL1_HOST", "0.0.0.0")
SENTINEL1_PORT = int(os.getenv("SENTINEL1_PORT", 5555))

# Default detector score threshold, overridable per request via the API.
SENTINEL1_SCORE_THRESHOLD = float(os.getenv("SENTINEL1_SCORE_THRESHOLD", "0.7"))

# Distance threshold for the near marine infrastructure filter in km
# (0.2 km = 200 m if unset).
INFRA_DISTANCE_THRESHOLD = float(os.getenv("SENTINEL1_INFRA_DISTANCE_KM", "0.2"))

# Number of workers the rslearn data loader uses during prediction.
NUM_DATA_LOADER_WORKERS = int(os.environ.get("RSLEARN_NUM_DATA_LOADER_WORKERS", 4))
