""" Use this script to inference the API with locally stored data"""

import json
import os

import requests

PORT = os.getenv("Landsat_PORT", default=5555)
LANDSAT_ENDPOINT = f"http://localhost:{PORT}/detections"
TIMEOUT_SECONDS = 60000
SCENE_ID = 'LC09_L1GT_106084_20241002_20241002_02_T2'
CROP_PATH = 'temp_crops'

def sample_request() -> None:
    """Sample request for files stored locally"""

    REQUEST_BODY = {
        "scene_id": SCENE_ID,
        "crop_path": CROP_PATH,
    }

    response = requests.post(LANDSAT_ENDPOINT, json=REQUEST_BODY, timeout=TIMEOUT_SECONDS)
    output_filename = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "sample_response.json"
    )
    if response.ok:
        with open(output_filename, "w") as outfile:
            json.dump(response.json(), outfile)


if __name__ == "__main__":
    sample_request()
