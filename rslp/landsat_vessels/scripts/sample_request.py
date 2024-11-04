"""Sample request to the landsat vessels API."""

import requests

from rslp.landsat_vessels.api_main import LANDSAT_HOST, LANDSAT_PORT

SCENE_ZIP_PATH = "gs://rslearn-eai/projects/2024_10_check_landsat/downloads/LC08_L1GT_010055_20241025_20241025_02_RT.zip"
TIMEOUT_SECONDS = 600


def sample_request() -> None:
    """Sample request to the landsat vessels API."""
    # Define the URL of the API endpoint
    url = f"http://{LANDSAT_HOST}:{LANDSAT_PORT}/detections"
    payload = {"scene_zip_path": SCENE_ZIP_PATH}
    # Send a POST request to the API
    response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    # Print the response from the API
    print(response.json())


if __name__ == "__main__":
    sample_request()
