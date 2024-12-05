"""Sample request to the landsat vessels API."""

import requests

from rslp.landsat_vessels.api_main import LANDSAT_HOST, LANDSAT_PORT

SCENE_ZIP_PATH = (
    "gs://test-bucket-rslearn/Landsat/LC08_L1TP_162042_20241103_20241103_02_RT.zip"
)
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
