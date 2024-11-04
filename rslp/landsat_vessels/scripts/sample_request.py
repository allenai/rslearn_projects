"""Sample request to the landsat vessels API."""

import requests

from rslp.landsat_vessels.api_main import LANDSAT_HOST, LANDSAT_PORT

SCENE_ZIP_PATH = "gs://skylight-sat-imagery-sky-int-a/sat-service/landsat_8_9/downloads/2024/11/04/LC08_L1GT_201022_20241104_20241104_02_RT.zip"
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
