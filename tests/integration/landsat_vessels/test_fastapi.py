from fastapi.testclient import TestClient

from rslp.landsat_vessels.api_main import app

client = TestClient(app)


def test_singapore_dense_scene() -> None:
    # LC08_L1TP_162042_20241103_20241103_02_RT is a cropped 10000m x 10000m scene
    # that should have at least 1 vessel detection.
    response = client.post(
        "/detections",
        json={
            "scene_zip_path": "gs://test-bucket-rslearn/Landsat/LC08_L1TP_162042_20241103_20241103_02_RT.zip"
        },
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    # There are many correct vessels in this scene.
    assert len(predictions) > 0
