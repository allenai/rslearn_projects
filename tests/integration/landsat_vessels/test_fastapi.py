import tempfile

from fastapi.testclient import TestClient

from rslp.landsat_vessels.api_main import app

client = TestClient(app)


def test_singapore_dense_scene():
    # LC08_L1TP_125059_20240913_20240920_02_T1 is a scene that includes southeast coast
    # of Singapore where there are hundreds of vessels.
    with tempfile.TemporaryDirectory() as tmp_dir:
        response = client.post(
            "/detections",
            json={
                "scene_id": "LC08_L1TP_125059_20240913_20240920_02_T1",
                "scratch_path": tmp_dir,
                "json_path": "",
                "crop_path": "",
            },
        )
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        # There are many correct vessels in this scene.
        assert len(predictions) >= 100
