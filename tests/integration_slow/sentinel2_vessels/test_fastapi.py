from fastapi.testclient import TestClient

from rslp.sentinel2_vessels.api_main import app

client = TestClient(app)


def test_scene() -> None:
    """Test inference using a scene from GCP (free bucket)."""
    response = client.post(
        "/detections",
        json={
            "scene_id": "S2A_MSIL1C_20180904T110621_N0206_R137_T30UYD_20180904T133425"
        },
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    # There are many correct vessels in this scene.
    assert len(predictions) > 0
