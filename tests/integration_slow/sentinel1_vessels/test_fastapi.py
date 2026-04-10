"""Slow integration tests for the Sentinel-1 vessel detection FastAPI service."""

import os
import tempfile

import boto3
from fastapi.testclient import TestClient

from rslp.sentinel1_vessels.api_main import app

client = TestClient(app)

SCENE_ID = "S1A_IW_GRDH_1SDV_20241001T003924_20241001T003949_055902_06D56E_11E3.SAFE"

BUCKET_NAME = "sentinel-s1-l1c"


def _s3_key(scene_id: str, band: str) -> str:
    """Build the S3 object key for a Sentinel-1 GRD band."""
    prefix = scene_id.split(".")[0]
    ts = prefix.split("_")[4]
    return (
        f"GRD/{int(ts[:4])}/{int(ts[4:6])}/{int(ts[6:8])}"
        f"/IW/DV/{prefix}/measurement/iw-{band}.tiff"
    )


def _download_raw(tmp_dir: str, scene_id: str, band: str) -> str:
    """Download a raw S1 GRD band GeoTIFF (with GCPs) from AWS."""
    s3 = boto3.client("s3")
    out_path = os.path.join(tmp_dir, f"{band}.tif")
    s3.download_file(
        BUCKET_NAME,
        _s3_key(scene_id, band),
        out_path,
        ExtraArgs={"RequestPayer": "requester"},
    )
    return out_path


def test_scene() -> None:
    """Test inference using a scene from Copernicus / AWS."""
    response = client.post(
        "/detections",
        json={"scene_id": SCENE_ID},
    )
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) > 0


def test_image_files() -> None:
    """Test inference using image files downloaded from the S1 AWS bucket.

    Downloads raw VV/VH GeoTIFFs (with GCPs) for a single scene and passes
    them directly as image / historical1 / historical2 to exercise the
    image-files code path end-to-end.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        vv_path = _download_raw(tmp_dir, SCENE_ID, "vv")
        vh_path = _download_raw(tmp_dir, SCENE_ID, "vh")

        image_payload = {"vv": vv_path, "vh": vh_path}

        response = client.post(
            "/detections",
            json={
                "image": image_payload,
                "historical1": image_payload,
                "historical2": image_payload,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert isinstance(data["predictions"], list)
