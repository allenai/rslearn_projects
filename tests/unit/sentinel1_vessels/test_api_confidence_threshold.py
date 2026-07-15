"""Unit tests for the Sentinel-1 API's per-request confidence_threshold filter.

The model decodes down to its configured floor (0.5); the API drops detections
below the caller's requested threshold. predict_pipeline is mocked so these run
without AWS or model inference.
"""

import pytest
from fastapi.testclient import TestClient

from rslp.sentinel1_vessels import api_main
from rslp.vessels import VesselDetectionSource

client = TestClient(api_main.app)

SCENE_ID = "S1A_IW_GRDH_1SDV_20241001T003924_20241001T003949_055902_06D56E_11E3.SAFE"


class _FakeDetection:
    """Stand-in for VesselDetection exposing only what the handler touches."""

    def __init__(self, score: float) -> None:
        self.score = score

    def to_dict(self) -> dict:
        return {
            "source": VesselDetectionSource.SENTINEL1,
            "col": 0,
            "row": 0,
            "projection": {},
            "score": self.score,
            "ts": None,
            "scene_id": SCENE_ID,
            "crop_fname": None,
            "crop_fnames": None,
            "longitude": 0.0,
            "latitude": 0.0,
            "attributes": None,
        }


@pytest.fixture
def _fake_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    detections = [_FakeDetection(0.55), _FakeDetection(0.72), _FakeDetection(0.95)]
    monkeypatch.setattr(
        api_main, "predict_pipeline", lambda tasks, scratch_path: [detections]
    )


def _scores(payload: dict) -> list[float]:
    resp = client.post("/detections", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    return sorted(p["score"] for p in body["predictions"])


def test_no_threshold_returns_all(_fake_pipeline: None) -> None:
    assert _scores({"scene_id": SCENE_ID}) == [0.55, 0.72, 0.95]


def test_threshold_filters_below(_fake_pipeline: None) -> None:
    assert _scores({"scene_id": SCENE_ID, "confidence_threshold": 0.7}) == [0.72, 0.95]


def test_threshold_is_inclusive(_fake_pipeline: None) -> None:
    assert _scores({"scene_id": SCENE_ID, "confidence_threshold": 0.72}) == [0.72, 0.95]


def test_floor_threshold_keeps_everything(_fake_pipeline: None) -> None:
    assert _scores({"scene_id": SCENE_ID, "confidence_threshold": 0.5}) == [
        0.55,
        0.72,
        0.95,
    ]


def test_env_default_applies_without_request_threshold(
    _fake_pipeline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "DEFAULT_CONFIDENCE_THRESHOLD", 0.7)
    assert _scores({"scene_id": SCENE_ID}) == [0.72, 0.95]


def test_request_threshold_overrides_env_default(
    _fake_pipeline: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "DEFAULT_CONFIDENCE_THRESHOLD", 0.9)
    # Request cutoff is looser than the env default, so it wins and more is returned.
    assert _scores({"scene_id": SCENE_ID, "confidence_threshold": 0.6}) == [0.72, 0.95]
