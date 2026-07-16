"""Unit tests for how the Sentinel-1 API resolves the confidence threshold.

The API picks the effective cutoff (request value, else SENTINEL1_SCORE_THRESHOLD env
default, else none) and hands it to predict_pipeline, which overrides the detector's
score threshold. predict_pipeline is mocked here to capture the threshold it receives.
"""

import pytest
from fastapi.testclient import TestClient

from rslp.sentinel1_vessels import api_main

client = TestClient(api_main.app)

SCENE_ID = "S1A_IW_GRDH_1SDV_20241001T003924_20241001T003949_055902_06D56E_11E3.SAFE"


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake_pipeline(tasks, scratch_path, confidence_threshold):  # noqa: ANN001
        seen["threshold"] = confidence_threshold
        return [[]]

    monkeypatch.setattr(api_main, "predict_pipeline", fake_pipeline)
    return seen


def _call(payload: dict) -> None:
    resp = client.post("/detections", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_no_request_no_env_passes_none(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SENTINEL1_SCORE_THRESHOLD", raising=False)
    _call({"scene_id": SCENE_ID})
    assert captured["threshold"] is None


def test_request_threshold_passed_through(captured: dict) -> None:
    _call({"scene_id": SCENE_ID, "confidence_threshold": 0.7})
    assert captured["threshold"] == 0.7


def test_env_default_used_without_request(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SENTINEL1_SCORE_THRESHOLD", "0.7")
    _call({"scene_id": SCENE_ID})
    assert captured["threshold"] == 0.7


def test_request_overrides_env_default(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SENTINEL1_SCORE_THRESHOLD", "0.9")
    _call({"scene_id": SCENE_ID, "confidence_threshold": 0.6})
    assert captured["threshold"] == 0.6
