"""Unit tests for how the Sentinel-1 API resolves the detector score threshold.

The API uses the request's score_threshold when provided, otherwise the
SENTINEL1_SCORE_THRESHOLD module default (from the env var, 0.7 if unset), and hands
that to predict_pipeline, which overrides the detector's score threshold. predict_pipeline
is mocked here to capture the value it receives.
"""

import pytest
from fastapi.testclient import TestClient

from rslp.sentinel1_vessels import api_main

client = TestClient(api_main.app)

SCENE_ID = "S1A_IW_GRDH_1SDV_20241001T003924_20241001T003949_055902_06D56E_11E3.SAFE"


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake_pipeline(tasks: list, score_threshold: float, scratch_path: str) -> list:
        seen["threshold"] = score_threshold
        return [[]]

    monkeypatch.setattr(api_main, "predict_pipeline", fake_pipeline)
    return seen


def _call(payload: dict) -> None:
    resp = client.post("/detections", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_default_used_without_request(captured: dict) -> None:
    _call({"scene_id": SCENE_ID})
    assert captured["threshold"] == api_main.SENTINEL1_SCORE_THRESHOLD


def test_default_is_configurable(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "SENTINEL1_SCORE_THRESHOLD", 0.5)
    _call({"scene_id": SCENE_ID})
    assert captured["threshold"] == 0.5


def test_request_overrides_default(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "SENTINEL1_SCORE_THRESHOLD", 0.9)
    _call({"scene_id": SCENE_ID, "score_threshold": 0.6})
    assert captured["threshold"] == 0.6


def test_zero_request_is_not_swallowed(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 0.0 is a real request value, not "unset" -- it must win over the default.
    monkeypatch.setattr(api_main, "SENTINEL1_SCORE_THRESHOLD", 0.7)
    _call({"scene_id": SCENE_ID, "score_threshold": 0.0})
    assert captured["threshold"] == 0.0
