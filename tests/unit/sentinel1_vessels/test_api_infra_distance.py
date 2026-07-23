"""Unit tests for how the Sentinel-1 API resolves the marine infrastructure distance.

The API uses the request's infra_distance_km when provided, otherwise the
SENTINEL1_INFRA_DISTANCE_KM module default (from the env var, 0.2 if unset), and hands
that to predict_pipeline. predict_pipeline is mocked here to capture the value.
"""

import pytest
from fastapi.testclient import TestClient

from rslp.sentinel1_vessels import api_main

client = TestClient(api_main.app)

SCENE_ID = "S1A_IW_GRDH_1SDV_20241001T003924_20241001T003949_055902_06D56E_11E3.SAFE"


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict:
    seen: dict = {}

    def fake_pipeline(
        tasks: list,
        score_threshold: float,
        scratch_path: str,
        infra_distance_km: float,
    ) -> list:
        seen["infra_distance_km"] = infra_distance_km
        return [[]]

    monkeypatch.setattr(api_main, "predict_pipeline", fake_pipeline)
    return seen


def _call(payload: dict) -> None:
    resp = client.post("/detections", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_default_used_without_request(captured: dict) -> None:
    _call({"scene_id": SCENE_ID})
    assert captured["infra_distance_km"] == api_main.SENTINEL1_INFRA_DISTANCE_KM


def test_default_is_200m() -> None:
    assert api_main.SENTINEL1_INFRA_DISTANCE_KM == 0.2


def test_default_is_configurable(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "SENTINEL1_INFRA_DISTANCE_KM", 0.05)
    _call({"scene_id": SCENE_ID})
    assert captured["infra_distance_km"] == 0.05


def test_request_overrides_default(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "SENTINEL1_INFRA_DISTANCE_KM", 0.2)
    _call({"scene_id": SCENE_ID, "infra_distance_km": 0.5})
    assert captured["infra_distance_km"] == 0.5


def test_zero_request_is_not_swallowed(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 0.0 is a real request value, not "unset" -- it must win over the default.
    monkeypatch.setattr(api_main, "SENTINEL1_INFRA_DISTANCE_KM", 0.2)
    _call({"scene_id": SCENE_ID, "infra_distance_km": 0.0})
    assert captured["infra_distance_km"] == 0.0
