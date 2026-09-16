"""Unit tests for the NISAR vessel detection API's request handling.

predict_pipeline is mocked throughout so these exercise only what the API layer does
with a request: resolving the score threshold, and turning the request into a
PredictionTask.
"""

from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

from rslp.nisar_vessels import api_main
from rslp.nisar_vessels.predict_pipeline import PredictionTask

client = TestClient(api_main.app)

H5_PATH = (
    "/shared/NISAR_L2_GCOV_009_055_A_014_4005_DHDH_A_20260101T000312_20260101T000347.h5"
)


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Capture the arguments predict_pipeline is called with."""
    seen: dict = {}

    def fake_pipeline(
        tasks: list[PredictionTask], score_threshold: float, scratch_path: str
    ) -> list[list]:
        seen["tasks"] = tasks
        seen["threshold"] = score_threshold
        seen["scratch_path"] = scratch_path
        return [[]]

    monkeypatch.setattr(api_main, "predict_pipeline", fake_pipeline)
    return seen


def _call(payload: dict) -> dict:
    resp = client.post("/detections", json=payload)
    assert resp.status_code == HTTPStatus.OK
    body = resp.json()
    assert body["status"] == "success", body.get("error_message")
    return body


def test_home() -> None:
    assert client.get("/").status_code == HTTPStatus.OK


def test_h5_path_is_required() -> None:
    # The sidecar has no data source of its own, so a request without a granule cannot
    # be served at all.
    assert (
        client.post("/detections", json={}).status_code
        == HTTPStatus.UNPROCESSABLE_ENTITY
    )


def test_request_becomes_a_prediction_task(captured: dict) -> None:
    _call({"h5_path": H5_PATH, "crop_path": "/shared/crops"})

    (task,) = captured["tasks"]
    assert task.h5_path == H5_PATH
    assert task.crop_path == "/shared/crops"


def test_scene_id_defaults_to_the_granule_filename(captured: dict) -> None:
    """A caller with only a path gets a usable scene ID without supplying one."""
    _call({"h5_path": H5_PATH})

    (task,) = captured["tasks"]
    assert (
        task.get_scene_id()
        == "NISAR_L2_GCOV_009_055_A_014_4005_DHDH_A_20260101T000312_20260101T000347"
    )


def test_scene_id_can_be_set_by_the_request(captured: dict) -> None:
    _call({"h5_path": H5_PATH, "scene_id": "some-other-granule"})

    (task,) = captured["tasks"]
    assert task.get_scene_id() == "some-other-granule"


def test_scratch_path_is_passed_through(captured: dict) -> None:
    _call({"h5_path": H5_PATH, "scratch_path": "/shared/scratch"})

    assert captured["scratch_path"] == "/shared/scratch"


def test_scratch_path_defaults_to_a_temporary_directory(captured: dict) -> None:
    _call({"h5_path": H5_PATH})

    assert captured["scratch_path"]


def test_default_threshold_used_without_request(captured: dict) -> None:
    _call({"h5_path": H5_PATH})

    assert captured["threshold"] == api_main.NISAR_SCORE_THRESHOLD


def test_default_threshold_is_configurable(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "NISAR_SCORE_THRESHOLD", 0.5)

    _call({"h5_path": H5_PATH})

    assert captured["threshold"] == 0.5


def test_request_threshold_overrides_default(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_main, "NISAR_SCORE_THRESHOLD", 0.9)

    _call({"h5_path": H5_PATH, "score_threshold": 0.6})

    assert captured["threshold"] == 0.6


def test_zero_threshold_is_not_swallowed(
    captured: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 0.0 is a real request value, not "unset" -- it must win over the default.
    monkeypatch.setattr(api_main, "NISAR_SCORE_THRESHOLD", 0.7)

    _call({"h5_path": H5_PATH, "score_threshold": 0.0})

    assert captured["threshold"] == 0.0


def test_pipeline_error_becomes_an_error_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A granule the pipeline cannot read is reported, not raised at the caller."""

    def failing_pipeline(**kwargs: object) -> list[list]:
        raise ValueError("No group in the granule has band HVHV")

    monkeypatch.setattr(api_main, "predict_pipeline", failing_pipeline)

    resp = client.post("/detections", json={"h5_path": H5_PATH})

    assert resp.status_code == HTTPStatus.OK
    body = resp.json()
    assert body["status"] == "error"
    assert "HVHV" in body["error_message"]
    assert body["predictions"] == []
