"""Slow integration tests for the Sentinel-1 vessel prediction pipeline."""

import json
import pathlib

from upath import UPath

from rslp.sentinel1_vessels.predict_pipeline import PredictionTask, predict_pipeline

SCENE_ID = "S1A_IW_GRDH_1SDV_20241001T003924_20241001T003949_055902_06D56E_11E3.SAFE"


def test_predict_pipeline_scene_with_vessels(tmp_path: pathlib.Path) -> None:
    """Verify that prediction detects vessels in a scene known to have vessels."""
    task = PredictionTask(
        scene_id=SCENE_ID,
        json_path=str(tmp_path / "detections.json"),
        crop_path=str(tmp_path / "crops"),
    )

    scratch_path = tmp_path / "scratch"
    scratch_path.mkdir(parents=True, exist_ok=True)
    UPath(task.crop_path).mkdir(parents=True, exist_ok=True)

    predict_pipeline(
        tasks=[task],
        scratch_path=str(scratch_path),
    )

    with UPath(task.json_path).open() as f:
        vessels = json.load(f)
        assert len(vessels) > 0
