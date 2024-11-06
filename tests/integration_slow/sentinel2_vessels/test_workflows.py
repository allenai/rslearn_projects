import json
import pathlib

from upath import UPath

from rslp.sentinel2_vessels.predict_pipeline import PredictionTask, predict_pipeline


def test_predict_pipeline(tmp_path: pathlib.Path) -> None:
    should_have_vessels_task = PredictionTask(
        scene_id="S2A_MSIL1C_20161130T110422_N0204_R094_T30UYD_20161130T110418",
        json_path=str(tmp_path / "1.json"),
        crop_path=str(tmp_path / "crops_1"),
    )
    tasks = [should_have_vessels_task]
    # TODO: Test S2B_MSIL1C_20200206T222749_N0209_R072_T01LAL_20200206T234349 too but
    # right now it doesn't work for scenes like that which cross 0 longitude.

    scratch_path = tmp_path / "scratch"
    scratch_path.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        UPath(task.crop_path).mkdir(parents=True, exist_ok=True)

    predict_pipeline(
        tasks=tasks,
        scratch_path=str(scratch_path),
    )

    with UPath(should_have_vessels_task.json_path).open() as f:
        # This is some scene off coast of UK which should have a bunch of vessels.
        vessels = json.load(f)
        assert len(vessels) > 0
