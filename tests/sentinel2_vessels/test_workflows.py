import os
import pathlib

from rslp.sentinel2_vessels import predict_pipeline


def test_predict_pipeline(tmp_path: pathlib.Path):
    scratch_path = tmp_path / "scratch"
    scratch_path.mkdir(parents=True, exist_ok=True)
    crop_path = tmp_path / "crops"
    crop_path.mkdir(parents=True, exist_ok=True)
    json_path = tmp_path / "vessels.json"
    predict_pipeline(
        scene_id="S2B_MSIL1C_20200206T222749_N0209_R072_T01LAL_20200206T234349",
        scratch_path=str(scratch_path),
        crop_path=str(crop_path),
        json_path=json_path,
    )
    assert os.path.exists(json_path)
