import pathlib

from rslp.landsat_vessels.predict_pipeline import predict_pipeline


def test_predict_pipeline(tmp_path: pathlib.Path) -> None:
    predict_pipeline(
        crop_path=str(tmp_path), scene_id="LC08_L1GT_114081_20241002_20241006_02_T2"
    )
