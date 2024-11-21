import pathlib

from rslp.landsat_vessels.predict_pipeline import predict_pipeline


def test_predict_pipeline(tmp_path: pathlib.Path) -> None:
    predict_pipeline(
        scene_zip_path="gs://test-bucket-rslearn/Landsat/LC08_L1TP_162042_20241103_20241103_02_RT.zip"
    )
