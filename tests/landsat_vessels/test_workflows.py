from rslp.landsat_vessels.predict_pipeline import predict_pipeline


def test_predict_pipeline(tmp_path):
    predict_pipeline(
        crop_path=tmp_path, scene_id="LC08_L1GT_114081_20241002_20241006_02_T2"
    )
