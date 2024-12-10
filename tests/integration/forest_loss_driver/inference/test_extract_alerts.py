"""Integration tests for extract_alerts.py step of the inference pipeline."""

import os
import tempfile
import uuid
from datetime import datetime, timezone

import pytest
from upath import UPath

from rslp.forest_loss_driver.inference.config import PredictPipelineConfig
from rslp.forest_loss_driver.inference.extract_alerts import extract_alerts_pipeline
from rslp.log_utils import get_logger

TEST_ID = str(uuid.uuid4())
logger = get_logger(__name__)


@pytest.fixture
def tiff_filename() -> str:
    """The path to the alert GeoTIFF file."""
    logger.warning("This tif is on GCS and is downloaded in conftest.py")
    return "cropped_070W_10S_060W_00N.tif"


def test_extract_alerts(
    tiff_filename: str,
    alert_tiffs_prefix: str,
    alert_date_tiffs_prefix: str,
    inference_dataset_config_path: str,
) -> None:
    """Tests extracting alerts from a single GeoTIFF file."""
    with tempfile.TemporaryDirectory(prefix=f"test_{TEST_ID}_") as temp_dir:
        index_cache_dir = UPath(temp_dir) / "index_cache"
        tile_store_root_dir = UPath(temp_dir) / "tile_store"
        os.environ["INFERENCE_DATASET_CONFIG"] = inference_dataset_config_path
        os.environ["INDEX_CACHE_DIR"] = str(index_cache_dir)
        os.environ["TILE_STORE_ROOT_DIR"] = str(tile_store_root_dir)
        dummy_model_cfg_fname = "dummy_model_cfg.json"  # Not used in this step
        predict_pipeline_config = PredictPipelineConfig(
            ds_root=UPath(temp_dir)
            / "datasets"
            / "forest_loss_driver"
            / "prediction"
            / "dataset_20241023",
            ignore_errors=False,
            model_cfg_fname=dummy_model_cfg_fname,
            gcs_tiff_filenames=[tiff_filename],
            workers=1,
            days=365,
            min_confidence=1,
            min_area=16.0,
            conf_prefix=alert_tiffs_prefix,
            date_prefix=alert_date_tiffs_prefix,
            prediction_utc_time=datetime(2024, 10, 23, tzinfo=timezone.utc),
        )
        extract_alerts_pipeline(predict_pipeline_config, tiff_filename)

        # Assert one of the windows has all the info
        expected_image_path = (
            UPath(temp_dir)
            / "datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/layers/mask/mask/image.png"
        )
        expected_info_json_path = (
            UPath(temp_dir)
            / "datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/info.json"
        )
        expected_metadata_json_path = (
            UPath(temp_dir)
            / "datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/metadata.json"
        )
        expected_image_metadata_json_path = (
            UPath(temp_dir)
            / "datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/layers/mask/mask/metadata.json"
        )
        expected_completed_path = (
            UPath(temp_dir)
            / "datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/layers/mask/completed"
        )
        expected_dataset_config_path = (
            UPath(temp_dir)
            / "datasets/forest_loss_driver/prediction/dataset_20241023/config.json"
        )
        # add step looking for the config.json
        assert (
            expected_image_path.exists()
        ), f"Path {expected_image_path} does not exist"
        assert (
            expected_info_json_path.exists()
        ), f"Path {expected_info_json_path} does not exist"
        assert (
            expected_metadata_json_path.exists()
        ), f"Path {expected_metadata_json_path} does not exist"
        assert (
            expected_image_metadata_json_path.exists()
        ), f"Path {expected_image_metadata_json_path} does not exist"
        assert (
            expected_completed_path.exists()
        ), f"Path {expected_completed_path} does not exist"
        assert (
            expected_dataset_config_path.exists()
        ), f"Path {expected_dataset_config_path} does not exist"
