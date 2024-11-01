"""Integration tests for extract_alerts.py step of the inference pipeline."""

import os
import uuid
from datetime import datetime, timezone

import pytest
from google.cloud import storage
from upath import UPath

from rslp.forest_loss_driver.inference.config import PredictPipelineConfig
from rslp.forest_loss_driver.inference.extract_alerts import extract_alerts
from rslp.log_utils import get_logger

TEST_ID = str(uuid.uuid4())
logger = get_logger(__name__)


@pytest.fixture
def tiff_filename() -> str:
    """The path to the alert GeoTIFF file."""
    return "cropped_070W_10S_060W_00N.tif"


@pytest.fixture
def predict_pipeline_config() -> PredictPipelineConfig:
    """The configuration for the predict pipeline."""
    test_id = TEST_ID
    return PredictPipelineConfig(
        ds_root=UPath(
            f"gs://rslearn-eai/tests/{test_id}/datasets/forest_loss_driver/prediction/dataset_20241023/"
        ),
        workers=1,
        days=365,
        min_confidence=1,
        min_area=16.0,
    )


def test_extract_alerts(
    predict_pipeline_config: PredictPipelineConfig,
    tiff_filename: str,
    alert_tiffs_prefix: str,
    alert_date_tiffs_prefix: str,
    inference_dataset_config_path: str,
) -> None:
    """Tests extracting alerts from a single GeoTIFF file."""
    # TODO: Do not love this I would rather have the inference config either passed in or accessible within the package
    os.environ["INFERENCE_DATASET_CONFIG"] = inference_dataset_config_path
    os.environ["INDEX_CACHE_DIR"] = (
        "/Users/henryh/Desktop/eai-repos/rslearn_projects/data/henryh/rslearn_cache/"
    )
    os.environ["TILE_STORE_ROOT_DIR"] = (
        "/Users/henryh/Desktop/eai-repos/rslearn_projects/data/henryh/tile_store "
    )
    current_utc_time = datetime(2024, 10, 23, tzinfo=timezone.utc)
    extract_alerts(
        predict_pipeline_config,
        tiff_filename,
        alert_date_tiffs_prefix,
        alert_tiffs_prefix,
        current_utc_time,
    )

    # We should use the test_bucket for this or do in memory
    bucket = storage.Client().bucket(os.environ.get("TEST_BUCKET", "rslearn-eai"))
    # could also use a memory filesystem here to simplify the test

    # Assert one of the windows has all the info
    expected_image_blob = bucket.blob(
        f"tests/{TEST_ID}/datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/layers/mask/mask/image.png"
    )
    expected_info_json_blob = bucket.blob(
        f"tests/{TEST_ID}/datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/info.json"
    )
    expected_metadata_json_blob = bucket.blob(
        f"tests/{TEST_ID}/datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/metadata.json"
    )
    expected_image_metadata_json_blob = bucket.blob(
        f"tests/{TEST_ID}/datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/layers/mask/mask/metadata.json"
    )
    expected_completed_blob = bucket.blob(
        f"tests/{TEST_ID}/datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/layers/mask/completed"
    )
    expected_dataset_config_blob = bucket.blob(
        f"tests/{TEST_ID}/datasets/forest_loss_driver/prediction/dataset_20241023/config.json"
    )
    # add step looking for the config.json
    assert (
        expected_image_blob.exists()
    ), f"Blob {expected_image_blob.name} does not exist in bucket {bucket.name}"
    assert (
        expected_info_json_blob.exists()
    ), f"Blob {expected_info_json_blob.name} does not exist in bucket {bucket.name}"
    assert (
        expected_metadata_json_blob.exists()
    ), f"Blob {expected_metadata_json_blob.name} does not exist in bucket {bucket.name}"
    assert expected_image_metadata_json_blob.exists(), f"Blob {expected_image_metadata_json_blob.name} does not exist in bucket {bucket.name}"
    assert (
        expected_completed_blob.exists()
    ), f"Blob {expected_completed_blob.name} does not exist in bucket {bucket.name}"
    assert expected_dataset_config_blob.exists(), f"Blob {expected_dataset_config_blob.name} does not exist in bucket {bucket.name}"

    # # delete the test data
    # bucket.delete_blobs(
    #     blobs=[
    #         expected_image_blob,
    #         expected_info_json_blob,
    #         expected_metadata_json_blob,
    #         expected_image_metadata_json_blob,
    #         expected_completed_blob,
    #         expected_dataset_config_blob,
    #     ]
    # )
