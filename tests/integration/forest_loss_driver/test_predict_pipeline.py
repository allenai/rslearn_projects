"""Integration tests for the predict pipeline.
This should be refactored along with the prediction file into seperate components.
"""

import os
import tempfile
import uuid
from collections.abc import Generator
from datetime import datetime, timezone

import pytest
from google.cloud import storage
from upath import UPath

from rslp.forest_loss_driver.inference.config import PredictPipelineConfig
from rslp.forest_loss_driver.inference.materialize_dataset import (
    VISUALIZATION_ONLY_LAYERS,
)
from rslp.forest_loss_driver.predict_pipeline import predict_pipeline
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


@pytest.fixture
def test_bucket() -> Generator[storage.Bucket, None, None]:
    """The test bucket."""
    bucket = storage.Client().bucket(os.environ.get("TEST_BUCKET", "rslearn-eai"))
    yield bucket


# lightning cli is pretty brittle can't have any other sys args sent in and it
# prevents programmatic pytest usage
# Need to limit number of windows to 1
def test_predict_pipeline(
    predict_pipeline_config: PredictPipelineConfig,
    inference_dataset_config_path: str,
    alert_tiffs_prefix: str,
    alert_date_tiffs_prefix: str,
    tiff_filename: str,
) -> None:
    """Test the predict pipeline."""
    with tempfile.TemporaryDirectory(prefix=f"test_{TEST_ID}_") as temp_dir:
        ds_path = UPath(temp_dir) / "dataset_20241023"
        index_cache_dir = UPath(temp_dir) / "index_cache"
        tile_store_root_dir = UPath(temp_dir) / "tile_store"
        predict_pipeline_config = PredictPipelineConfig(
            ds_root=ds_path,
            workers=1,
            days=365,
            min_confidence=1,
            min_area=16.0,
            conf_prefix=alert_tiffs_prefix,
            date_prefix=alert_date_tiffs_prefix,
            prediction_utc_time=datetime(2024, 10, 23, tzinfo=timezone.utc),
            disabled_layers=VISUALIZATION_ONLY_LAYERS,
        )
        os.environ["INFERENCE_DATASET_CONFIG"] = inference_dataset_config_path
        # Need to make these both temp dirs  also would want the predict_pipeline path to have a temp dir
        os.environ["INDEX_CACHE_DIR"] = str(index_cache_dir)
        os.environ["TILE_STORE_ROOT_DIR"] = str(tile_store_root_dir)
        os.environ["RSLP_PREFIX"] = "gs://rslearn-eai"  # make this a secret
        predict_pipeline(
            predict_pipeline_config, inference_dataset_config_path, [tiff_filename]
        )
        # Make sure we are only using the cropped image as input
        # make sure we are deleting extra widnows files?
