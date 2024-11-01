"""Integration tests for the predict pipeline.
This should be refactored along with the prediction file into seperate components.
"""

import os
import uuid
from collections.abc import Generator

import pytest
from google.cloud import storage

from rslp.log_utils import get_logger

TEST_ID = str(uuid.uuid4())
logger = get_logger(__name__)


@pytest.fixture
def tiff_filename() -> str:
    """The path to the alert GeoTIFF file."""
    return "cropped_070W_10S_060W_00N.tif"


# @pytest.fixture
# def predict_pipeline_config() -> PredictPipelineConfig:
#     """The configuration for the predict pipeline."""
#     test_id = TEST_ID
#     return PredictPipelineConfig(
#         ds_root=UPath(
#             f"gs://rslearn-eai/tests/{test_id}/datasets/forest_loss_driver/prediction/dataset_20241023/"
#         ),
#         workers=1,
#         days=365,
#         min_confidence=1,
#         min_area=16.0,
#     )


@pytest.fixture
def test_bucket() -> Generator[storage.Bucket, None, None]:
    """The test bucket."""
    bucket = storage.Client().bucket(os.environ.get("TEST_BUCKET", "rslearn-eai"))
    yield bucket


# We should have a test configuration for the predict pipeline or a test modification to some extent


# def test_select_best_images_pipeline(bucket: storage.Bucket) -> None:
#     """Tests the select best images pipeline."""
#     # need to create an example folder properly formatted for this pipeline locally
#     ds_path = UPath(
#         "gs://rslearn-eai/tests/f1af41d5-87e8-4431-ad5f-b03d48ad36a3/datasets/forest_loss_driver/prediction/dataset_20241023"
#     )
#     workers = 2
#     select_best_images_pipeline(ds_path, workers)
#     # assert the existence of a best times path
#     expected_best_times_blob = bucket.blob(
#         "tests/f1af41d5-87e8-4431-ad5f-b03d48ad36a3/datasets/forest_loss_driver/prediction/dataset_20241023/windows/default/feat_x_1281600_2146388_5_2221/best_times.json"
#     )
#     assert expected_best_times_blob.exists()
