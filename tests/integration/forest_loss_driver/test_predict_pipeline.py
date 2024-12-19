"""Integration tests for the predict pipeline."""

import json
import multiprocessing
import os
import tempfile
import uuid
from collections.abc import Generator
from datetime import datetime, timezone

import pytest
from google.cloud import storage
from upath import UPath

from rslp.forest_loss_driver.inference.config import (
    ExtractAlertsArgs,
    ModelPredictArgs,
    PredictPipelineConfig,
)
from rslp.forest_loss_driver.predict_pipeline import ForestLossDriverPredictionPipeline
from rslp.log_utils import get_logger

TEST_ID = str(uuid.uuid4())
logger = get_logger(__name__)


@pytest.fixture
def tiff_filename() -> str:
    """The path to the alert GeoTIFF file."""
    return "cropped_070W_10S_060W_00N.tif"


@pytest.fixture
def test_bucket() -> Generator[storage.Bucket, None, None]:
    """The test bucket."""
    # TODO: Fix this
    bucket = storage.Client().bucket(os.environ.get("TEST_BUCKET", "rslearn-eai"))
    yield bucket


@pytest.fixture
def predict_pipeline_config(
    inference_dataset_config_path: str,
    model_cfg_fname: str,
    alert_tiffs_prefix: str,
    alert_date_tiffs_prefix: str,
    tiff_filename: str,
) -> PredictPipelineConfig:
    """The predict pipeline config."""
    # TODO: make this not an env var

    os.environ["INFERENCE_DATASET_CONFIG"] = inference_dataset_config_path
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    predict_pipeline_config = PredictPipelineConfig(
        model_predict_args=ModelPredictArgs(model_cfg_fname=model_cfg_fname),
        extract_alerts_args=ExtractAlertsArgs(
            gcs_tiff_filenames=[tiff_filename],
            workers=num_workers,
            min_confidence=1,
            min_area=16.0,
            conf_prefix=alert_tiffs_prefix,
            date_prefix=alert_date_tiffs_prefix,
            prediction_utc_time=datetime(2024, 10, 23, tzinfo=timezone.utc),
            max_number_of_events=1,
        ),
    )
    predict_pipeline_config.set_num_workers_for_all_steps(num_workers)
    return predict_pipeline_config


def test_predict_pipeline(
    predict_pipeline_config: PredictPipelineConfig,
) -> None:
    """Test the predict pipeline."""
    with tempfile.TemporaryDirectory(prefix=f"test_{TEST_ID}_") as temp_dir:
        ds_path = UPath(temp_dir) / "dataset_20241023"
        index_cache_dir = UPath(temp_dir) / "index_cache"
        tile_store_root_dir = UPath(temp_dir) / "tile_store"
        predict_pipeline_config.ds_root = ds_path
        os.environ["INDEX_CACHE_DIR"] = str(index_cache_dir)
        os.environ["TILE_STORE_ROOT_DIR"] = str(tile_store_root_dir)
        if "RSLP_PREFIX" not in os.environ:
            raise OSError(
                "RSLP_PREFIX must be set in the environment for the test bucket"
            )
        prediction_pipeline = ForestLossDriverPredictionPipeline(
            pred_pipeline_config=predict_pipeline_config
        )
        prediction_pipeline.extract_dataset()
        prediction_pipeline.run_model_predict()
        # assert that the output files exist
        output_path = (
            UPath(temp_dir)
            / "dataset_20241023"
            / "windows"
            / "default"
            / "feat_x_1281600_2146388_5_2221"
            / "layers"
            / "output"
            / "data.geojson"
        )
        # TODO: Make a pydantic model for this output
        expected_output_json = {
            "type": "FeatureCollection",
            "properties": {
                "crs": "EPSG:3857",
                "x_resolution": 9.554628535647032,
                "y_resolution": -9.554628535647032,
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "new_label": "river",
                        "probs": [
                            0.0003058495349250734,
                            8.582491318520624e-06,
                            0.0032749103847891092,
                            7.261763190768988e-09,
                            1.4538236428052187e-06,
                            1.59567116497783e-05,
                            1.8003102297825535e-07,
                            2.3082723288325724e-08,
                            0.989401638507843,
                            0.0069913845509290695,
                        ],
                    },
                    "geometry": {"type": "Point", "coordinates": [-815616.0, 49172.0]},
                }
            ],
        }
        logger.info(f"Expected output: {expected_output_json}")
        with output_path.open("r") as f:
            output_json = json.load(f)
        tol = 0.1
        # Check everything except probs
        assert output_json["type"] == expected_output_json["type"]  # type: ignore
        assert output_json["properties"] == expected_output_json["properties"]  # type: ignore
        assert len(output_json["features"]) == len(expected_output_json["features"])  # type: ignore
        assert (
            output_json["features"][0]["type"]  # type: ignore
            == expected_output_json["features"][0]["type"]  # type: ignore
        )
        assert (
            output_json["features"][0]["geometry"]  # type: ignore
            == expected_output_json["features"][0]["geometry"]  # type: ignore
        )
        assert (
            output_json["features"][0]["properties"]["new_label"]  # type: ignore
            == expected_output_json["features"][0]["properties"]["new_label"]  # type: ignore
        )

        # Check probs are within 0.1
        actual_probs = output_json["features"][0]["properties"]["probs"]  # type: ignore
        expected_probs = expected_output_json["features"][0]["properties"]["probs"]  # type: ignore
        assert len(actual_probs) == len(expected_probs)
        for actual, expected in zip(actual_probs, expected_probs):
            assert (
                abs(actual - expected) < tol
            ), f"Probability difference {abs(actual - expected)} exceeds threshold {tol}"


def test_config_load() -> None:
    """Test the config loading for the forest loss driver inference pipeline."""
    pipeline = ForestLossDriverPredictionPipeline()
    assert isinstance(pipeline.pred_config, PredictPipelineConfig)
