"""Integration tests for the predict pipeline."""

import json
import os
import pathlib
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from google.cloud import storage
from upath import UPath

from rslp.forest_loss_driver.inference.config import (
    ExtractAlertsArgs,
    PredictPipelineConfig,
)
from rslp.forest_loss_driver.predict_pipeline import ForestLossDriverPredictionPipeline
from rslp.log_utils import get_logger
from rslp.main import main

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
def predict_pipeline_config_path() -> str:
    """The path to the config file used for inference."""
    return "rslp/forest_loss_driver/inference/config/forest_loss_driver_predict_pipeline_config.yaml"


@pytest.fixture
def predict_pipeline_config(
    inference_dataset_config_path: str,
    alert_tiffs_prefix: str,
    alert_date_tiffs_prefix: str,
    tiff_filename: str,
) -> PredictPipelineConfig:
    """The predict pipeline config."""
    predict_pipeline_config = PredictPipelineConfig(
        extract_alerts_args=ExtractAlertsArgs(
            gcs_tiff_filenames=[tiff_filename],
            min_confidence=1,
            min_area=16.0,
            conf_prefix=alert_tiffs_prefix,
            date_prefix=alert_date_tiffs_prefix,
            prediction_utc_time=datetime(2024, 10, 23, tzinfo=timezone.utc),
            max_number_of_events=1,
        ),
    )
    return predict_pipeline_config


def test_predict_pipeline(
    predict_pipeline_config: PredictPipelineConfig,
    tmp_path: pathlib.Path,
) -> None:
    """Test the predict pipeline."""
    ds_path = UPath(tmp_path) / "dataset_20241023"
    predict_pipeline_config.ds_root = ds_path
    prediction_pipeline = ForestLossDriverPredictionPipeline(
        pred_pipeline_config=predict_pipeline_config
    )
    prediction_pipeline.extract_dataset()
    prediction_pipeline.run_model_predict()
    # assert that the output files exist
    output_path = (
        ds_path
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


def test_forest_loss_driver_predict_cli_config_load(
    predict_pipeline_config_path: str,
) -> None:
    def assert_config(pred_pipeline_config: PredictPipelineConfig) -> bool:
        # Verify the config is the correct type
        logger.info(f"Pred pipeline config: {pred_pipeline_config}")
        assert isinstance(pred_pipeline_config, PredictPipelineConfig)
        return True

    with (
        patch(
            "sys.argv",
            [
                "rslp",
                "forest_loss_driver",
                "predict",
                "--pred_pipeline_config",
                predict_pipeline_config_path,
            ],
        ),
        patch("rslp.forest_loss_driver.workflows", {"predict": assert_config}),
    ):
        main()
