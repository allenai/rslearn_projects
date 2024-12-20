"""Integration test for the model predict step for the forest loss driver inference pipeline."""

import json
import multiprocessing
import os
import shutil
import tempfile
import uuid

import pytest
from upath import UPath

from rslp.forest_loss_driver.inference import (
    forest_loss_driver_model_predict,
    select_least_cloudy_images_pipeline,
)
from rslp.forest_loss_driver.inference.config import (
    ModelPredictArgs,
    SelectLeastCloudyImagesArgs,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)


@pytest.fixture
def model_predict_args(model_cfg_fname: str) -> ModelPredictArgs:
    """The arguments for the model_predict step."""
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    return ModelPredictArgs(
        model_cfg_fname=model_cfg_fname,
        data_load_workers=num_workers,
    )


# Why are model outputs nto stable in different envs
def test_forest_loss_driver_model_predict(
    test_materialized_dataset_path: UPath,
    model_predict_args: ModelPredictArgs,
) -> None:
    # This should probably be a secret on Beaker.
    os.environ["RSLP_PREFIX"] = "gs://rslearn-eai"
    # materialized dataset path
    with tempfile.TemporaryDirectory(prefix=f"test_{uuid.uuid4()}_") as temp_dir:
        shutil.copytree(test_materialized_dataset_path, temp_dir, dirs_exist_ok=True)
        # Set up Materialized dataset for best times
        select_least_cloudy_images_pipeline(
            UPath(temp_dir), SelectLeastCloudyImagesArgs()
        )
        # Run model predict
        forest_loss_driver_model_predict(
            UPath(temp_dir),
            model_predict_args,
        )
        output_path = (
            UPath(temp_dir)
            / "windows"
            / "default"
            / "feat_x_1281600_2146388_5_2221"
            / "layers"
            / "output"
            / "data.geojson"
        )
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
                            0.00027457400574348867,
                            9.164694347418845e-06,
                            0.004422641359269619,
                            7.985765826390434e-09,
                            1.6661474546708632e-06,
                            1.7722986740409397e-05,
                            2.0580247905854776e-07,
                            2.0334262273991044e-08,
                            0.9876694083213806,
                            0.007604612968862057,
                        ],
                    },
                    "geometry": {"type": "Point", "coordinates": [-815616.0, 49172.0]},
                }
            ],
        }

        with output_path.open("r") as f:
            output_json = json.load(f)
        # TODO: Ideally we would have a pydantic model for this output perhaps that we could subclass from rslearn?
        # Check properties except probs
        tol = 0.01
        assert output_json["type"] == expected_output_json["type"]
        assert output_json["properties"] == expected_output_json["properties"]
        assert len(output_json["features"]) == len(expected_output_json["features"])
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

        # Check probs are within 0.001
        actual_probs = output_json["features"][0]["properties"]["probs"]  # type: ignore
        expected_probs = expected_output_json["features"][0]["properties"]["probs"]  # type: ignore
        assert len(actual_probs) == len(expected_probs)
        for actual, expected in zip(actual_probs, expected_probs):
            assert (
                abs(actual - expected) < tol
            ), f"Probability difference {abs(actual - expected)} exceeds threshold {tol}"
