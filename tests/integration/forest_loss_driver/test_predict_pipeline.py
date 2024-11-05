"""Integration tests for the predict pipeline.
This should be refactored along with the prediction file into seperate components.
"""

import json
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
# We need to have a predict config that directly loads into the predict pipeline so that I can easily set up the pipeline
# only on a cropped image for a single window
# Issue is we have these required args that are needed for the predict pipeline
# But they are not obviously needed for the predict pipeline
# We should have an inference mode for predict?
def test_predict_pipeline(
    predict_pipeline_config: PredictPipelineConfig,
    inference_dataset_config_path: str,
    model_cfg_fname: str,
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
            max_number_of_events=1,
            disabled_layers=VISUALIZATION_ONLY_LAYERS,
        )
        os.environ["INFERENCE_DATASET_CONFIG"] = inference_dataset_config_path
        # Need to make these both temp dirs  also would want the predict_pipeline path to have a temp dir
        os.environ["INDEX_CACHE_DIR"] = str(index_cache_dir)
        os.environ["TILE_STORE_ROOT_DIR"] = str(tile_store_root_dir)
        os.environ["RSLP_PREFIX"] = "gs://rslearn-eai"  # make this a secret
        predict_pipeline(predict_pipeline_config, model_cfg_fname, [tiff_filename])
        # assert that the output files exist
        output_path = (
            UPath(temp_dir)
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
            "properties": {
                "crs": "EPSG:3857",
                "x_resolution": 9.554628535647032,
                "y_resolution": -9.554628535647032,
            },
        }
        with output_path.open("r") as f:
            output_json = json.load(f)
        assert output_json == expected_output_json
