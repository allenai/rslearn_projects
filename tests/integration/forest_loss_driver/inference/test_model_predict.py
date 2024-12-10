"""Integration test for the model predict step for the forest loss driver inference pipeline."""

import json
import os
import shutil
import tempfile
import uuid

from upath import UPath

from rslp.forest_loss_driver.inference.model_predict import (
    forest_loss_driver_model_predict,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)


# Why are model outputs nto stable in different envs
def test_forest_loss_driver_model_predict(
    test_materialized_dataset_path: UPath,
    model_cfg_fname: str,
) -> None:
    # This should probably be a secret on Beaker.
    os.environ["RSLP_PREFIX"] = "gs://rslearn-eai"
    # materialized dataset path
    with tempfile.TemporaryDirectory(prefix=f"test_{uuid.uuid4()}_") as temp_dir:
        shutil.copytree(test_materialized_dataset_path, temp_dir, dirs_exist_ok=True)
        forest_loss_driver_model_predict(
            model_cfg_fname,
            UPath(temp_dir),
            model_data_load_workers=1,
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
        logger.info(f"Expected output: {expected_output_json}")
        with output_path.open("r") as f:
            output_json = json.load(f)
        assert output_json == expected_output_json
