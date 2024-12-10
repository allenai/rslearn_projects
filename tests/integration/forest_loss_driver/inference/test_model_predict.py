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
        with output_path.open("r") as f:
            output_json = json.load(f)
        assert output_json == expected_output_json
