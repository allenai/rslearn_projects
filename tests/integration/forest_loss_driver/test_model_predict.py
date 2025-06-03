"""Integration test for the model predict step for the forest loss driver inference pipeline."""

import json
import pathlib
import shutil
from typing import Any

from upath import UPath

from rslp.forest_loss_driver.extract_dataset.least_cloudy_image_selector import (
    SelectLeastCloudyImagesArgs,
    select_least_cloudy_images_pipeline,
)
from rslp.forest_loss_driver.predict_pipeline import MODEL_CFG_FNAME
from rslp.forest_loss_driver.train import CATEGORIES
from rslp.log_utils import get_logger
from rslp.utils.rslearn import run_model_predict

logger = get_logger(__name__)


# Why are model outputs nto stable in different envs
def test_forest_loss_driver_model_predict(
    test_materialized_dataset_path: UPath, tmp_path: pathlib.Path
) -> None:
    # materialized dataset path
    shutil.copytree(test_materialized_dataset_path, tmp_path, dirs_exist_ok=True)
    # Set up Materialized dataset for best times
    select_least_cloudy_images_pipeline(UPath(tmp_path), SelectLeastCloudyImagesArgs())
    # Run model predict
    run_model_predict(MODEL_CFG_FNAME, UPath(tmp_path))
    output_path = (
        UPath(tmp_path)
        / "windows"
        / "default"
        / "feat_x_1281600_2146388_5_2221"
        / "layers"
        / "output"
        / "data.geojson"
    )
    expected_output_json: dict[str, Any] = {
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
                },
                "geometry": {"type": "Point", "coordinates": [-815616.0, 49172.0]},
            }
        ],
    }

    with output_path.open("r") as f:
        output_json = json.load(f)
    # TODO: Ideally we would have a pydantic model for this output perhaps that we could subclass from rslearn?
    # Check properties except probs
    assert output_json["type"] == expected_output_json["type"]
    assert output_json["properties"] == expected_output_json["properties"]
    assert len(output_json["features"]) == len(expected_output_json["features"])
    assert (
        output_json["features"][0]["type"]
        == expected_output_json["features"][0]["type"]
    )
    assert (
        output_json["features"][0]["geometry"]
        == expected_output_json["features"][0]["geometry"]
    )
    assert (
        output_json["features"][0]["properties"]["new_label"]
        == expected_output_json["features"][0]["properties"]["new_label"]
    )

    # Ensure river class is at least 0.9 probability and others at most 0.05.
    actual_probs = output_json["features"][0]["properties"]["probs"]
    expected_category = expected_output_json["features"][0]["properties"]["new_label"]
    assert len(actual_probs) == len(CATEGORIES)
    for prob, category_name in zip(actual_probs, CATEGORIES):
        if category_name == expected_category:
            assert prob >= 0.9, f"Probability for category {category_name} < 0.9"
        else:
            assert prob <= 0.05, f"Probability for category {category_name} > 0.05"
