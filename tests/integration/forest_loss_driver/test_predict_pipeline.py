"""Integration tests for the predict pipeline."""

import json
import pathlib
import uuid
from datetime import UTC, datetime

from upath import UPath

from rslp.forest_loss_driver.extract_dataset import (
    INFERENCE_LAYERS,
    VISUALIZATION_ONLY_LAYERS,
    ExtractAlertsArgs,
    VisLayerMaterializeArgs,
    extract_dataset,
)
from rslp.forest_loss_driver.predict_pipeline import predict_pipeline
from rslp.log_utils import get_logger

TEST_ID = str(uuid.uuid4())
logger = get_logger(__name__)


def test_predict_pipeline(
    tmp_path: pathlib.Path,
    alert_tiffs_prefix: str,
    alert_date_tiffs_prefix: str,
    tiff_filename: str,
) -> None:
    """Test the predict pipeline."""
    ds_path = UPath(tmp_path)
    # Don't materialize the planet layers, which uses Planet quota.
    vis_materialize_args = VisLayerMaterializeArgs(
        disabled_layers=INFERENCE_LAYERS + VISUALIZATION_ONLY_LAYERS
    )
    extract_dataset(
        ds_path,
        extract_alerts_args=ExtractAlertsArgs(
            gcs_tiff_filenames=[tiff_filename],
            conf_prefix=alert_tiffs_prefix,
            date_prefix=alert_date_tiffs_prefix,
            prediction_utc_time=datetime(2024, 10, 23, tzinfo=UTC),
            max_number_of_events=1,
            min_confidence=1,
            min_area=16.0,
        ),
        vis_materialize_args=vis_materialize_args,
    )
    predict_pipeline(ds_path)

    # assert that the output files exist
    output_path = (
        ds_path
        / "windows"
        / "default"
        / "feat_x_1281601_2146388_4_5"
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
                    "probs": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                },
                "geometry": {"type": "Point", "coordinates": [-815615.0, 49172.0]},
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
