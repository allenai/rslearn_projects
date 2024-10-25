"""Unit tests for the predict pipeline."""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
import shapely
import shapely.wkt
from rasterio.crs import CRS
from rslearn.utils import Projection, STGeometry
from upath import UPath

from rslp.forest_loss_driver.predict_pipeline import ForestLossEvent, write_event

SAMPLE_EVENT_FOLDER = Path("test_data/forest_loss_driver/sample_forest_loss_events")

FOLDER_PATH = Path(__file__).parents[3] / SAMPLE_EVENT_FOLDER
print(FOLDER_PATH)


@pytest.fixture
def forest_loss_event() -> ForestLossEvent:
    """Create a ForestLossEvent."""
    event_dict = {
        "polygon_geom": STGeometry(
            projection=Projection(
                crs=CRS.from_epsg(4326), x_resolution=1, y_resolution=1
            ),
            shp=shapely.wkt.loads(
                "POLYGON ((-69.98965 -4.27165, -69.98965 -4.27175, -69.98975 -4.27175, -69.98975 -4.271850000000001, -69.98985 -4.271850000000001, -69.98985 -4.27195, -69.98995 -4.27195, -69.98995 -4.271850000000001, -69.99005 -4.271850000000001, -69.99005 -4.27205, -69.98975 -4.27205, -69.98975 -4.27195, -69.98965 -4.27195, -69.98965 -4.271850000000001, -69.98955 -4.271850000000001, -69.98955 -4.27165, -69.98965 -4.27165))"
            ),
            time_range=None,
        ),
        "center_geom": STGeometry(
            projection=Projection(
                crs=CRS.from_epsg(4326), x_resolution=1, y_resolution=1
            ),
            shp=shapely.wkt.loads("POINT (-69.98985 -4.271850000000001)"),
            time_range=None,
        ),
        "center_pixel": (101, 42718),
        "ts": datetime.fromisoformat("2024-10-16T00:00:00+00:00"),
    }
    event = ForestLossEvent(**event_dict)
    return event


def test_write_event(forest_loss_event: ForestLossEvent) -> None:
    """Tests writing an event to a file."""

    with tempfile.TemporaryDirectory() as temp_dir:
        write_event(forest_loss_event, "test_filename.tif", UPath(temp_dir))

        expected_subdirectory = (
            "windows/default/feat_x_1281712_2146968_101_42718/layers/mask/mask"
        )
        assert (
            UPath(temp_dir) / expected_subdirectory / "image.png"
        ).exists(), "image.png not found"
        assert (
            UPath(temp_dir) / expected_subdirectory / "metadata.json"
        ).exists(), "metadata.json not found"
        with (UPath(temp_dir) / expected_subdirectory / "metadata.json").open() as f:
            metadata = json.load(f)
        assert metadata == {
            "bounds": [-815504, 49752, -815376, 49880]
        }, "metadata.json is incorrect"
