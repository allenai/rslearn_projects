"""Unit tests for the predict pipeline."""

from datetime import datetime, timezone

import numpy as np
import pytest
import shapely
import shapely.wkt
from rasterio.crs import CRS
from rslearn.utils import Projection, STGeometry

from rslp.forest_loss_driver.extract_dataset.extract_alerts import (
    ForestLossEvent,
    create_forest_loss_mask,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)


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


def test_create_forest_loss_mask_events_found() -> None:
    """Tests creating the forest loss mask."""
    conf_data = np.zeros((10, 10), dtype=np.uint8)
    date_data = 2148 * np.ones(
        (10, 10), dtype=np.uint16
    )  # Equivalent to all data is from 5 days ago
    # Set specific elements to nonzero
    conf_data[5:7, 5:7] = 2  # Example region of confidence data
    days_to_look_back = 6
    min_confidence = 1

    expected_mask = np.zeros((10, 10), dtype=np.uint8)
    expected_mask[5:7, 5:7] = 1
    mask = create_forest_loss_mask(
        conf_data,
        date_data,
        min_confidence,
        days_to_look_back,
        datetime(2024, 11, 23, tzinfo=timezone.utc),
    )
    assert np.all(mask == expected_mask)


def test_create_forest_loss_mask_events_not_found() -> None:
    """Tests creating the forest loss mask."""
    days_to_look_back = 6
    min_confidence = 1

    conf_data = np.zeros((10, 10), dtype=np.uint8)
    date_data_2 = 2120 * np.ones(
        (10, 10), dtype=np.uint16
    )  # Equivalent to all data is from 9 days ago
    mask_2 = create_forest_loss_mask(
        conf_data,
        date_data_2,
        min_confidence,
        days_to_look_back,
        datetime(2024, 11, 23, tzinfo=timezone.utc),
    )
    assert np.all(mask_2 == np.zeros_like(mask_2))
