"""Unit tests for the predict pipeline."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import shapely
import shapely.wkt
from affine import Affine
from rasterio.crs import CRS
from rslearn.utils import Projection, STGeometry
from upath import UPath

from rslp.forest_loss_driver.predict_pipeline import (
    ForestLossEvent,
    create_forest_loss_mask,
    load_country_polygon,
    read_forest_alerts_confidence_raster,
    read_forest_alerts_date_raster,
    write_event,
)

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


@pytest.fixture
def country_data_path() -> UPath:
    """Create a country data path."""
    return UPath(
        Path(__file__).parents[3]
        / "test_data/forest_loss_driver/artifacts/natural_earth_countries/20240830/20240830/ne_10m_admin_0_countries.shp"
    )


@pytest.fixture
def alert_tiffs_prefix() -> str:
    return str(Path(__file__).parents[3] / "test_data/forest_loss_driver/alert_tiffs")


@pytest.fixture
def alert_date_tiffs_prefix() -> str:
    return str(Path(__file__).parents[3] / "test_data/forest_loss_driver/alert_dates")


def test_write_event(forest_loss_event: ForestLossEvent) -> None:
    """Tests writing an event to a file."""

    with tempfile.TemporaryDirectory() as temp_dir:
        write_event(forest_loss_event, "test_filename.tif", UPath(temp_dir))

        expected_subdirectory = "windows/default/feat_x_1281712_2146968_101_42718/"
        assert (
            UPath(temp_dir)
            / expected_subdirectory
            / "layers"
            / "mask"
            / "mask"
            / "image.png"
        ).exists(), "image.png not found"

        assert (
            UPath(temp_dir) / expected_subdirectory / "metadata.json"
        ).exists(), "window metadata.json not found"

        with (
            UPath(temp_dir)
            / expected_subdirectory
            / "layers"
            / "mask"
            / "mask"
            / "metadata.json"
        ).open() as f:
            metadata = json.load(f)
        assert metadata == {
            "bounds": [-815504, 49752, -815376, 49880]
        }, "forest loss event metadata.json is incorrect"

        # assert completed file exists
        expected_layers_subdirectory = (
            "windows/default/feat_x_1281712_2146968_101_42718/layers/mask"
        )
        assert (
            UPath(temp_dir) / expected_layers_subdirectory / "completed"
        ).exists(), "completed file not found"


def test_load_country_polygon(country_data_path: UPath) -> None:
    """Tests loading the country polygon."""
    country_wgs84_shp = load_country_polygon(country_data_path)
    expected_type = shapely.geometry.multipolygon.MultiPolygon
    expected_centroid = shapely.geometry.point.Point(
        -74.37806457210715, -9.154388480752162
    )
    assert isinstance(
        country_wgs84_shp, expected_type
    ), f"country_wgs84_shp is not a {expected_type}"
    assert country_wgs84_shp.centroid.equals(expected_centroid)


def test_read_forest_alerts_confidence_raster(alert_tiffs_prefix: str) -> None:
    """Tests reading the forest alerts confidence raster."""
    fname = "cropped_070W_10S_060W_00N.tif"
    conf_data, conf_raster = read_forest_alerts_confidence_raster(
        fname,
        alert_tiffs_prefix,
    )
    assert conf_data.shape == (10000, 10000)
    assert conf_raster.profile == {
        "driver": "GTiff",
        "dtype": "uint8",
        "nodata": None,
        "width": 10000,
        "height": 10000,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(0.0001, 0.0, -70.0, 0.0, -0.0001, 0.0),
        "blockxsize": 10000,
        "blockysize": 1,
        "tiled": False,
        "compress": "lzw",
        "interleave": "band",
    }


def test_read_forest_alerts_date_raster(alert_date_tiffs_prefix: str) -> None:
    """Tests reading the forest alerts date raster."""
    fname = "cropped_070W_10S_060W_00N.tif"
    date_data, date_raster = read_forest_alerts_date_raster(
        fname, alert_date_tiffs_prefix
    )
    assert date_data.shape == (10000, 10000)
    assert date_raster.profile == {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": None,
        "width": 10000,
        "height": 10000,
        "count": 1,
        "crs": CRS.from_epsg(4326),
        "transform": Affine(0.0001, 0.0, -70.0, 0.0, -0.0001, 0.0),
        "blockxsize": 10000,
        "blockysize": 1,
        "tiled": False,
        "compress": "lzw",
        "interleave": "band",
    }


def test_create_forest_loss_mask() -> None:
    """Tests creating the forest loss mask."""
    conf_data = np.zeros((10, 10), dtype=np.uint8)
    date_data = 2124 * np.ones(
        (10, 10), dtype=np.uint16
    )  # Equivalent to all data is from 5 days ago
    # Set specific elements to nonzero
    conf_data[5:7, 5:7] = 2  # Example region of confidence data
    min_days = 6
    min_confidence = 1

    expected_mask = np.zeros((10, 10), dtype=np.uint8)
    expected_mask[5:7, 5:7] = 1
    mask = create_forest_loss_mask(
        conf_data,
        date_data,
        min_confidence,
        min_days,
        datetime.now(timezone.utc),
    )
    assert np.all(mask == expected_mask)
    date_data_2 = 2120 * np.ones(
        (10, 10), dtype=np.uint16
    )  # Equivalent to all data is from 9 days ago
    mask_2 = create_forest_loss_mask(
        conf_data,
        date_data_2,
        min_confidence,
        min_days,
        datetime.now(timezone.utc),
    )
    assert np.all(mask_2 == 0)
