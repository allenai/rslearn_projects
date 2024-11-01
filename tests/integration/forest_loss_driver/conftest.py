"""Fixtures for the forest loss driver tests."""

from pathlib import Path

import pytest
from upath import UPath

# TODO: these are duplicated in the unit tests for forest loss driver


@pytest.fixture
def country_data_path() -> UPath:
    """Create a country data path."""
    return UPath(
        Path(__file__).parents[3]
        / "test_data/forest_loss_driver/artifacts/natural_earth_countries/20240830/20240830/ne_10m_admin_0_countries.shp"
    )


@pytest.fixture
def alert_tiffs_prefix() -> str:
    """The prefix for the alert GeoTIFF files with confidence data."""
    return str(Path(__file__).parents[3] / "test_data/forest_loss_driver/alert_tiffs")


@pytest.fixture
def alert_date_tiffs_prefix() -> str:
    """The prefix for the alert GeoTIFF files with date data."""
    return str(Path(__file__).parents[3] / "test_data/forest_loss_driver/alert_dates")


@pytest.fixture
def inference_dataset_config_path() -> str:
    """The path to the inference dataset config."""
    return str(
        Path(__file__).resolve().parents[3]
        / "data"
        / "forest_loss_driver"
        / "config.json"
    )


@pytest.fixture
def test_materialized_dataset_path() -> UPath:
    """The path to the test materialized dataset."""
    return UPath(
        Path(__file__).resolve().parents[3]
        / "test_data/forest_loss_driver/test_materialized_dataset/dataset_20241023"
    )
