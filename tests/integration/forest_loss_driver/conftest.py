"""Fixtures for the forest loss driver tests."""

import uuid
from collections.abc import Generator
from pathlib import Path
from unittest import mock

import pytest
from upath import UPath

# TODO: these are duplicated in the unit tests for forest loss driver
TEST_ID = str(uuid.uuid4())


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


@pytest.fixture
def model_cfg_fname() -> str:
    """The path to the model configuration file."""
    return str(
        Path(__file__).resolve().parents[3]
        # TODO: This should be hooked up to whatever the latest model is.
        / "data/forest_loss_driver/config_satlaspretrain_flip_oldmodel_unfreeze.yaml"
    )


@pytest.fixture(scope="session", autouse=True)
def clear_sys_argv() -> Generator[None, None, None]:
    """Clear the sys.argv."""
    with mock.patch("sys.argv", ["pytest"]):
        yield
