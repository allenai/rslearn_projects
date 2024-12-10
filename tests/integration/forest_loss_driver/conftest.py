"""Fixtures for the forest loss driver tests.

Note: All data is downloaded to the test_data directory in the root of the repo at the
start of the testing session
"""

import uuid
from collections.abc import Generator
from pathlib import Path
from unittest import mock

import pytest
from upath import UPath

from rslp.log_utils import get_logger

# TODO: these are duplicated in the unit tests for forest loss driver
TEST_ID = str(uuid.uuid4())

logger = get_logger(__name__)


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


@pytest.fixture(scope="session", autouse=True)
def download_test_data() -> Generator[None, None, None]:
    """Download test data from GCS bucket if not present locally."""
    test_data_path = Path(__file__).parents[3] / "test_data/forest_loss_driver"
    gcs_path = (
        "gs://test-bucket-rslearn/forest_loss_driver/test_data/forest_loss_driver"
    )

    # Define which folders we want to download
    folders_to_download = ["test_materialized_dataset", "alert_dates", "alert_tiffs"]

    # Create test data directory if it doesn't exist
    test_data_path.mkdir(parents=True, exist_ok=True)

    # Download each required folder
    gcs_upath = UPath(gcs_path)
    for folder in folders_to_download:
        folder_path = test_data_path / folder
        if not folder_path.exists() or not any(folder_path.iterdir()):
            logger.info(f"Downloading {folder} from GCS...")
            folder_gcs_path = gcs_upath / folder

            for src_path in folder_gcs_path.rglob("*"):
                if src_path.is_file():
                    rel_path = Path(*src_path.relative_to(gcs_upath).parts[4:])
                    dst_path = test_data_path / rel_path
                    logger.info(f"Downloading {src_path} to {dst_path}")
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    with src_path.open("rb") as src, dst_path.open("wb") as dst:
                        dst.write(src.read())

            logger.info(f"Finished downloading {folder}")

    # Log contents of test data folder
    logger.info("\nTest data directory contents:")
    for path in sorted(test_data_path.rglob("*")):
        logger.info(f"  {path.relative_to(test_data_path)}")
    yield
