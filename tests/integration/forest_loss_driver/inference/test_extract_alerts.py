"""Integration tests for extract_alerts.py step of the inference pipeline."""

import pathlib
import uuid
from datetime import datetime, timezone

import pytest
import shapely
import shapely.wkt
from affine import Affine
from rasterio.crs import CRS
from upath import UPath

from rslp.forest_loss_driver.inference.config import ExtractAlertsArgs
from rslp.forest_loss_driver.inference.extract_alerts import (
    extract_alerts_pipeline,
    load_country_polygon,
    read_forest_alerts_confidence_raster,
    read_forest_alerts_date_raster,
)
from rslp.log_utils import get_logger

TEST_ID = str(uuid.uuid4())
logger = get_logger(__name__)


@pytest.fixture
def tiff_filename() -> str:
    """The path to the alert GeoTIFF file."""
    logger.warning("This tif is on GCS and is downloaded in conftest.py")
    return "cropped_070W_10S_060W_00N.tif"


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
        "transform": Affine(0.0001, 0.0, -70.0, 0.0, -0.0001, -4.0),
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
        "transform": Affine(0.0001, 0.0, -70.0, 0.0, -0.0001, -4.0),
        "blockxsize": 10000,
        "blockysize": 1,
        "tiled": False,
        "compress": "lzw",
        "interleave": "band",
    }


def test_load_country_polygon(country_data_path: UPath) -> None:
    """Tests loading the country polygon."""
    # This data is dynamically loaded from gcs in conftest.py
    country_wgs84_shp = load_country_polygon(country_data_path)
    expected_type = shapely.geometry.multipolygon.MultiPolygon
    expected_centroid = shapely.geometry.point.Point(
        -74.37806457210715, -9.154388480752162
    )
    assert isinstance(
        country_wgs84_shp, expected_type
    ), f"country_wgs84_shp is not a {expected_type}"
    assert country_wgs84_shp.centroid.equals(expected_centroid)


def test_extract_alerts(
    tiff_filename: str,
    alert_tiffs_prefix: str,
    alert_date_tiffs_prefix: str,
    country_data_path: UPath,
    tmp_path: pathlib.Path,
) -> None:
    """Tests extracting alerts from a single GeoTIFF file."""
    ds_root = (
        UPath(tmp_path)
        / "datasets"
        / "forest_loss_driver"
        / "prediction"
        / "dataset_20241023"
    )
    extract_alerts_args = ExtractAlertsArgs(
        gcs_tiff_filenames=[tiff_filename],
        workers=1,
        days=365,
        min_confidence=1,
        min_area=16.0,
        conf_prefix=alert_tiffs_prefix,
        date_prefix=alert_date_tiffs_prefix,
        prediction_utc_time=datetime(2024, 10, 23, tzinfo=timezone.utc),
        country_data_path=country_data_path,
    )
    extract_alerts_pipeline(ds_root, extract_alerts_args)

    # Assert one of the windows has all the info
    window_dir = ds_root / "windows" / "default" / "feat_x_1281600_2146388_5_2221"
    expected_image_path = window_dir / "layers" / "mask" / "mask" / "image.png"
    expected_info_json_path = window_dir / "info.json"
    expected_metadata_json_path = window_dir / "metadata.json"
    expected_image_metadata_json_path = (
        window_dir / "layers" / "mask" / "mask" / "metadata.json"
    )
    expected_completed_path = window_dir / "layers" / "mask" / "completed"
    expected_dataset_config_path = ds_root / "config.json"
    # add step looking for the config.json
    assert expected_image_path.exists(), f"Path {expected_image_path} does not exist"
    assert (
        expected_info_json_path.exists()
    ), f"Path {expected_info_json_path} does not exist"
    assert (
        expected_metadata_json_path.exists()
    ), f"Path {expected_metadata_json_path} does not exist"
    assert (
        expected_image_metadata_json_path.exists()
    ), f"Path {expected_image_metadata_json_path} does not exist"
    assert (
        expected_completed_path.exists()
    ), f"Path {expected_completed_path} does not exist"
    assert (
        expected_dataset_config_path.exists()
    ), f"Path {expected_dataset_config_path} does not exist"
