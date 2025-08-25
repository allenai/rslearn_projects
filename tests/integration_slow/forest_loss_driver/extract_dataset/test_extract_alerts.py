"""Integration tests for extract_alerts.py step of the inference pipeline."""

import pathlib
import uuid
from datetime import datetime, timezone

import shapely
import shapely.wkt
from upath import UPath

from rslp.forest_loss_driver.extract_dataset.extract_alerts import (
    ExtractAlertsArgs,
    extract_alerts,
    load_country_polygon,
    read_forest_alerts_confidence_raster,
    read_forest_alerts_date_raster,
)
from rslp.log_utils import get_logger

TEST_ID = str(uuid.uuid4())
logger = get_logger(__name__)


def test_read_forest_alerts_confidence_raster(
    alert_tiffs_prefix: str, tiff_filename: str
) -> None:
    """Tests reading the forest alerts confidence raster."""
    conf_data, _ = read_forest_alerts_confidence_raster(
        tiff_filename,
        alert_tiffs_prefix,
    )
    assert conf_data.shape == (10, 10)


def test_read_forest_alerts_date_raster(
    alert_date_tiffs_prefix: str, tiff_filename: str
) -> None:
    """Tests reading the forest alerts date raster."""
    date_data, _ = read_forest_alerts_date_raster(
        tiff_filename, alert_date_tiffs_prefix
    )
    assert date_data.shape == (10, 10)


def test_load_country_polygon(country_data_path: UPath) -> None:
    """Tests loading the country polygon."""
    # This data is dynamically loaded from gcs in conftest.py
    country_wgs84_shp = load_country_polygon(country_data_path, ["PE"])
    # Check that a few points collected via Google Maps are correct.
    peru_points = [
        shapely.Point(-74.529119, -6.463668),
        shapely.Point(-70.367139, -17.281150),
    ]
    not_peru_points = [
        shapely.Point(-67.601992, -14.821313),
        shapely.Point(-76.684202, -2.048203),
        shapely.Point(-107.931170, -18.224945),
    ]
    for point in peru_points:
        assert country_wgs84_shp.contains(point)
    for point in not_peru_points:
        assert not country_wgs84_shp.contains(point)


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
    extract_alerts(ds_root, extract_alerts_args)

    # Assert one of the windows has all the info
    window_dir = ds_root / "windows" / "default" / "feat_x_1281601_2146388_4_5"
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
