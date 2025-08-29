"""Integration tests for the best image selector."""

import shutil
from pathlib import Path

from upath import UPath

from rslp.forest_loss_driver.extract_dataset.least_cloudy_image_selector import (
    SelectLeastCloudyImagesArgs,
    select_least_cloudy_images_pipeline,
)


def test_select_least_cloudy_images_pipeline(
    test_materialized_dataset_path: UPath,
    tmp_path: Path,
) -> None:
    """Test the least cloudy image selector pipeline."""
    shutil.copytree(test_materialized_dataset_path, tmp_path, dirs_exist_ok=True)
    select_least_cloudy_images_pipeline(UPath(tmp_path), SelectLeastCloudyImagesArgs())
    least_cloudy_times_path = (
        tmp_path
        / "windows"
        / "default"
        / "feat_x_1281600_2146388_5_2221"
        / "least_cloudy_times.json"
    )
    assert least_cloudy_times_path.exists(), f"{least_cloudy_times_path} does not exist"
