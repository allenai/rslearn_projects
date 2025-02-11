"""Integration tests for the best image selector."""

import shutil
from pathlib import Path

import pytest
from upath import UPath

from rslp.forest_loss_driver.inference import select_least_cloudy_images_pipeline
from rslp.forest_loss_driver.inference.config import SelectLeastCloudyImagesArgs


@pytest.fixture
def select_least_cloudy_images_args() -> SelectLeastCloudyImagesArgs:
    """The arguments for the select_least_cloudy_images step."""
    return SelectLeastCloudyImagesArgs()


def test_select_least_cloudy_images_pipeline(
    test_materialized_dataset_path: UPath,
    select_least_cloudy_images_args: SelectLeastCloudyImagesArgs,
    tmp_path: Path,
) -> None:
    """Test the least cloudy image selector pipeline."""
    # Want to make sure we have the best times for each layer?
    # Will all layers in config be present? Is there the case of no best images? What is expected behavior?
    shutil.copytree(test_materialized_dataset_path, tmp_path, dirs_exist_ok=True)
    select_least_cloudy_images_pipeline(
        UPath(tmp_path), select_least_cloudy_images_args
    )
    least_cloudy_times_path = (
        tmp_path
        / "windows"
        / "default"
        / "feat_x_1281600_2146388_5_2221"
        / "least_cloudy_times.json"
    )
    assert least_cloudy_times_path.exists(), f"{least_cloudy_times_path} does not exist"
