"""Integration tests for the best image selector."""

import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from upath import UPath

from rslp.forest_loss_driver.inference.best_image_selector import (
    select_best_images_pipeline,
)


@pytest.fixture
def test_materialized_dataset_path() -> UPath:
    """The path to the test materialized dataset."""
    return UPath(
        Path(__file__).resolve().parents[4]
        / "test_data/forest_loss_driver/test_materialized_dataset/dataset_20241023"
    )


def test_select_best_images_pipeline(
    test_materialized_dataset_path: UPath,
) -> None:
    # Want to make sure we have the best times for each layer?
    # Will all layers in config be present? Is there the case of no best images? What is expected behavior?
    with tempfile.TemporaryDirectory(prefix=f"test_{uuid.uuid4()}_") as temp_dir:
        shutil.copytree(test_materialized_dataset_path, temp_dir, dirs_exist_ok=True)
        select_best_images_pipeline(UPath(temp_dir))
        best_times_path = (
            Path(temp_dir)
            / "windows"
            / "default"
            / "feat_x_1281600_2146388_5_2221"
            / "best_times.json"
        )
        assert best_times_path.exists(), f"{best_times_path} does not exist"
