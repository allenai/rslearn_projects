"""Integration test for dataset materialization for the forest loss driver inference pipeline."""

import shutil
from pathlib import Path

import pytest
from upath import UPath

from rslp.forest_loss_driver.extract_dataset import InferenceLayerMaterializeArgs
from rslp.log_utils import get_logger
from rslp.utils.rslearn import materialize_dataset

logger = get_logger(__name__)


@pytest.fixture
def test_unmaterialized_dataset_path() -> UPath:
    """The path to the test unmaterialized dataset."""
    return UPath(
        Path(__file__).resolve().parents[4]
        / "test_data/forest_loss_driver/test_unmaterialized_dataset/dataset_20241023"
    )


def test_materialize_forest_loss_driver_dataset(
    test_unmaterialized_dataset_path: UPath,
    tmp_path: Path,
) -> None:
    """Test materializing the forest loss driver dataset."""
    # copy the unmaterialized dataset to a temp directory that won't be automatically removed
    logger.info(
        f"Copying unmaterialized dataset from {test_unmaterialized_dataset_path} "
        f"to {tmp_path}"
    )
    if not UPath(test_unmaterialized_dataset_path).exists():
        raise FileNotFoundError(
            f"Unmaterialized dataset not found at {test_unmaterialized_dataset_path}"
        )
    shutil.copytree(test_unmaterialized_dataset_path, tmp_path, dirs_exist_ok=True)

    materialize_dataset(UPath(tmp_path), InferenceLayerMaterializeArgs())
    # Output of Prepare Step
    items_json_path = (
        tmp_path
        / "windows"
        / "default"
        / "feat_x_1281600_2146388_5_2221"
        / "items.json"
    )
    # Output of Ingest Step
    tiles_path = tmp_path / "tiles"
    tiff_files = list(tiles_path.rglob("*.tif"))
    completed_files = list(tiles_path.rglob("completed"))
    expected_num_tif_files = 13
    expected_num_completed_files = 13

    # Output of Materialize Step
    expected_layers = [
        "post",
        "post.1",
        "post.2",
        "post.3",
        "post.4",
        "post.5",
        "pre_0",
        "pre_1",
        "pre_2",
        "pre_3",
        "pre_4",
        "pre_5",
        "pre_6",
    ]

    assert items_json_path.exists(), f"{items_json_path} does not exist"
    assert len(tiff_files) == expected_num_tif_files, (
        f"Expected {expected_num_tif_files} TIFF files in the materialized dataset "
        f"found {len(tiff_files)}"
    )
    assert len(completed_files) == expected_num_completed_files, (
        f"Expected {expected_num_completed_files} completed files in the "
        f"materialized dataset found {len(completed_files)}"
    )
    layers_dir = (
        tmp_path / "windows" / "default" / "feat_x_1281600_2146388_5_2221" / "layers"
    )
    for layer in expected_layers:
        layer_path = layers_dir / layer / "R_G_B"
        image_path = layer_path / "image.png"
        metadata_path = layer_path / "metadata.json"

        assert layer_path.exists(), f"{layer_path} does not exist"
        assert image_path.exists(), f"{image_path} does not exist"
        assert metadata_path.exists(), f"{metadata_path} does not exist"
