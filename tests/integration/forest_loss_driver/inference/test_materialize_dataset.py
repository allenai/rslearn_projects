"""Integration test for dataset materialization for the forest loss driver inference pipeline."""

import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from upath import UPath

from rslp.forest_loss_driver.inference.materialize_dataset import (
    materialize_forest_loss_driver_dataset,
)
from rslp.log_utils import get_logger

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
) -> None:
    """Test materializing the forest loss driver dataset."""
    # copy the unmaterialized dataset to a temp directory
    with tempfile.TemporaryDirectory(prefix=f"test_{uuid.uuid4()}_") as temp_dir:
        shutil.copytree(test_unmaterialized_dataset_path, temp_dir, dirs_exist_ok=True)
        materialize_forest_loss_driver_dataset(UPath(temp_dir))
        # Output of Prepare Step
        items_json_path = (
            Path(temp_dir)
            / "windows"
            / "default"
            / "feat_x_1281600_2146388_5_2221"
            / "items.json"
        )
        # Output of Ingest Step
        tiles_path = Path(temp_dir) / "tiles"
        # Log all contents of tiles directory
        logger.info("\nTiles directory contents:")
        for path in sorted(tiles_path.rglob("*")):
            logger.info(f"  {path.relative_to(tiles_path)}")
        tiff_files = list(tiles_path.rglob("*.tif"))
        metadata_json_files = list(tiles_path.rglob("metadata.json"))
        expected_num_tif_files = 13
        expected_num_metadata_json_files = 13

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
        assert (
            len(tiff_files) == expected_num_tif_files
        ), f"Expected {expected_num_tif_files} TIFF files in the materialized dataset"
        assert (
            len(metadata_json_files) == expected_num_metadata_json_files
        ), f"Expected {expected_num_metadata_json_files} metadata.json files in the materialized dataset"
        layers_dir = (
            Path(temp_dir)
            / "windows"
            / "default"
            / "feat_x_1281600_2146388_5_2221"
            / "layers"
        )
        for layer in expected_layers:
            layer_path = layers_dir / layer / "R_G_B"
            image_path = layer_path / "image.png"
            metadata_path = layer_path / "metadata.json"

            assert layer_path.exists(), f"{layer_path} does not exist"
            assert image_path.exists(), f"{image_path} does not exist"
            assert metadata_path.exists(), f"{metadata_path} does not exist"
