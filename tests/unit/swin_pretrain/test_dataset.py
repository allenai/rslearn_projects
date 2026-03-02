"""Unit tests for rslp.swin_pretrain.dataset."""

from datetime import datetime, timezone
from typing import Any

import torch
from rasterio.crs import CRS
from rslearn.train.model_context import SampleMetadata
from rslearn.utils.geometry import Projection

from rslp.swin_pretrain.dataset import TILE_SIZE, CollateFunction


class TestCollateFunction:
    """Tests for CollateFunction."""

    def make_example(
        self,
        height: int,
        width: int,
        input_modalities: dict[str, int],
        segment_targets: list[str],
        example_idx: int,
        timesteps: int = 1,
    ) -> tuple[dict, dict, SampleMetadata]:
        """Make an example to pass to CollateFunction."""
        input_dict: dict[str, torch.Tensor] = {}
        for modality, num_bands in input_modalities.items():
            image = torch.zeros(
                (timesteps * num_bands, height, width), dtype=torch.float32
            )
            # Set each timestep to different value so tests can distinguish them.
            for timestep in range(timesteps):
                image[timestep * num_bands : (timestep + 1) * num_bands] = timestep
            input_dict[modality] = image
        target_dict: dict[str, Any] = {}
        for modality in segment_targets:
            target_dict[modality] = {
                "classes": torch.zeros((height, width), dtype=torch.int32),
                "valid": torch.zeros((height, width), dtype=torch.int32),
            }
        metadata = SampleMetadata(
            window_group="fake",
            window_name=f"window{example_idx}",
            window_bounds=(0, 0, height, width),
            crop_bounds=(0, 0, height, width),
            crop_idx=0,
            num_crops_in_window=1,
            time_range=(
                datetime(2024, 1, 1, tzinfo=timezone.utc),
                datetime(2024, 2, 1, tzinfo=timezone.utc),
            ),
            projection=Projection(CRS.from_epsg(32610), 10, -10),
            dataset_source=None,
        )
        return input_dict, target_dict, metadata

    def test_random_cropping(self) -> None:
        """Test that we get random crops of different sizes."""
        min_size = 16
        max_size = 48
        patch_size = 8
        collate_fn = CollateFunction(
            randomize=True,
            min_size=min_size,
            max_size=max_size,
            patch_size=patch_size,
        )
        widths = set()
        for _ in range(10):
            batch = [
                self.make_example(
                    height=TILE_SIZE,
                    width=TILE_SIZE,
                    input_modalities={"10_sentinel2_l2a_monthly": 12},
                    segment_targets=["10_worldcover"],
                    example_idx=0,
                ),
                self.make_example(
                    height=TILE_SIZE,
                    width=TILE_SIZE,
                    input_modalities={"10_sentinel2_l2a_monthly": 12},
                    segment_targets=["10_worldcover"],
                    example_idx=1,
                ),
            ]
            inputs, targets, _ = collate_fn(batch)
            # Make sure all examples in the batch have the same shape.
            assert (
                inputs[0]["10_sentinel2_l2a_monthly"].shape
                == inputs[1]["10_sentinel2_l2a_monthly"].shape
            )
            assert (
                targets[0]["10_worldcover"]["classes"].shape
                == targets[1]["10_worldcover"]["classes"].shape
            )
            # Make sure within an example it has the same height/width.
            assert (
                inputs[0]["10_sentinel2_l2a_monthly"].shape[1:3]
                == targets[0]["10_worldcover"]["classes"].shape[0:2]
            )
            # Make sure it is square and a multiple of the requested patch size.
            height, width = inputs[0]["10_sentinel2_l2a_monthly"].shape[1:3]
            assert height == width
            assert width % patch_size == 0
            assert width >= min_size and width <= max_size
            widths.add(width)

        # Make sure we got at least two unique widths.
        assert len(widths) >= 2

    def test_non_random(self) -> None:
        """Verify that the same window name always is cropped the same way."""
        min_size = 16
        max_size = 48
        patch_size = 8
        collate_fn = CollateFunction(
            randomize=False,
            min_size=min_size,
            max_size=max_size,
            patch_size=patch_size,
        )
        widths: list[set[int]] = [set() for _ in range(10)]
        for _ in range(3):
            for example_idx in range(10):
                batch = [
                    self.make_example(
                        height=TILE_SIZE,
                        width=TILE_SIZE,
                        input_modalities={"10_sentinel2_l2a_monthly": 12},
                        segment_targets=["10_worldcover"],
                        example_idx=example_idx,
                    )
                ]
                inputs, _, _ = collate_fn(batch)
                width = inputs[0]["10_sentinel2_l2a_monthly"].shape[2]
                assert width % patch_size == 0
                widths[example_idx].add(width)

        # Make sure each example has same width across batches.
        all_widths = set()
        for width_set in widths:
            assert len(width_set) == 1
            all_widths.update(width_set)

        # Make sure we got at least two unique widths.
        assert len(all_widths) >= 2

    def test_temporal_subset(self) -> None:
        """Verify that we get different timesteps but they are always in order."""
        min_size = 8
        max_size = 8
        patch_size = 8
        collate_fn = CollateFunction(
            randomize=True,
            min_size=min_size,
            max_size=max_size,
            patch_size=patch_size,
        )
        first_timesteps_set = set()
        num_timesteps_set = set()
        for _ in range(8):
            batch = [
                self.make_example(
                    height=TILE_SIZE,
                    width=TILE_SIZE,
                    input_modalities={"10_sentinel2_l2a_monthly": 12},
                    segment_targets=[],
                    example_idx=0,
                    timesteps=4,
                )
            ]
            inputs, _, _ = collate_fn(batch)
            image = inputs[0]["10_sentinel2_l2a_monthly"]
            num_timesteps = image.shape[0] // 12

            # Verify order of timesteps.
            selected_timesteps = []
            for timestep in range(num_timesteps):
                selected_timesteps.append(image[timestep * 12, 0, 0])
                if len(selected_timesteps) >= 2:
                    assert selected_timesteps[-1] > selected_timesteps[-2]

            num_timesteps_set.add(num_timesteps)
            first_timesteps_set.add(selected_timesteps[0])

        # Make sure we got at least two different first timesteps.
        assert len(first_timesteps_set) >= 2
