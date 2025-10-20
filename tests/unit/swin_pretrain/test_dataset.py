"""Unit tests for rslp.swin_pretrain.dataset."""

from typing import Any

import torch

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
    ) -> tuple[dict, dict, dict]:
        """Make an example to pass to CollateFunction."""
        input_dict: dict[str, torch.Tensor] = {}
        for modality, num_bands in input_modalities.items():
            input_dict[modality] = torch.zeros(
                (num_bands, height, width), dtype=torch.float32
            )
        target_dict: dict[str, Any] = {}
        for modality in segment_targets:
            target_dict[modality] = {
                "classes": torch.zeros((height, width), dtype=torch.int32),
                "valid": torch.zeros((height, width), dtype=torch.int32),
            }
        metadata = {
            "window_name": f"window{example_idx}",
        }
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
