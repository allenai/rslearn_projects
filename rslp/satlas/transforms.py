"""Transforms for Satlas models."""

from typing import Any

import torch
from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import (
    Transform,
    read_selector,
    selector_exists,
)


class RandomConsecutiveTimesteps(Transform):
    """Keep a random run of consecutive timesteps from a CTHW image.

    This is a temporal augmentation: e.g. the weekly marine infrastructure dataset has
    up to 12 weekly images per training window, and the model is trained on a random
    run of 6 consecutive ones. Images with at most num_timesteps timesteps are left
    unchanged. The same run is used for every selector so multiple inputs stay aligned
    (they must then have the same number of timesteps).
    """

    def __init__(
        self,
        num_timesteps: int,
        selectors: list[str] = ["image"],
        skip_missing: bool = False,
    ):
        """Initialize a new RandomConsecutiveTimesteps.

        Args:
            num_timesteps: the number of consecutive timesteps to keep.
            selectors: the input selectors to apply the transform on.
            skip_missing: if True, skip selectors that don't exist in the
                input/target dicts. Useful when working with optional inputs.
        """
        super().__init__(skip_missing=skip_missing)
        if num_timesteps < 1:
            raise ValueError(f"num_timesteps must be at least 1, got {num_timesteps}")
        self.num_timesteps = num_timesteps
        self.selectors = selectors

    def apply_image(self, image: RasterImage, start: int) -> RasterImage:
        """Keep timesteps [start, start + num_timesteps) of the image.

        Args:
            image: the image to transform.
            start: the first timestep to keep.
        """
        num_total = image.shape[1]
        if num_total <= self.num_timesteps:
            return image
        end = start + self.num_timesteps
        if end > num_total:
            raise ValueError(
                f"selectors have different numbers of timesteps (got {num_total}, "
                f"need at least {end})"
            )
        image.image = image.image[:, start:end]
        if image.timestamps is not None:
            image.timestamps = image.timestamps[start:end]
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply the temporal crop over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            transformed (input_dicts, target_dicts) tuple
        """
        # Pick the start from the first image that has more timesteps than we keep.
        start = 0
        for selector in self.selectors:
            if self.skip_missing and not selector_exists(
                input_dict, target_dict, selector
            ):
                continue
            num_total = read_selector(input_dict, target_dict, selector).shape[1]
            if num_total > self.num_timesteps:
                start = int(
                    torch.randint(num_total - self.num_timesteps + 1, size=()).item()
                )
                break
        self.apply_fn(
            lambda image: self.apply_image(image, start),
            input_dict,
            target_dict,
            self.selectors,
        )
        return input_dict, target_dict
