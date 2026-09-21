"""Transforms for NISAR vessel detection."""

from typing import Any

import torch
from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform


class FillNaN(Transform):
    """Replace NaN (and infinite) pixel values with a finite fill value.

    NISAR GCOV granules use NaN as nodata (e.g. outside the swath), which would
    otherwise pass through the decibel conversion and normalization unchanged and
    poison training with NaN losses.
    """

    def __init__(
        self,
        selectors: list[str] = ["image"],
        fill_value: float = 0.0,
        skip_missing: bool = False,
    ):
        """Initialize a new FillNaN.

        Args:
            selectors: the input selectors to apply the transform on.
            fill_value: the value to replace NaN with. With the default of 0, the
                epsilon clamp in Sentinel1ToDecibels maps these pixels to -60 dB
                (i.e. no backscatter).
            skip_missing: if True, skip selectors that don't exist in the
                input/target dicts. Useful when working with optional inputs.
        """
        super().__init__(skip_missing=skip_missing)
        self.selectors = selectors
        self.fill_value = fill_value

    def apply_image(self, image: RasterImage) -> RasterImage:
        """Replace NaN values in the specified image.

        Args:
            image: the image to transform.
        """
        image.image = torch.nan_to_num(
            image.image,
            nan=self.fill_value,
            posinf=self.fill_value,
            neginf=self.fill_value,
        )
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply the NaN filling over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            transformed (input_dicts, target_dicts) tuple
        """
        self.apply_fn(self.apply_image, input_dict, target_dict, self.selectors)
        return input_dict, target_dict
