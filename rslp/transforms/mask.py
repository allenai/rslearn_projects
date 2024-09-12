"""Mask transform."""

import torch
from rslearn.train.transforms.transform import Transform


class Mask(Transform):
    """Apply a mask to one or more images."""

    def __init__(
        self,
        selectors: list[str] = ["image"],
        mask_selector: str = "mask",
    ):
        """Initialize a new Mask.

        Args:
            selectors: images to mask.
            mask_selector: the selector for the mask image to apply.
        """
        super().__init__()
        self.selectors = selectors
        self.mask_selector = mask_selector

    def apply_image(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply the mask on the image.

        Args:
            image: the image
            mask: the mask

        Returns:
            masked image
        """
        return image * (mask > 0).float()

    def forward(self, input_dict, target_dict):
        """Apply mask.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        mask = self.read_selector(input_dict, target_dict, self.mask_selector)
        self.apply_fn(
            self.apply_image, input_dict, target_dict, self.selectors, mask=mask
        )
        return input_dict, target_dict
