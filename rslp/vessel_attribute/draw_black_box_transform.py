"""Transform to draw a black box over the center of an image.

Used for vessel attribute experiments to measure how well the model can recover
attributes when the center of the vessel crop is masked out.
"""

from typing import Any

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform


class DrawBlackBox(Transform):
    """Set a centered square region of the image to a constant fill value."""

    def __init__(
        self,
        box_size: int,
        image_selectors: list[str] = ["image"],
        fill_value: float = 0.0,
        skip_missing: bool = False,
    ):
        """Initialize a new DrawBlackBox.

        Args:
            box_size: side length (in pixels) of the square box to draw at the center.
            image_selectors: image items to transform.
            fill_value: value to fill the box with (post-normalization, default 0).
            skip_missing: if True, skip selectors that don't exist in the input/target
                dicts. Useful when working with optional inputs.
        """
        super().__init__(skip_missing=skip_missing)
        self.box_size = box_size
        self.image_selectors = image_selectors
        self.fill_value = fill_value

    def sample_state(self) -> dict[str, Any]:
        """Randomly decide how to transform the input.

        Returns:
            dict of sampled choices (none needed; the box is deterministic).
        """
        return {}

    def apply_image(self, image: RasterImage, state: dict[str, Any]) -> RasterImage:
        """Apply the centered black box on the specified image.

        Args:
            image: the image to transform (4D CTHW tensor).
            state: the sampled state.

        Returns:
            the transformed image.
        """
        height = image.image.shape[-2]
        width = image.image.shape[-1]
        if self.box_size > height or self.box_size > width:
            raise ValueError(
                f"box_size {self.box_size} exceeds image dimensions "
                f"(height={height}, width={width})"
            )
        row_start = height // 2 - self.box_size // 2
        col_start = width // 2 - self.box_size // 2
        image.image[
            ...,
            row_start : row_start + self.box_size,
            col_start : col_start + self.box_size,
        ] = self.fill_value
        return image

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply transform over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            transformed (input_dict, target_dict) tuple
        """
        state = self.sample_state()
        self.apply_fn(
            self.apply_image, input_dict, target_dict, self.image_selectors, state=state
        )
        return input_dict, target_dict
