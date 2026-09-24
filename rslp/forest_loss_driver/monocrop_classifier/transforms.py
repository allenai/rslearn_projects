"""Temporal transforms for monocrop classification."""

from __future__ import annotations

import random
from typing import Any

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform

INPUT_KEY = "sentinel2_l2a"
NUM_PRE_TIMESTEPS = 11
MIN_SOURCE_TIMESTEPS = NUM_PRE_TIMESTEPS + 1
MAX_SOURCE_TIMESTEPS = NUM_PRE_TIMESTEPS + 12
NUM_MODEL_TIMESTEPS = 12
MAX_POST_MONTHS = 12


class PostLossMonthSampler(Transform):
    """Select 12 timesteps containing 1-12 months after forest loss.

    The source stack contains 11 pre-loss months followed by 1-12 available
    post-loss months.
    For ``num_post_months=m``, the selected 12-frame slice contains ``12-m``
    pre-loss frames and the first ``m`` post-loss frames. Training omits the
    parameter to sample ``m`` uniformly from 1 through the available maximum.
    A fixed request above that maximum is clamped to the maximum for that sample.
    """

    def __init__(
        self,
        num_post_months: int | None = None,
        default_num_post_months: int | None = None,
        input_key: str = INPUT_KEY,
    ) -> None:
        """Initialize the sampler.

        Args:
            num_post_months: fixed elapsed post-loss months, or None for random.
            default_num_post_months: fallback when num_post_months is None. This
                supports an optional environment-substituted test parameter while
                keeping six months as the default.
            input_key: RasterImage key to slice.
        """
        super().__init__()
        effective_num_post_months = (
            num_post_months if num_post_months is not None else default_num_post_months
        )
        if effective_num_post_months is not None and not (
            1 <= effective_num_post_months <= MAX_POST_MONTHS
        ):
            raise ValueError(
                f"num_post_months must be in [1, {MAX_POST_MONTHS}], "
                f"got {effective_num_post_months}"
            )
        self.num_post_months = effective_num_post_months
        self.input_key = input_key

    def forward(
        self,
        input_dict: dict[str, Any],
        target_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sort monthly frames and select one contiguous 12-frame slice."""
        image = input_dict[self.input_key]
        if not isinstance(image, RasterImage):
            raise TypeError(f"{self.input_key} must be a RasterImage")
        num_timesteps = image.image.shape[1]
        if not MIN_SOURCE_TIMESTEPS <= num_timesteps <= MAX_SOURCE_TIMESTEPS:
            raise ValueError(
                f"{self.input_key} must have {MIN_SOURCE_TIMESTEPS}-"
                f"{MAX_SOURCE_TIMESTEPS} timesteps, "
                f"got {num_timesteps}"
            )
        if image.timestamps is None or len(image.timestamps) != num_timesteps:
            raise ValueError(
                f"{self.input_key} must have one timestamp range per timestep"
            )

        centers = [start + (end - start) / 2 for start, end in image.timestamps]
        order = sorted(range(num_timesteps), key=centers.__getitem__)
        sorted_image = image.image[:, order, :, :]
        sorted_timestamps = [image.timestamps[index] for index in order]

        available_post_months = num_timesteps - NUM_PRE_TIMESTEPS
        if self.num_post_months is None:
            num_post_months = random.randint(1, available_post_months)
        else:
            num_post_months = min(self.num_post_months, available_post_months)
        start = num_post_months - 1
        end = start + NUM_MODEL_TIMESTEPS
        input_dict[self.input_key] = RasterImage(
            image=sorted_image[:, start:end, :, :].contiguous(),
            timestamps=sorted_timestamps[start:end],
        )
        return input_dict, target_dict
