"""Transforms for the annotation timestamp helper."""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

import torch
from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform

from .constants import (
    MODEL_INPUT_KEY,
    NUM_CROP_MONTHS,
    SENTINEL2_LAYER,
    TIMESTAMP_HEADS,
)


def _parse_datetime(value: str) -> datetime:
    """Parse an ISO timestamp and return UTC datetime."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class TemporalCropFiveYears(Transform):
    """Crop a prepared 30-day sequence to 60 frames.

    Training uses a random crop start. Validation and prediction use the middle
    60-frame crop. Labels are stored as timestamp strings and mapped to the nearest
    cropped frame center.
    """

    def __init__(
        self,
        deterministic: bool = True,
        input_key: str = SENTINEL2_LAYER,
        output_key: str = MODEL_INPUT_KEY,
        fixed_start: int | None = None,
    ) -> None:
        """Initialize the temporal crop transform."""
        super().__init__()
        self.deterministic = deterministic
        self.input_key = input_key
        self.output_key = output_key
        self.fixed_start = fixed_start

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Crop imagery and make labels crop-relative."""
        image: RasterImage = input_dict.pop(self.input_key)
        num_frames = image.image.shape[1]
        if num_frames < NUM_CROP_MONTHS:
            raise ValueError(
                f"{self.input_key} must have at least {NUM_CROP_MONTHS} timesteps, "
                f"got {num_frames}"
            )
        if image.timestamps is None:
            raise ValueError(f"{self.input_key} must have timestamps")

        centers = [ts[0] + (ts[1] - ts[0]) / 2 for ts in image.timestamps]
        order = sorted(range(num_frames), key=lambda idx: centers[idx])
        sorted_image = image.image[:, order, :, :]
        sorted_timestamps = [image.timestamps[idx] for idx in order]
        sorted_centers = [centers[idx] for idx in order]

        max_start = num_frames - NUM_CROP_MONTHS
        if self.deterministic:
            if self.fixed_start is None:
                crop_start = max_start // 2
            else:
                crop_start = min(max(self.fixed_start, 0), max_start)
        else:
            crop_start = random.randint(0, max_start)

        crop_end = crop_start + NUM_CROP_MONTHS
        cropped_timestamps = sorted_timestamps[crop_start:crop_end]
        cropped_centers = sorted_centers[crop_start:crop_end]
        input_dict[self.output_key] = RasterImage(
            image=sorted_image[:, crop_start:crop_end, :, :],
            timestamps=cropped_timestamps,
        )
        input_dict["_timestamp_crop_start"] = crop_start

        for head in TIMESTAMP_HEADS:
            key = f"{head}_date"
            if key not in target_dict:
                continue
            target_date = _parse_datetime(target_dict[key]["date"])
            rel_idx = min(
                range(NUM_CROP_MONTHS),
                key=lambda idx: abs(cropped_centers[idx] - target_date),
            )
            in_crop = (
                cropped_timestamps[0][0] <= target_date <= cropped_timestamps[-1][1]
            )
            valid = target_dict[key]["valid"] * float(in_crop)
            target_dict[head] = {
                "class": torch.tensor(rel_idx, dtype=torch.int64),
                "valid": valid,
            }
            del target_dict[key]

        return input_dict, target_dict
