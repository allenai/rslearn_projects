"""Training transform for the 20260610 temporal LCC model.

This is the single-pass variant of ``FrequentOptionSampler``. Instead of
building a 20-image stack (16 quarterly + 4 frequent) split across two encoder
passes, it builds a single 12-image stack:

1. Randomly picks one of the available frequent image options (4 recent images).
2. Selects 8 semi-annual quarterly images preceding that option's block by
   taking every other quarter (skip every other 90-day mosaic).
3. Concatenates 8 semi-annual + 4 recent = 12 images into ``sentinel2_l2a``.
4. Computes per-pixel start/end timestamp index targets: the index (within the
   12 chronological timesteps) of the timestep at the change start and end.

The model consumes the 12-image stack in a single forward pass (no split).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import torch
from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform

from .transforms import (
    ANNOTATION_KEY,
    FREQUENT_KEY_PREFIX,
    OUTPUT_KEY,
    QUARTERLY_KEY,
)

NUM_SEMIANNUAL = 8
NUM_RECENT = 4
NUM_TIMESTEPS = NUM_SEMIANNUAL + NUM_RECENT

# Spacing used for the fake timestamps assigned to padding (duplicated) images.
SEMIANNUAL_PERIOD = timedelta(days=180)


def _select_semiannual_indices(valid_indices: list[int]) -> list[int]:
    """Select up to NUM_SEMIANNUAL quarterly indices by skipping every other quarter.

    Takes the most recent quarterly image and steps back two quarters at a time
    (semi-annual cadence). Returns up to NUM_SEMIANNUAL indices in chronological
    (ascending) order. Does not pad; the caller pads with fake-older duplicates.
    """
    if not valid_indices:
        return []
    # Newest first, every other quarter, take NUM_SEMIANNUAL, back to chronological.
    return valid_indices[::-1][::2][:NUM_SEMIANNUAL][::-1]


def _build_semiannual_stack(
    quarterly: RasterImage, valid_indices: list[int]
) -> tuple[torch.Tensor, list[tuple[datetime, datetime]]]:
    """Build exactly NUM_SEMIANNUAL quarterly images and their timestamps.

    Selects every other quarter from the most recent valid candidates. When
    fewer than NUM_SEMIANNUAL are available, prepends copies of the earliest
    selected image with progressively older fake timestamps so the encoder sees
    distinct timesteps (it rejects duplicate timestamps within a modality).
    """
    assert quarterly.timestamps is not None
    semi_indices = _select_semiannual_indices(valid_indices)
    q_ts = [quarterly.timestamps[i] for i in semi_indices]

    img_parts: list[torch.Tensor] = []
    if semi_indices:
        img_parts.append(quarterly.image[:, semi_indices, :, :])

    if len(semi_indices) < NUM_SEMIANNUAL:
        deficit = NUM_SEMIANNUAL - len(semi_indices)
        base_idx = semi_indices[0] if semi_indices else 0
        base_ts = q_ts[0][0] if q_ts else quarterly.timestamps[0][0]
        pad_img = quarterly.image[:, base_idx : base_idx + 1, :, :].repeat(
            1, deficit, 1, 1
        )
        # Oldest first; each padding step is one SEMIANNUAL_PERIOD older than the
        # next, all strictly older than base_ts.
        pad_ts = [
            (base_ts - SEMIANNUAL_PERIOD * (deficit - k),) * 2 for k in range(deficit)
        ]
        img_parts.insert(0, pad_img)
        q_ts = pad_ts + q_ts

    return torch.cat(img_parts, dim=1), q_ts


def _change_index(centers: list[datetime], target: datetime, is_start: bool) -> int:
    """Index of the timestep closest to a change boundary.

    For the start boundary, returns the latest center that is <= target (the
    last image before/at the change start). For the end boundary, returns the
    earliest center that is >= target. Defaults to the first/last index when no
    center satisfies the condition.
    """
    if is_start:
        candidates = [i for i, c in enumerate(centers) if c <= target]
        return candidates[-1] if candidates else 0
    candidates = [i for i, c in enumerate(centers) if c >= target]
    return candidates[0] if candidates else len(centers) - 1


class SemiAnnualOptionSampler(Transform):
    """Pick a frequent option, subset semi-annual quarterly, build sentinel2_l2a."""

    def __init__(self, deterministic: bool = False) -> None:
        """Initialize the transform.

        Args:
            deterministic: if True, always pick option 2 (for val/test). This
                matches FrequentOptionSampler: option 2 is a mid-range temporal
                context rather than the hardest immediate-detection case.
        """
        super().__init__()
        self.deterministic = deterministic

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sample a frequent option, subset semi-annual, produce sentinel2_l2a."""
        ann = input_dict.pop(ANNOTATION_KEY, None)
        if ann is None:
            raise KeyError(
                f"Expected {ANNOTATION_KEY!r} in input_dict; is LCCMultiTask in use?"
            )

        pre_change: datetime = ann["pre_change"]
        post_change: datetime = ann["post_change"]

        quarterly: RasterImage = input_dict.pop(QUARTERLY_KEY)
        if quarterly.timestamps is None:
            raise ValueError("sentinel2_quarterly must have timestamps")

        # Collect available frequent options.
        frequent_options: list[RasterImage] = []
        for i in range(8):
            key = f"{FREQUENT_KEY_PREFIX}{i}"
            if key not in input_dict:
                break
            freq_img = input_dict.pop(key)
            if freq_img.image.shape[1] == NUM_RECENT:
                frequent_options.append(freq_img)

        if not frequent_options:
            raise ValueError("No valid frequent options available")

        # Pick an option.
        if self.deterministic:
            opt_idx = 2 if len(frequent_options) > 2 else 0
        else:
            opt_idx = random.randrange(len(frequent_options))
        chosen_frequent = frequent_options[opt_idx]

        if not chosen_frequent.timestamps:
            raise ValueError("Frequent option must have timestamps")

        # Quarterly images end where the frequent block begins.
        earliest_freq_ts = min(ts[0] for ts in chosen_frequent.timestamps)

        # Strict inequality so a quarterly scene captured exactly at the frequent
        # block start (the same Sentinel-2 scene) is not pulled in as a baseline
        # image, which would create a duplicate timestamp with the first frequent.
        valid_indices = [
            i for i, ts in enumerate(quarterly.timestamps) if ts[1] < earliest_freq_ts
        ]
        # Take the most recent 16 candidates, then every other quarter -> 8.
        valid_indices = valid_indices[-(NUM_SEMIANNUAL * 2) :]
        q_img, q_ts = _build_semiannual_stack(quarterly, valid_indices)

        combined_img = torch.cat([q_img, chosen_frequent.image], dim=1)
        combined_ts = q_ts + chosen_frequent.timestamps
        input_dict[OUTPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        # Compute start/end timestamp index targets over the 12 chronological steps.
        centers = [ts[0] + (ts[1] - ts[0]) / 2 for ts in combined_ts]
        start_idx = _change_index(centers, pre_change, is_start=True)
        end_idx = _change_index(centers, post_change, is_start=False)

        H, W = quarterly.image.shape[2], quarterly.image.shape[3]
        start_map = torch.zeros(H, W, dtype=torch.long)
        end_map = torch.zeros(H, W, dtype=torch.long)

        if "binary" in target_dict:
            binary_classes = target_dict["binary"]["classes"].get_hw_tensor()
            change_mask = binary_classes == 2
            start_map[change_mask] = start_idx
            end_map[change_mask] = end_idx
            valid_mask = change_mask.float()
        else:
            valid_mask = torch.ones(H, W, dtype=torch.float32)

        target_dict["timestamps"] = {
            "start": RasterImage(image=start_map[None, None, :, :]),
            "end": RasterImage(image=end_map[None, None, :, :]),
            "valid": RasterImage(image=valid_mask[None, None, :, :]),
        }

        # Mask dst loss when the latest frequent image is before post_change,
        # since the model can't predict destination land cover without post-change imagery.
        latest_freq_ts = max(ts[1] for ts in chosen_frequent.timestamps)
        if latest_freq_ts < post_change and "dst" in target_dict:
            dst_valid = target_dict["dst"]["valid"]
            target_dict["dst"]["valid"] = RasterImage(
                image=torch.zeros_like(dst_valid.image)
            )

        return input_dict, target_dict


class PredictSemiAnnualBuilder(Transform):
    """Build sentinel2_l2a for prediction (no annotation, single frequent layer).

    Mirrors PredictPassBuilder but produces the 8 semi-annual + 4 recent = 12
    stack consumed by TemporalChangeModel. Not exercised by training; provided
    for when prediction is wired up.
    """

    def __init__(self) -> None:
        """Initialize the prediction stack builder."""
        super().__init__()

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Concatenate 8 semi-annual quarterly + 4 frequent into sentinel2_l2a."""
        quarterly: RasterImage = input_dict.pop(QUARTERLY_KEY)
        if quarterly.timestamps is None:
            raise ValueError("sentinel2_quarterly must have timestamps")

        frequent: RasterImage = input_dict.pop(f"{FREQUENT_KEY_PREFIX}0")
        if frequent.image.shape[1] != NUM_RECENT:
            raise ValueError(
                f"Expected prediction frequent layer to have {NUM_RECENT} "
                f"timesteps, got {frequent.image.shape[1]}"
            )

        T = quarterly.image.shape[1]
        valid_indices = list(range(T))[-(NUM_SEMIANNUAL * 2) :]
        q_img, q_ts = _build_semiannual_stack(quarterly, valid_indices)

        combined_img = torch.cat([q_img, frequent.image], dim=1)
        combined_ts = q_ts + (frequent.timestamps or [])
        input_dict[OUTPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        return input_dict, target_dict
