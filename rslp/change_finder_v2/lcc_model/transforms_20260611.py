"""Training transform for the 20260611 dual-pass temporal LCC model.

This is the dual-pass variant of ``SemiAnnualOptionSampler``. It builds the same
20-image stack as the original ``FrequentOptionSampler`` (16 quarterly + 4
frequent), so all 16 quarterly mosaics are used (no semi-annual sub-sampling),
but it emits the per-pixel start/end timestep-index targets consumed by the
temporal start/end heads (like ``SemiAnnualOptionSampler``) instead of the
20-channel per-image membership targets.

``DualPassTemporalChangeModel`` splits the 20-image stack across two encoder
passes (the encoder is limited to 12 timesteps per pass) and concatenates the
per-timestep tokens, so the start/end indices computed here index over the 20
chronological timesteps.
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
    NUM_FREQUENT,
    NUM_QUARTERLY,
    OUTPUT_KEY,
    QUARTERLY_KEY,
)
from .transforms_20260610 import _change_index

# Spacing used for the fake timestamps assigned to padding (duplicated) images.
QUARTERLY_PERIOD = timedelta(days=90)


def _build_quarterly_stack(
    quarterly: RasterImage, valid_indices: list[int]
) -> tuple[torch.Tensor, list[tuple[datetime, datetime]]]:
    """Build exactly NUM_QUARTERLY quarterly images and their timestamps.

    Uses the (already-trimmed) ``valid_indices`` in chronological order. When
    fewer than NUM_QUARTERLY are available, prepends copies of the earliest
    selected image with progressively-older fake timestamps so the encoder sees
    distinct timesteps (it rejects duplicate timestamps within a modality when
    ``token_pooling=False``).
    """
    assert quarterly.timestamps is not None
    q_ts = [quarterly.timestamps[i] for i in valid_indices]

    img_parts: list[torch.Tensor] = []
    if valid_indices:
        img_parts.append(quarterly.image[:, valid_indices, :, :])

    if len(valid_indices) < NUM_QUARTERLY:
        deficit = NUM_QUARTERLY - len(valid_indices)
        base_idx = valid_indices[0] if valid_indices else 0
        base_ts = q_ts[0][0] if q_ts else quarterly.timestamps[0][0]
        pad_img = quarterly.image[:, base_idx : base_idx + 1, :, :].repeat(
            1, deficit, 1, 1
        )
        # Oldest first; each padding step is one QUARTERLY_PERIOD older than the
        # next, all strictly older than base_ts.
        pad_ts = [
            (base_ts - QUARTERLY_PERIOD * (deficit - k),) * 2 for k in range(deficit)
        ]
        img_parts.insert(0, pad_img)
        q_ts = pad_ts + q_ts

    return torch.cat(img_parts, dim=1), q_ts


class FrequentOptionSamplerV2(Transform):
    """Pick a frequent option, take 16 quarterly, build sentinel2_l2a + idx targets."""

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
        """Sample a frequent option, take 16 quarterly, produce sentinel2_l2a."""
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
                continue
            freq_img = input_dict.pop(key)
            if freq_img.image.shape[1] == NUM_FREQUENT:
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
        # Take the most recent NUM_QUARTERLY candidates (consecutive, no skipping).
        valid_indices = valid_indices[-NUM_QUARTERLY:]
        q_img, q_ts = _build_quarterly_stack(quarterly, valid_indices)

        combined_img = torch.cat([q_img, chosen_frequent.image], dim=1)
        combined_ts = q_ts + chosen_frequent.timestamps
        input_dict[OUTPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        # Compute start/end timestamp index targets over the chronological steps.
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
