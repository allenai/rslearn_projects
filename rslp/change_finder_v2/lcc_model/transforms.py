"""Training transform: select frequent option, subset quarterly, build sentinel2_l2a.

This transform runs after the task injects annotation metadata. It:
1. Randomly picks one of the available frequent image options (0-3)
2. Determines the latest frequent image timestamp
3. Selects 16 quarterly images ending at/before that timestamp
4. Concatenates 16 quarterly + 4 frequent = 20 images into a single
   ``sentinel2_l2a`` RasterImage so OlmoEarthNormalize works normally
5. Computes per-image timestamp binary targets

The model splits the 20-image tensor back into two encoder passes.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

import torch
from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform

QUARTERLY_KEY = "sentinel2_quarterly"
FREQUENT_KEY_PREFIX = "sentinel2_frequent_"
ANNOTATION_KEY = "_lcc_annotation"
OUTPUT_KEY = "sentinel2_l2a"

NUM_QUARTERLY = 16
NUM_FREQUENT = 4


class FrequentOptionSampler(Transform):
    """Pick a frequent option, subset quarterly images, and build sentinel2_l2a."""

    def __init__(self, deterministic: bool = False) -> None:
        """Initialize the transform.

        Args:
            deterministic: if True, always pick option 0 (for val/test).
        """
        super().__init__()
        self.deterministic = deterministic

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sample a frequent option, subset quarterly, and produce sentinel2_l2a."""
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

        # Collect available frequent options
        frequent_options: list[RasterImage] = []
        for i in range(4):
            key = f"{FREQUENT_KEY_PREFIX}{i}"
            if key in input_dict:
                freq_img = input_dict.pop(key)
                if freq_img.image.shape[1] == NUM_FREQUENT:
                    frequent_options.append(freq_img)
            else:
                break

        if not frequent_options:
            raise ValueError("No valid frequent options available")

        # Pick an option
        if self.deterministic:
            opt_idx = 0
        else:
            opt_idx = random.randrange(len(frequent_options))
        chosen_frequent = frequent_options[opt_idx]

        # Get the earliest frequent image timestamp to determine quarterly cutoff.
        # Quarterly images end where frequent images begin (matching prediction semantics).
        if chosen_frequent.timestamps:
            earliest_freq_ts = min(ts[0] for ts in chosen_frequent.timestamps)
        else:
            earliest_freq_ts = post_change

        # Select quarterly images with end timestamp <= earliest frequent timestamp
        valid_indices = []
        for i, ts in enumerate(quarterly.timestamps):
            if ts[1] <= earliest_freq_ts:
                valid_indices.append(i)

        # Take the 16 most recent; pad with earliest if insufficient
        if len(valid_indices) > NUM_QUARTERLY:
            valid_indices = valid_indices[-NUM_QUARTERLY:]
        elif len(valid_indices) < NUM_QUARTERLY:
            deficit = NUM_QUARTERLY - len(valid_indices)
            pad_idx = valid_indices[0] if valid_indices else 0
            valid_indices = [pad_idx] * deficit + valid_indices

        q_img = quarterly.image[:, valid_indices, :, :]
        q_ts = [quarterly.timestamps[i] for i in valid_indices]

        combined_img = torch.cat([q_img, chosen_frequent.image], dim=1)
        combined_ts = q_ts + (chosen_frequent.timestamps or [])
        input_dict[OUTPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        # Compute timestamp targets: for each of 20 images, 1 if in change period.
        # Snap pre/post_change to nearest image centers so that at least the
        # closest image before pre_change and after post_change are included.
        all_timestamps = combined_ts
        centers = [ts[0] + (ts[1] - ts[0]) / 2 for ts in all_timestamps]
        pre_snapped = max((c for c in centers if c <= pre_change), default=min(centers))
        post_snapped = min(
            (c for c in centers if c >= post_change), default=max(centers)
        )
        ts_targets = [1.0 if pre_snapped <= c <= post_snapped else 0.0 for c in centers]

        H, W = quarterly.image.shape[2], quarterly.image.shape[3]
        num_ts = len(ts_targets)
        ts_tensor = torch.tensor(ts_targets, dtype=torch.float32)

        # Rasterize timestamp targets only at positive (change) point pixels.
        ts_classes = torch.zeros(num_ts, H, W, dtype=torch.float32)
        if "binary" in target_dict:
            binary_classes = target_dict["binary"]["classes"].get_hw_tensor()
            change_mask = binary_classes == 2
            ts_classes[:, change_mask] = ts_tensor.unsqueeze(1)
            ts_valid_mask = change_mask.float()
            ts_valid = RasterImage(image=ts_valid_mask[None, None, :, :])
        else:
            ts_valid = RasterImage(image=torch.ones(1, 1, H, W, dtype=torch.float32))

        # Wrap as RasterImage (C=num_ts, T=1) so Flip can handle it.
        ts_classes_img = RasterImage(image=ts_classes[:, None, :, :])

        target_dict["timestamps"] = {
            "classes": ts_classes_img,
            "valid": ts_valid,
        }

        return input_dict, target_dict


class PredictPassBuilder(Transform):
    """Build sentinel2_l2a for prediction (no annotation, single frequent layer).

    At prediction time there is only one sentinel2_frequent layer (not 4 options)
    and no annotation sidecar. This transform takes the last 16 quarterly images,
    concatenates 16 quarterly + 4 frequent = 20 into ``sentinel2_l2a``.
    """

    def __init__(self) -> None:
        """Initialize the prediction pass builder."""
        super().__init__()

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Concatenate last 16 quarterly + 4 frequent into sentinel2_l2a."""
        quarterly: RasterImage = input_dict.pop(QUARTERLY_KEY)
        if quarterly.timestamps is None:
            raise ValueError("sentinel2_quarterly must have timestamps")

        frequent: RasterImage = input_dict.pop(f"{FREQUENT_KEY_PREFIX}0")

        T = quarterly.image.shape[1]
        start = max(0, T - NUM_QUARTERLY)
        indices = list(range(start, T))
        if len(indices) < NUM_QUARTERLY:
            deficit = NUM_QUARTERLY - len(indices)
            pad_idx = indices[0] if indices else 0
            indices = [pad_idx] * deficit + indices

        q_img = quarterly.image[:, indices, :, :]
        q_ts = [quarterly.timestamps[i] for i in indices]

        combined_img = torch.cat([q_img, frequent.image], dim=1)
        combined_ts = q_ts + (frequent.timestamps or [])
        input_dict[OUTPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        return input_dict, target_dict
