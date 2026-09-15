"""Transforms that build the 20-image ``sentinel2_l2a`` stack for the LCC model.

- ``StackSampler`` (train/val): picks one of the materialized frequent options,
  takes the 16 most recent quarterly mosaics before it, and emits the 20-image
  ``sentinel2_l2a`` stack plus per-pixel start/end timestep-index targets. It can
  optionally drop quarterly mosaics at train time as augmentation.
- ``PredictStackBuilder`` (predict): same stack from the single
  ``sentinel2_frequent_0`` layer, with no annotation or targets.
- ``mark_negative_points_none``: labels negative (no-change) points as "none" for
  the change-category heads.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import torch
from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform
from typing_extensions import override

QUARTERLY_KEY = "sentinel2_quarterly"
FREQUENT_KEY_PREFIX = "sentinel2_frequent_"
ANNOTATION_KEY = "_lcc_annotation"
INPUT_KEY = "sentinel2_l2a"
OUTPUT_KEY = "sentinel2_l2a"

NUM_QUARTERLY = 16
NUM_FREQUENT = 4
# Maximum number of frequent option layers materialized by the prepare script.
NUM_FREQUENT_OPTIONS = 8

# Spacing used for the fake timestamps assigned to padding (duplicated) images.
QUARTERLY_PERIOD = timedelta(days=90)

# Change-category targets that receive a "none" label at negative points.
CHANGE_CATEGORY_TASKS = ("pre_change", "post_change")


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


def mark_negative_points_none(target_dict: dict[str, Any]) -> None:
    """Label negative (no-change) points as "none" for the change-category heads.

    At a negative point there is by definition no change over the annotated time
    range, so the pre/post change-category heads should be trained to predict
    class 1 ("none") there instead of being masked out. The label rasters only
    mark negatives in the binary layer (change-category rasters are nodata
    there), so this derives the negative mask from binary class 1 (no_change) and
    sets classes/valid at those pixels for any change-category target present.
    Positive points and unlabeled pixels are untouched.
    """
    if "binary" not in target_dict:
        return
    binary_classes = target_dict["binary"]["classes"].get_hw_tensor().long()
    binary_valid = target_dict["binary"]["valid"].get_hw_tensor() > 0
    neg_mask = binary_valid & (binary_classes == 1)
    if not neg_mask.any():
        return
    for name in CHANGE_CATEGORY_TASKS:
        if name not in target_dict:
            continue
        classes = target_dict[name]["classes"].get_hw_tensor().clone()
        valid = target_dict[name]["valid"].get_hw_tensor().clone()
        classes[neg_mask] = 1  # class 1 = "none"
        valid[neg_mask] = 1
        target_dict[name]["classes"] = RasterImage(image=classes[None, None, :, :])
        target_dict[name]["valid"] = RasterImage(image=valid[None, None, :, :])


class StackSampler(Transform):
    """Pick a frequent option, take 16 quarterly, build sentinel2_l2a + idx targets.

    Builds the 20-image stack (16 quarterly + 4 frequent) in chronological order
    and emits the per-pixel start/end timestep-index targets consumed by the
    start/end heads.

    With ``quarterly_dropout > 0`` (train only), each available quarterly image is
    independently dropped before sampling (keeping at least ``min_keep``), so
    across epochs the model sees different mosaic compositions of the same
    window and learns invariance to composition.
    """

    def __init__(
        self,
        deterministic: bool = False,
        quarterly_dropout: float = 0.0,
        min_keep: int = 8,
    ) -> None:
        """Initialize the transform.

        Args:
            deterministic: if True, always pick option 2 (for val/test) and
                disable quarterly dropout. Option 2 is a mid-range temporal
                context rather than the hardest immediate-detection case.
            quarterly_dropout: probability of dropping each quarterly image.
            min_keep: minimum number of quarterly images to keep under dropout.
        """
        super().__init__()
        self.deterministic = deterministic
        self.quarterly_dropout = quarterly_dropout
        self.min_keep = min_keep

    def _drop_quarterly(self, quarterly: RasterImage) -> RasterImage:
        """Randomly drop quarterly images (train-time augmentation)."""
        if (
            self.deterministic
            or self.quarterly_dropout <= 0
            or quarterly.timestamps is None
        ):
            return quarterly
        T = quarterly.image.shape[1]
        keep = [i for i in range(T) if random.random() >= self.quarterly_dropout]
        if len(keep) < min(self.min_keep, T):
            keep = sorted(random.sample(range(T), min(self.min_keep, T)))
        if len(keep) == T:
            return quarterly
        return RasterImage(
            image=quarterly.image[:, keep, :, :],
            timestamps=[quarterly.timestamps[i] for i in keep],
        )

    @override
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
        quarterly = self._drop_quarterly(quarterly)

        # Collect available frequent options.
        frequent_options: list[RasterImage] = []
        for i in range(NUM_FREQUENT_OPTIONS):
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

        # Mask dst loss when the latest frequent image is before post_change, since
        # the model can't predict destination land cover without post-change imagery.
        latest_freq_ts = max(ts[1] for ts in chosen_frequent.timestamps)
        if latest_freq_ts < post_change and "dst" in target_dict:
            dst_valid = target_dict["dst"]["valid"]
            target_dict["dst"]["valid"] = RasterImage(
                image=torch.zeros_like(dst_valid.image)
            )

        # Train the change-category heads to predict "none" at negative points.
        mark_negative_points_none(target_dict)

        return input_dict, target_dict


class PredictStackBuilder(Transform):
    """Build sentinel2_l2a for prediction (no annotation, single frequent layer).

    At prediction time there is one ``sentinel2_frequent_0`` layer with four 15-day
    periods and no annotation sidecar. This takes the last 16 quarterly images and
    concatenates 16 quarterly + 4 frequent = 20 into ``sentinel2_l2a``, matching the
    20-timestep stack the model is trained on.
    """

    def __init__(self) -> None:
        """Initialize the prediction stack builder."""
        super().__init__()

    @override
    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Concatenate last 16 quarterly + 4 frequent into sentinel2_l2a."""
        quarterly: RasterImage = input_dict.pop(QUARTERLY_KEY)
        if quarterly.timestamps is None:
            raise ValueError("sentinel2_quarterly must have timestamps")

        frequent: RasterImage = input_dict.pop(f"{FREQUENT_KEY_PREFIX}0")
        if frequent.image.shape[1] != NUM_FREQUENT:
            raise ValueError(
                f"Expected prediction frequent layer to have {NUM_FREQUENT} "
                f"timesteps, got {frequent.image.shape[1]}"
            )

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
