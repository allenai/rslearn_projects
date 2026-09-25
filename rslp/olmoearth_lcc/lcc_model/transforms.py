"""Transforms that build the 22-image ``sentinel2_l2a`` stack for the LCC model.

The stack is 16 quarterly mosaics followed by 6 frequent images. A frequent layer
holds up to 6 of the most recent scenes in a 90-day block; when it has fewer, the
earliest frequent image is duplicated (with fake timestamps between the last
quarterly and the first real frequent image) to fill the 6 slots.

- ``StackSampler`` (train/val): picks one of the materialized frequent options,
  takes the 16 most recent quarterly mosaics before it, and emits the 22-image
  ``sentinel2_l2a`` stack plus per-pixel start/end timestep-index targets (the
  ``TS_START_KEY`` / ``TS_END_KEY`` segmentation-style tasks). It can optionally
  drop quarterly mosaics at train time as augmentation.
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

# Task names of the per-pixel start/end timestep-index targets. Each is a
# segmentation-style ``{"classes", "valid"}`` target over the input timesteps, valid
# only at change points.
TS_START_KEY = "ts_start"
TS_END_KEY = "ts_end"

NUM_QUARTERLY = 16
NUM_FREQUENT = 6
# Maximum number of frequent option layers materialized by the prepare script.
NUM_FREQUENT_OPTIONS = 8

# Spacing used for the fake timestamps assigned to padding (duplicated) images.
QUARTERLY_PERIOD = timedelta(days=90)

# Frequent images closer than this are the same acquisition (e.g. the same
# datatake from overlapping MGRS tiles).
SAME_ACQUISITION_TOLERANCE = timedelta(hours=1)

# Change-category targets that receive a "none" label at negative points.
CHANGE_CATEGORY_TASKS = ("pre_change", "post_change")


def _change_index(
    centers: list[datetime | None], target: datetime, is_start: bool
) -> int:
    """Index of the timestep closest to a change boundary.

    For the start boundary, returns the latest center that is <= target (the
    last image before/at the change start). For the end boundary, returns the
    earliest center that is >= target. Timesteps with a None center are never
    selected. Defaults to the first/last index when no center satisfies the
    condition.
    """
    if is_start:
        candidates = [i for i, c in enumerate(centers) if c is not None and c <= target]
        return candidates[-1] if candidates else 0
    candidates = [i for i, c in enumerate(centers) if c is not None and c >= target]
    return candidates[0] if candidates else len(centers) - 1


def _center(time_range: tuple[datetime, datetime]) -> datetime:
    return time_range[0] + (time_range[1] - time_range[0]) / 2


def _select_frequent(freq: RasterImage) -> RasterImage:
    """Sort frequent images, drop duplicate acquisitions, keep the most recent.

    At most NUM_FREQUENT images are kept, in chronological order.
    """
    if not freq.timestamps:
        raise ValueError("Frequent images must have timestamps")
    timestamps = freq.timestamps
    order = sorted(range(len(timestamps)), key=lambda i: _center(timestamps[i]))
    keep: list[int] = []
    for i in order:
        if keep and (
            _center(timestamps[i]) - _center(timestamps[keep[-1]])
            < SAME_ACQUISITION_TOLERANCE
        ):
            continue
        keep.append(i)
    keep = keep[-NUM_FREQUENT:]
    return RasterImage(
        image=freq.image[:, keep, :, :], timestamps=[timestamps[i] for i in keep]
    )


def _pad_frequent(freq: RasterImage, lower_bound: datetime) -> tuple[RasterImage, int]:
    """Pad chronologically-sorted frequent images to exactly NUM_FREQUENT.

    Prepends copies of the earliest image with fake timestamps evenly spaced
    strictly between ``lower_bound`` (the last quarterly timestamp) and the
    earliest real frequent timestamp, so the stack stays quarterly-then-frequent
    in time and has no duplicate timestamps (the encoder sorts by timestamp and
    rejects exact duplicates).

    Returns the padded image and the number of padding images prepended.
    """
    assert freq.timestamps
    deficit = NUM_FREQUENT - freq.image.shape[1]
    if deficit <= 0:
        return freq, 0
    upper_bound = _center(freq.timestamps[0])
    if lower_bound >= upper_bound:
        lower_bound = upper_bound - QUARTERLY_PERIOD
    step = (upper_bound - lower_bound) / (deficit + 1)
    pad_ts = [(lower_bound + step * (k + 1),) * 2 for k in range(deficit)]
    pad_img = freq.image[:, 0:1, :, :].repeat(1, deficit, 1, 1)
    return (
        RasterImage(
            image=torch.cat([pad_img, freq.image], dim=1),
            timestamps=pad_ts + list(freq.timestamps),
        ),
        deficit,
    )


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

    Builds the 22-image stack (16 quarterly + 6 frequent) in chronological order
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
            if freq_img.image.shape[1] >= 1:
                frequent_options.append(freq_img)

        if not frequent_options:
            raise ValueError("No valid frequent options available")

        # Pick an option.
        if self.deterministic:
            opt_idx = 2 if len(frequent_options) > 2 else 0
        else:
            opt_idx = random.randrange(len(frequent_options))
        chosen_frequent = _select_frequent(frequent_options[opt_idx])
        assert chosen_frequent.timestamps

        # Quarterly images end where the frequent images begin.
        earliest_freq_ts = chosen_frequent.timestamps[0][0]

        # Strict inequality so a quarterly scene captured exactly at the frequent
        # block start (the same Sentinel-2 scene) is not pulled in as a baseline
        # image, which would create a duplicate timestamp with the first frequent.
        valid_indices = [
            i for i, ts in enumerate(quarterly.timestamps) if ts[1] < earliest_freq_ts
        ]
        # Take the most recent NUM_QUARTERLY candidates (consecutive, no skipping).
        valid_indices = valid_indices[-NUM_QUARTERLY:]
        q_img, q_ts = _build_quarterly_stack(quarterly, valid_indices)
        padded_frequent, num_pad = _pad_frequent(chosen_frequent, _center(q_ts[-1]))
        assert padded_frequent.timestamps

        combined_img = torch.cat([q_img, padded_frequent.image], dim=1)
        combined_ts = q_ts + padded_frequent.timestamps
        input_dict[OUTPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        # Compute start/end timestamp index targets over the chronological steps.
        # Frequent padding slots are excluded: they duplicate the first real
        # frequent image but carry earlier fake timestamps.
        centers: list[datetime | None] = [_center(ts) for ts in combined_ts]
        for i in range(len(q_ts), len(q_ts) + num_pad):
            centers[i] = None
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

        for key, idx_map in ((TS_START_KEY, start_map), (TS_END_KEY, end_map)):
            target_dict[key] = {
                "classes": RasterImage(image=idx_map[None, None, :, :]),
                "valid": RasterImage(image=valid_mask.clone()[None, None, :, :]),
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

    At prediction time there is one ``sentinel2_frequent_0`` layer with up to six
    of the most recent scenes in [T - 90d, T], and no annotation sidecar. The
    quarterly layer already ends at T - 90d, so this takes the last 16 quarterly
    images as-is and concatenates 16 quarterly + 6 frequent (padded) = 22 into
    ``sentinel2_l2a``, matching the 22-timestep stack the model is trained on.
    """

    def __init__(self) -> None:
        """Initialize the prediction stack builder."""
        super().__init__()

    @override
    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Concatenate last 16 quarterly + 6 (padded) frequent into sentinel2_l2a."""
        quarterly: RasterImage = input_dict.pop(QUARTERLY_KEY)
        if quarterly.timestamps is None:
            raise ValueError("sentinel2_quarterly must have timestamps")

        frequent: RasterImage = input_dict.pop(f"{FREQUENT_KEY_PREFIX}0")
        if frequent.image.shape[1] < 1:
            raise ValueError("Expected prediction frequent layer to be non-empty")
        frequent = _select_frequent(frequent)

        T = quarterly.image.shape[1]
        indices = list(range(max(0, T - NUM_QUARTERLY), T))
        q_img, q_ts = _build_quarterly_stack(quarterly, indices)
        padded_frequent, _ = _pad_frequent(frequent, _center(q_ts[-1]))
        assert padded_frequent.timestamps

        combined_img = torch.cat([q_img, padded_frequent.image], dim=1)
        combined_ts = q_ts + padded_frequent.timestamps
        input_dict[OUTPUT_KEY] = RasterImage(image=combined_img, timestamps=combined_ts)

        return input_dict, target_dict
