"""Training transform: select frequent option, subset quarterly, build sentinel2_l2a.

This transform runs after the task injects annotation metadata. It:
1. Randomly picks one of the available frequent image options
2. Determines when that option's four 15-day periods begin
3. Selects 16 quarterly images ending before the frequent block
4. Concatenates 16 quarterly + 4 frequent = 20 images into a single
   ``sentinel2_l2a`` RasterImage so OlmoEarthNormalize works normally
5. Computes per-image timestamp binary targets

The model splits the 20-image tensor back into two encoder passes.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import torch
import torch.nn.functional as F
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

    def __init__(
        self,
        deterministic: bool = False,
        num_quarterly: int = NUM_QUARTERLY,
        num_frequent: int = NUM_FREQUENT,
        option_index: int | None = None,
        quarterly_anchor: str = "recent",
    ) -> None:
        """Initialize the transform.

        Args:
            deterministic: if True, always pick option 2 (for val/test). Ignored
                when ``option_index`` is set.
            num_quarterly: number of quarterly (historical) images to use.
            num_frequent: number of frequent (recent) periods to keep, taking the
                most recent ones (e.g. 1 for a single recent mosaic). Must be in
                ``[1, NUM_FREQUENT]``.
            option_index: if set, deterministically select
                ``sentinel2_frequent_{option_index}`` instead of using the
                random / option-2 logic (used by the temporal-input ablations).
            quarterly_anchor: how to choose the quarterly images. ``"recent"`` (the
                default) takes the most recent ``num_quarterly`` ending before the
                frequent block. ``"one_year_before"`` instead picks the
                ``num_quarterly`` images whose centers are nearest one year before
                the frequent block (used for the bitemporal ablation).
        """
        super().__init__()
        if not 1 <= num_frequent <= NUM_FREQUENT:
            raise ValueError(
                f"num_frequent must be in [1, {NUM_FREQUENT}], got {num_frequent}"
            )
        if quarterly_anchor not in ("recent", "one_year_before"):
            raise ValueError(
                "quarterly_anchor must be 'recent' or 'one_year_before', got "
                f"{quarterly_anchor!r}"
            )
        self.deterministic = deterministic
        self.num_quarterly = num_quarterly
        self.num_frequent = num_frequent
        self.option_index = option_index
        self.quarterly_anchor = quarterly_anchor

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

        # Collect available frequent options, keyed by their layer index.
        options_by_idx: dict[int, RasterImage] = {}
        for i in range(8):
            key = f"{FREQUENT_KEY_PREFIX}{i}"
            if key not in input_dict:
                continue
            freq_img = input_dict.pop(key)
            if freq_img.image.shape[1] == NUM_FREQUENT:
                options_by_idx[i] = freq_img

        if not options_by_idx:
            raise ValueError("No valid frequent options available")

        # Pick an option.
        if self.option_index is not None:
            # Select a specific frequent layer (used by the ablations). If the
            # requested option was not materialized for this window (e.g. the change
            # is close to IMAGE_CUTOFF so later blocks don't exist), fall back to the
            # latest available option at or before the requested index.
            if self.option_index in options_by_idx:
                chosen_idx = self.option_index
            else:
                earlier = [i for i in options_by_idx if i <= self.option_index]
                chosen_idx = max(earlier) if earlier else max(options_by_idx)
            chosen_frequent = options_by_idx[chosen_idx]
        else:
            frequent_options = [options_by_idx[i] for i in sorted(options_by_idx)]
            if self.deterministic:
                # Use 2 so that it's a random timestep (if we use 0 then it is the
                # option based on the annotated first-noticeable timestamp, which
                # would only test the hardest cases where we want to detect change
                # immediately after it happens).
                opt_idx = 2 if len(frequent_options) > 2 else 0
            else:
                opt_idx = random.randrange(len(frequent_options))
            chosen_frequent = frequent_options[opt_idx]

        if not chosen_frequent.timestamps:
            raise ValueError("Frequent option must have timestamps")

        # Optionally keep only the most recent ``num_frequent`` periods (e.g. a
        # single recent mosaic for the bitemporal ablation).
        if self.num_frequent < chosen_frequent.image.shape[1]:
            chosen_frequent = RasterImage(
                image=chosen_frequent.image[:, -self.num_frequent :, :, :],
                timestamps=chosen_frequent.timestamps[-self.num_frequent :],
            )

        # Determine which quarterly (historical) images to use.
        earliest_freq_ts = min(ts[0] for ts in chosen_frequent.timestamps)
        if self.quarterly_anchor == "one_year_before":
            # Pick the ``num_quarterly`` quarterly images whose centers are closest
            # to one year before the frequent block (bitemporal historical image).
            target_time = earliest_freq_ts - timedelta(days=365)
            q_centers = [ts[0] + (ts[1] - ts[0]) / 2 for ts in quarterly.timestamps]
            nearest = sorted(
                range(len(q_centers)),
                key=lambda i: abs((q_centers[i] - target_time).total_seconds()),
            )
            valid_indices = sorted(nearest[: self.num_quarterly])
        else:
            # Quarterly images end where frequent images begin (prediction semantics).
            valid_indices = [
                i
                for i, ts in enumerate(quarterly.timestamps)
                if ts[1] <= earliest_freq_ts
            ]
            # Take the most recent ``num_quarterly``; pad with earliest if short.
            if len(valid_indices) > self.num_quarterly:
                valid_indices = valid_indices[-self.num_quarterly :]
            elif len(valid_indices) < self.num_quarterly:
                deficit = self.num_quarterly - len(valid_indices)
                pad_idx = valid_indices[0] if valid_indices else 0
                valid_indices = [pad_idx] * deficit + valid_indices

        q_img = quarterly.image[:, valid_indices, :, :]
        q_ts = [quarterly.timestamps[i] for i in valid_indices]

        combined_img = torch.cat([q_img, chosen_frequent.image], dim=1)
        combined_ts = q_ts + chosen_frequent.timestamps
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

        # Mask dst and post_change loss when the latest frequent image is before
        # post_change, since the model can't predict the destination land cover or
        # the post-change category without post-change imagery.
        latest_freq_ts = max(ts[1] for ts in chosen_frequent.timestamps)
        if latest_freq_ts < post_change:
            for key in ("dst", "post_change"):
                if key in target_dict:
                    valid = target_dict[key]["valid"]
                    target_dict[key]["valid"] = RasterImage(
                        image=torch.zeros_like(valid.image)
                    )

        return input_dict, target_dict


class PredictPassBuilder(Transform):
    """Build sentinel2_l2a for prediction (no annotation, single frequent layer).

    At prediction time there is one sentinel2_frequent layer with four 15-day
    periods and no annotation sidecar. This transform takes the last 16 quarterly
    images, concatenates 16 quarterly + 4 frequent = 20 into ``sentinel2_l2a``.
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


class BufferPointLabels(Transform):
    """Dilate sparse point labels to a square buffer (train-time augmentation).

    The LCC dataset stores single-pixel point labels: each task's ``valid`` mask is
    True only at clicked pixels (``label != nodata=0``), so binary/src/dst supervision
    is extremely sparse. This transform expands every labeled pixel to a
    ``buffer_size`` x ``buffer_size`` neighborhood, under a local-homogeneity
    assumption, to densify training supervision. Apply it to ``train_config`` only so
    val/test keep their original sparse labels.

    Class 0 is "unlabeled/nodata" for every task (binary uses 1=no_change, 2=change;
    src/dst use 1..12), so the dilation encodes labels as their class id and treats 0
    as empty. Overlaps are resolved ring-by-ring with a max over class id, so nearer
    labels win and (for binary) change (2) beats no_change (1).
    """

    def __init__(
        self,
        buffer_size: int = 3,
        task_keys: list[str] = ["binary", "src", "dst"],
    ) -> None:
        """Initialize the transform.

        Args:
            buffer_size: side length of the square buffer; must be a positive odd
                int (3 means a single 3x3 ring around each point).
            task_keys: which target tasks to buffer.
        """
        super().__init__()
        if buffer_size < 1 or buffer_size % 2 == 0:
            raise ValueError(
                f"buffer_size must be a positive odd int, got {buffer_size}"
            )
        self.radius = buffer_size // 2
        self.task_keys = task_keys

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Buffer the point labels of each configured task in target_dict."""
        for key in self.task_keys:
            if key not in target_dict:
                continue
            tgt = target_dict[key]
            cls = tgt["classes"].get_hw_tensor().long()  # (H, W)
            valid = tgt["valid"].get_hw_tensor() > 0  # (H, W)

            # Encode labels as class id at valid pixels, 0 elsewhere (0 = unlabeled).
            enc = torch.where(valid, cls, torch.zeros_like(cls)).float()[None, None]
            for _ in range(self.radius):
                pooled = F.max_pool2d(enc, kernel_size=3, stride=1, padding=1)
                # Only fill currently-unlabeled pixels so nearer labels win, and a
                # higher class id (e.g. change over no_change) wins ties.
                newly = (enc == 0) & (pooled > 0)
                enc = torch.where(newly, pooled, enc)

            out = enc[0, 0].long()
            tgt["classes"] = RasterImage(out[None, None, :, :])
            tgt["valid"] = RasterImage((out > 0).float()[None, None, :, :])
        return input_dict, target_dict
