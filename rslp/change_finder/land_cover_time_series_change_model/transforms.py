"""Random 12-quarter sub-window sampler for the time-series change detector.

The dataset window holds 20 quarterly Sentinel-2 mosaics (5 years). At training
time we slice down to 12 timesteps (3 years). A valid slice must bracket at
least one side of the annotated change event:

* ``has_pre`` if window starts before ``pre_change`` AND ends after
  ``change_start`` -> the model can see the source land cover transitioning.
* ``has_post`` if window starts before ``change_end`` AND ends after
  ``post_change`` -> the model can see the destination land cover settling in.

If only ``has_pre`` is true we zero out the destination-class target's valid
mask (so the dst head is not penalized); symmetric for ``has_post`` and src.
The binary-change target is always valid because we require at least one of
the two transitions to be bracketed.
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any

import torch
from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform

from .tasks import ANNOTATION_KEY

SENTINEL2_KEY = "sentinel2_l2a"
SRC_TASK_NAME = "src"
DST_TASK_NAME = "dst"


class TimeSeriesChangeSubsample(Transform):
    """Pick a 12-quarter sub-window that brackets the change and mask invalid targets."""

    def __init__(
        self,
        num_keep: int = 12,
        deterministic: bool = False,
    ) -> None:
        """Create a new TimeSeriesChangeSubsample.

        Args:
            num_keep: number of contiguous timesteps to keep.
            deterministic: if True, always pick the earliest valid candidate
                (for reproducible val/test).
        """
        super().__init__()
        self.num_keep = num_keep
        self.deterministic = deterministic

    def _enumerate_candidates(
        self,
        timestamps: list[tuple[datetime, datetime]],
        pre: datetime,
        cs: datetime,
        ce: datetime,
        post: datetime,
    ) -> list[tuple[int, bool, bool]]:
        """Return (start_idx, has_pre, has_post) for every valid 12-step sub-window."""
        candidates: list[tuple[int, bool, bool]] = []
        for start in range(len(timestamps) - self.num_keep + 1):
            w_start = timestamps[start][0]
            w_end = timestamps[start + self.num_keep - 1][1]
            has_pre = (w_start < pre) and (w_end > cs)
            has_post = (w_start < ce) and (w_end > post)
            if not has_pre and not has_post:
                continue
            candidates.append((start, has_pre, has_post))
        return candidates

    @staticmethod
    def _mask_target_invalid(target_subdict: dict[str, Any]) -> None:
        """Zero out the ``valid`` mask for a SegmentationTask target sub-dict."""
        valid = target_subdict.get("valid")
        if isinstance(valid, RasterImage):
            valid.image = torch.zeros_like(valid.image)
        elif isinstance(valid, torch.Tensor):
            target_subdict["valid"] = torch.zeros_like(valid)

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Sub-sample timesteps and mask invalid src/dst targets."""
        ann = input_dict.pop(ANNOTATION_KEY, None)
        if ann is None:
            raise KeyError(
                f"Expected {ANNOTATION_KEY!r} in input_dict; is ChangeMultiTask in use?"
            )
        image: RasterImage = input_dict[SENTINEL2_KEY]
        if image.timestamps is None:
            raise ValueError(
                "sentinel2 RasterImage must carry per-item-group timestamps"
            )
        T = image.image.shape[1]
        if len(image.timestamps) != T:
            raise ValueError(f"timestamps length {len(image.timestamps)} != T={T}")
        if T < self.num_keep:
            raise ValueError(f"got T={T} timesteps, need at least {self.num_keep}")

        pre = ann["pre_change"]
        cs = ann["change_start"]
        ce = ann["change_end"]
        post = ann["post_change"]

        candidates = self._enumerate_candidates(image.timestamps, pre, cs, ce, post)

        if candidates:
            if self.deterministic:
                start, has_pre, has_post = candidates[0]
            else:
                start, has_pre, has_post = random.choice(candidates)
        else:
            # Fallback: pick the 12-quarter window ending just after post_change.
            end_idx = self.num_keep - 1
            for i in range(T):
                if image.timestamps[i][1] > post:
                    end_idx = i
                    break
            start = min(max(end_idx - self.num_keep + 1, 0), T - self.num_keep)
            w_start = image.timestamps[start][0]
            w_end = image.timestamps[start + self.num_keep - 1][1]
            has_pre = (w_start < pre) and (w_end > cs)
            has_post = (w_start < ce) and (w_end > post)

        end = start + self.num_keep
        image.image = image.image[:, start:end, :, :].contiguous()
        image.timestamps = image.timestamps[start:end]

        if not has_pre and SRC_TASK_NAME in target_dict:
            self._mask_target_invalid(target_dict[SRC_TASK_NAME])
        if not has_post and DST_TASK_NAME in target_dict:
            self._mask_target_invalid(target_dict[DST_TASK_NAME])

        return input_dict, target_dict
