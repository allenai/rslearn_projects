"""Multi-task wrapper that injects per-window change annotations into the input dict.

The sub-sampling transform (``TimeSeriesChangeSubsample``) needs to know the
four annotated change-period months for the current window so it can pick a
valid 12-quarter sub-window and mask the src/dst targets accordingly. The
standard rslearn transform signature (``forward(input_dict, target_dict)``)
doesn't include any per-sample metadata, so we attach the annotation here, at
task time, where the sample's ``window_group``/``window_name`` are available.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from rslearn.train.model_context import RasterImage, SampleMetadata
from rslearn.train.tasks.multi_task import MultiTask
from rslearn.train.tasks.task import Task
from rslearn.utils import Feature
from upath import UPath

# Key the transform reads from input_dict to find the annotation for this sample.
ANNOTATION_KEY = "_ts_change_annotation"

# Task ordering in the combined 29-band output GeoTIFF.
# Bands 0..2 = binary (nodata, no_change, change)
# Bands 3..15 = src class probs (13 WorldCover classes)
# Bands 16..28 = dst class probs (13 WorldCover classes)
OUTPUT_TASK_ORDER = ("binary", "src", "dst")


def _parse_month(s: str) -> datetime:
    """Parse 'YYYY-MM' into a UTC datetime at the first of the month."""
    return datetime.strptime(s, "%Y-%m").replace(tzinfo=timezone.utc)


class ChangeMultiTask(MultiTask):
    """A :class:`MultiTask` that injects per-window change annotations into the input dict.

    Annotations are loaded once per process from a sidecar JSON written by the
    ``create_windows`` script. The sidecar is keyed by ``"{group}/{name}"``.
    """

    def __init__(
        self,
        tasks: dict[str, Task],
        input_mapping: dict[str, dict[str, str]],
        annotations_path: str,
    ):
        """Create a new ChangeMultiTask.

        Args:
            tasks: map from task name to the task object (passed to MultiTask).
            input_mapping: per-task raw-input remapping (passed to MultiTask).
            annotations_path: path to the sidecar JSON written by
                ``rslp.change_finder.land_cover_time_series_change_model.create_windows``
                (``ts_change_annotations.json`` at the dataset root).
        """
        super().__init__(tasks=tasks, input_mapping=input_mapping)
        self.annotations_path = annotations_path
        self._annotations: dict[str, dict[str, Any]] | None = None

    def _load_annotations(self) -> dict[str, dict[str, Any]]:
        if self._annotations is None:
            with UPath(self.annotations_path).open() as f:
                self._annotations = json.load(f)
        return self._annotations

    def process_inputs(
        self,
        raw_inputs: dict[str, RasterImage | list[Feature]],
        metadata: SampleMetadata,
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run MultiTask.process_inputs then attach the annotation for this sample.

        When ``load_targets=False`` (predict mode) the annotation sidecar is
        not consulted and the annotation key is not injected, since the
        ``TimeSeriesChangeSubsample`` transform is not used during prediction.
        """
        input_dict, target_dict = super().process_inputs(
            raw_inputs, metadata=metadata, load_targets=load_targets
        )
        if load_targets:
            ann = self._load_annotations().get(
                f"{metadata.window_group}/{metadata.window_name}"
            )
            if ann is None:
                raise KeyError(
                    f"No annotation found for {metadata.window_group}/{metadata.window_name} "
                    f"in {self.annotations_path}"
                )
            input_dict[ANNOTATION_KEY] = {
                "pre_change": _parse_month(ann["pre_change"]),
                "change_start": _parse_month(ann["change_start"]),
                "change_end": _parse_month(ann["change_end"]),
                "post_change": _parse_month(ann["post_change"]),
            }
        return input_dict, target_dict

    def process_output(
        self, raw_output: Any, metadata: SampleMetadata
    ) -> npt.NDArray[np.uint8]:
        """Stack per-task softmax probabilities into a single 29-band uint8 CHW array.

        Band layout follows ``OUTPUT_TASK_ORDER``:
        0..2 = binary, 3..15 = src, 16..28 = dst.
        """
        parts: list[torch.Tensor] = []
        for task_name in OUTPUT_TASK_ORDER:
            logits = raw_output[task_name]
            probs = torch.softmax(logits.float(), dim=0)
            parts.append((probs * 255).clamp(0, 255).to(torch.uint8))
        stacked = torch.cat(parts, dim=0)
        return stacked.cpu().numpy()
