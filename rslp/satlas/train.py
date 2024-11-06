"""Satlas custom training code."""

from typing import Any

import torch
from rslearn.train.tasks.detection import DetectionTask
from rslearn.utils import Feature

CATEGORY_MAPPING = {
    "power": "platform",
}


class MarineInfraTask(DetectionTask):
    """Marine infrastructure detection task.

    We just add a category remapping pre-processing.
    """

    def process_inputs(
        self,
        raw_inputs: dict[str, torch.Tensor | list[Feature]],
        metadata: dict[str, Any],
        load_targets: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Processes the data into targets.

        Args:
            raw_inputs: raster or vector data to process
            metadata: metadata about the patch being read
            load_targets: whether to load the targets or only inputs

        Returns:
            tuple (input_dict, target_dict) containing the processed inputs and targets
                that are compatible with both metrics and loss functions
        """
        if not load_targets:
            return {}, {}

        for feat in raw_inputs["targets"]:
            if self.property_name not in feat.properties:
                continue
            category = feat.properties[self.property_name]
            if category not in CATEGORY_MAPPING:
                continue
            feat.properties[self.property_name] = CATEGORY_MAPPING[category]

        return super().process_inputs(raw_inputs, metadata, load_targets)
