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


class TeePipe(torch.nn.Module):
    """TeePipe passes different channels of the input image to different backbones.

    The features from the different backbones are then concatenated and returned.
    """

    def __init__(
        self,
        encoders: list[torch.nn.Module],
        channels: list[list[int]],
    ):
        """Create a new TeePipe.

        Args:
            encoders: the encoders to apply.
            channels: the subset of channels that each encoder should input. For
                example, if the input is ABCDEF and first encoder should see ABC while
                second should see DEF, then the list should be [[0, 1, 2], [3, 4, 5]].
        """
        self.encoders = encoders
        self.channels = channels

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute features.

        Args:
            inputs: input dicts that must include "image" key containing the images to
                process.
        """
        # index in feature map -> encoder index -> feature map
        all_features: list[list[torch.Tensor]] | None = None

        for encoder, cur_channels in zip(self.encoders, self.channels):
            cur_features = encoder(
                [{"image": inp["image"][cur_channels, :, :]} for inp in inputs]
            )
            if all_features is None:
                all_features = [[] for _ in cur_features]
            for idx, feat_map in enumerate(cur_features):
                all_features[idx].append(feat_map)

        # Final feature map should concatenate at each scale.
        assert all_features is not None
        return [torch.cat(feat_map_list, dim=1) for feat_map_list in all_features]
