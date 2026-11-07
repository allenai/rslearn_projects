"""Training-only components for the rslearn MultiTaskModel version of the LCC model.

The rslearn-native LCC configs (``config_rslearn_*.yaml``) express the change model
as an ``rslearn.models.multitask.MultiTaskModel`` built from generic rslearn
components (OlmoEarth, TemporalTransformer, BreakpointScan, TokensToChannels, Conv,
Upsample, SegmentationHead). ``BalancedSegmentationHead`` is a drop-in replacement
for ``rslearn.train.tasks.segmentation.SegmentationHead`` that uses the per-sample
class-balanced cross-entropy (``balanced_loss``, which ChangeModel also uses for the
binary and pre/post change-category tasks). It is only needed at training time and
carries no parameters, so the predict config can use the plain rslearn head while
loading the same checkpoint.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from rslearn.models.component import FeatureMaps, Predictor
from rslearn.train.model_context import ModelContext, ModelOutput


def balanced_loss(
    logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Per-sample class-balanced cross-entropy.

    For each sample, the loss is the sum over the classes present among its valid
    points of the mean loss over that class's points (each class divided by its own
    point count). The batch loss is the mean over samples with any valid points, so
    samples with many points (e.g. auto-annotated negatives) do not dominate samples
    with few, and within a sample a frequent class does not dominate a rare one.

    Args:
        logits: (B, C, H, W) class logits.
        labels: (B, H, W) integer labels in [0, C).
        valid: (B, H, W) boolean validity mask.

    Returns:
        scalar loss.
    """
    if not valid.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    batch_size, num_classes = logits.shape[:2]
    loss = F.cross_entropy(logits, labels, reduction="none").flatten(1)  # (B, H*W)
    weight = valid.flatten(1).to(loss.dtype)
    index = labels.flatten(1)

    zeros = loss.new_zeros(batch_size, num_classes)
    class_sum = zeros.scatter_add(1, index, loss * weight)  # (B, C)
    class_count = zeros.scatter_add(1, index, weight)
    class_mean = class_sum / class_count.clamp(min=1)

    sample_loss = class_mean.sum(dim=1)  # (B,)
    has_any = class_count.sum(dim=1) > 0
    return sample_loss[has_any].mean()


class BalancedSegmentationHead(Predictor):
    """SegmentationHead variant with the per-sample class-balanced loss.

    The forward output (softmax probabilities) matches
    ``rslearn.train.tasks.segmentation.SegmentationHead`` so the two are
    interchangeable in the model config; only the loss differs. This module has no
    parameters, so checkpoints trained with it load into a config that uses the
    plain ``SegmentationHead`` for prediction.
    """

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compute softmax outputs and the balanced loss.

        Args:
            intermediates: a FeatureMaps with a single feature map of class logits.
            context: the model context.
            targets: list of target dicts with "classes" and "valid" RasterImages.

        Returns:
            ModelOutput with per-sample softmax probabilities and a "cls" loss.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to BalancedSegmentationHead must be a FeatureMaps")
        if len(intermediates.feature_maps) != 1:
            raise ValueError(
                "input to BalancedSegmentationHead must have one feature map, "
                f"but got {len(intermediates.feature_maps)}"
            )

        logits = intermediates.feature_maps[0]
        outputs = F.softmax(logits, dim=1)

        losses: dict[str, torch.Tensor] = {}
        if targets:
            labels = torch.stack(
                [target["classes"].get_hw_tensor() for target in targets], dim=0
            ).long()
            valid = torch.stack(
                [target["valid"].get_hw_tensor() for target in targets], dim=0
            ).bool()
            losses["cls"] = balanced_loss(logits, labels, valid)

        return ModelOutput(outputs=outputs, loss_dict=losses)
