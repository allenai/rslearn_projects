"""Training-only components for the rslearn MultiTaskModel version of the LCC model.

The rslearn-native LCC configs (``config_rslearn_*.yaml``) express the change model
as an ``rslearn.models.multitask.MultiTaskModel`` built from generic rslearn
components (OlmoEarth, TemporalTransformer, BreakpointScan, TokensToChannels, Conv,
Upsample, SegmentationHead). ``BalancedBinarySegmentationHead`` is a drop-in
replacement for ``rslearn.train.tasks.segmentation.SegmentationHead`` that uses the
per-sample balanced cross-entropy from ``ChangeModel._balanced_binary_loss``. It is
only needed at training time and carries no parameters, so the predict config can
use the plain rslearn head while loading the same checkpoint.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from rslearn.models.component import FeatureMaps, Predictor
from rslearn.train.model_context import ModelContext, ModelOutput

# Binary label convention shared with the label rasters and the ChangeModel.
NO_CHANGE_CLASS = 1
CHANGE_CLASS = 2


def balanced_binary_loss(
    logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Per-sample class-balanced cross-entropy for the binary change task.

    For each sample, the loss is the mean over its change points plus the mean over
    its no-change points (each group divided by its own point count), so every
    sample with any valid points contributes equally and samples with many
    auto-annotated negatives do not dominate.

    Args:
        logits: (B, C, H, W) class logits.
        labels: (B, H, W) integer labels (1 = no_change, 2 = change).
        valid: (B, H, W) boolean validity mask.

    Returns:
        scalar loss.
    """
    if not valid.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    loss = F.cross_entropy(logits, labels, reduction="none")  # (B, H, W)

    change_mask = (valid & (labels == CHANGE_CLASS)).flatten(1).float()  # (B, H*W)
    nochange_mask = (valid & (labels == NO_CHANGE_CLASS)).flatten(1).float()
    loss_flat = loss.flatten(1)

    pos_count = change_mask.sum(dim=1)  # (B,)
    neg_count = nochange_mask.sum(dim=1)
    has_pos = pos_count > 0
    has_neg = neg_count > 0

    pos_mean = (loss_flat * change_mask).sum(dim=1) / pos_count.clamp(min=1)
    neg_mean = (loss_flat * nochange_mask).sum(dim=1) / neg_count.clamp(min=1)

    sample_loss = pos_mean * has_pos + neg_mean * has_neg  # (B,)
    has_any = has_pos | has_neg
    if not has_any.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return sample_loss[has_any].mean()


class BalancedBinarySegmentationHead(Predictor):
    """SegmentationHead variant with the per-sample balanced binary change loss.

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
            raise ValueError(
                "input to BalancedBinarySegmentationHead must be a FeatureMaps"
            )
        if len(intermediates.feature_maps) != 1:
            raise ValueError(
                "input to BalancedBinarySegmentationHead must have one feature map, "
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
            losses["cls"] = balanced_binary_loss(logits, labels, valid)

        return ModelOutput(outputs=outputs, loss_dict=losses)
