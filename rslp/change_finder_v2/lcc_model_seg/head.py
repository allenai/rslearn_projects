"""Segmentation head with a per-sample, per-class balanced loss.

Drop-in replacement for ``rslearn.train.tasks.segmentation.SegmentationHead``: the
prediction outputs (per-pixel softmax) are identical, so ``SegmentationTask``
metrics, ``process_output`` and prediction writing are unaffected. Only the loss
aggregation differs.

The balanced loss matches the ``lcc_model`` approach
(``_balanced_transition_loss`` in ``rslp.change_finder_v2.lcc_model.model_20260618``):
within each sample we average the cross-entropy within each present class, then
average across classes, then average across samples in the batch. This keeps a
dominant class (e.g. ``no_change`` with 100 points) from swamping a rare
transition class (e.g. 1 ``deforestation`` point).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from rslearn.models.component import FeatureMaps, Predictor
from rslearn.train.model_context import ModelContext, ModelOutput


class BalancedSegmentationHead(Predictor):
    """Segmentation head with per-sample, per-class balanced cross-entropy loss."""

    def __init__(
        self,
        weights: list[float] | None = None,
        temperature: float = 1.0,
    ):
        """Initialize a new BalancedSegmentationHead.

        Args:
            weights: optional per-class weights for cross entropy (Tensor of size C).
            temperature: temperature scaling for softmax, does not affect the loss,
                only the predictor outputs.
        """
        super().__init__()
        if weights is not None:
            self.register_buffer("weights", torch.Tensor(weights))
        else:
            self.weights = None
        self.temperature = temperature

    def forward(
        self,
        intermediates: Any,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Compute segmentation outputs and the balanced loss.

        Args:
            intermediates: a FeatureMaps with a single feature map containing the
                segmentation logits.
            context: the model context.
            targets: list of target dicts, each with a "classes" key (per-pixel
                class labels) and a "valid" key (mask of valid pixels).

        Returns:
            a ModelOutput with per-pixel softmax outputs and the balanced loss.
        """
        if not isinstance(intermediates, FeatureMaps):
            raise ValueError("input to BalancedSegmentationHead must be a FeatureMaps")
        if len(intermediates.feature_maps) != 1:
            raise ValueError(
                "input to BalancedSegmentationHead must have one feature map, "
                f"but got {len(intermediates.feature_maps)}"
            )

        logits = intermediates.feature_maps[0]
        outputs = torch.nn.functional.softmax(logits / self.temperature, dim=1)

        losses = {}
        if targets:
            labels = torch.stack(
                [target["classes"].get_hw_tensor() for target in targets], dim=0
            ).long()
            mask = torch.stack(
                [target["valid"].get_hw_tensor() > 0 for target in targets], dim=0
            )
            losses["cls"] = self._balanced_loss(logits, labels, mask)

        return ModelOutput(outputs=outputs, loss_dict=losses)

    def _balanced_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-sample, per-class balanced cross-entropy over valid pixels.

        Within each sample, the loss is the mean over the classes present (each
        averaged over its own valid pixels), then averaged across samples. Samples
        contribute equally regardless of how many points they have.

        Args:
            logits: model logits, (B, C, H, W).
            labels: per-pixel class labels, (B, H, W).
            mask: valid-pixel mask, (B, H, W), bool.

        Returns:
            the scalar balanced loss.
        """
        if not mask.any():
            # Keep the loss connected to the graph so DDP is happy.
            return logits.sum() * 0.0

        per_pixel = F.cross_entropy(
            logits, labels, weight=self.weights, reduction="none"
        )  # (B, H, W)
        per_pixel = per_pixel.flatten(1)
        labels_flat = labels.flatten(1)
        mask_flat = mask.flatten(1)

        sample_losses: list[torch.Tensor] = []
        for b in range(per_pixel.shape[0]):
            v = mask_flat[b]
            if not v.any():
                continue
            lb = labels_flat[b][v]
            ls = per_pixel[b][v]
            per_class = [ls[lb == cls].mean() for cls in lb.unique()]
            sample_losses.append(torch.stack(per_class).mean())

        return torch.stack(sample_losses).mean()
