"""FocalBinaryChangeModel: focal-modulated balanced binary change loss.

Variant of ``DualPassChangeModel`` that keeps the per-sample balancing of
``_balanced_binary_loss`` but replaces the plain cross-entropy term with a focal
term ``(1 - p_t)^gamma * CE`` (``p_t`` = softmax prob of the true class), focusing
training on hard change/no-change pixels. No feature or decoder changes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .model import DualPassChangeModel


class FocalBinaryChangeModel(DualPassChangeModel):
    """DualPass change model with a focal-modulated balanced binary loss."""

    def __init__(self, *args: Any, focal_gamma: float = 2.0, **kwargs: Any):
        """Initialize.

        Args:
            focal_gamma: focusing parameter for the focal modulation. 0 reduces to
                the base balanced cross-entropy loss.
            args/kwargs: forwarded to ``DualPassChangeModel.__init__``.
        """
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma

    def _balanced_binary_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Focal-modulated version of the per-sample balanced binary loss."""
        labels = torch.stack(
            [t["binary"]["classes"].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t["binary"]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        ce = F.cross_entropy(logits, labels, reduction="none")  # (B, H, W)
        # p_t = exp(-CE); focal modulation focuses on low-confidence (hard) pixels.
        p_t = torch.exp(-ce)
        loss = (1.0 - p_t).pow(self.focal_gamma) * ce

        change_mask = (valid & (labels == 2)).flatten(1).float()  # (B, H*W)
        nochange_mask = (valid & (labels == 1)).flatten(1).float()
        loss_flat = loss.flatten(1)  # (B, H*W)

        pos_count = change_mask.sum(dim=1)  # (B,)
        neg_count = nochange_mask.sum(dim=1)
        has_pos = pos_count > 0
        has_neg = neg_count > 0

        pos_mean = (loss_flat * change_mask).sum(dim=1) / pos_count.clamp(min=1)
        neg_mean = (loss_flat * nochange_mask).sum(dim=1) / neg_count.clamp(min=1)

        sample_loss = pos_mean * has_pos + neg_mean * has_neg  # (B,)
        has_any = has_pos | has_neg
        return sample_loss[has_any].mean()
