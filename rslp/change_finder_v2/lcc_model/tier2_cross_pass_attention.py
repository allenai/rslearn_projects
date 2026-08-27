"""CrossPassAttentionChangeModel: let the two encoder passes attend before decoding.

Variant of ``DualPassChangeModel`` that, at each spatial location, treats the two
pass features as a length-2 token sequence and runs a 1-layer transformer so pass1
and pass2 can exchange information. The two contextualized tokens are then
concatenated (still ``2 * embedding_dim`` channels), so the per-task decoders are
unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .model import DualPassChangeModel


class CrossPassAttentionChangeModel(DualPassChangeModel):
    """DualPass change model with cross-pass attention before the decoders."""

    def __init__(
        self,
        *args: Any,
        embedding_dim: int = 768,
        cross_attn_heads: int = 8,
        cross_attn_dim_feedforward: int = 2048,
        **kwargs: Any,
    ):
        """Initialize and add the 1-layer cross-pass transformer.

        Args mirror ``DualPassChangeModel`` plus the attention hyperparameters.

        Args:
            cross_attn_heads: number of attention heads in the cross-pass layer.
            cross_attn_dim_feedforward: FFN hidden size of the cross-pass layer.
        """
        super().__init__(*args, embedding_dim=embedding_dim, **kwargs)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=cross_attn_heads,
            dim_feedforward=cross_attn_dim_feedforward,
            batch_first=True,
        )
        self.cross_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def _combine_features(
        self, feat1: torch.Tensor, feat2: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attend the two passes per spatial location, then concatenate."""
        b, c, h, w = feat1.shape
        t1 = feat1.permute(0, 2, 3, 1).reshape(b * h * w, c)
        t2 = feat2.permute(0, 2, 3, 1).reshape(b * h * w, c)
        tokens = torch.stack([t1, t2], dim=1)  # (b*h*w, 2, C)
        attended = self.cross_attn(tokens)  # (b*h*w, 2, C)
        combined = attended.reshape(b * h * w, 2 * c)
        return combined.reshape(b, h, w, 2 * c).permute(0, 3, 1, 2)
