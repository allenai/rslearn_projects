"""TemporalPosEncAttnPoolChangeModel: temporal positional encoding + attention pooling.

Variant of the dual-pass temporal model (``DualPassTemporalChangeModel``,
model_20260611) that adds two changes to the temporal stack:

1. A learned positional embedding over the T chronological timesteps, added before
   the temporal transformer (the base temporal transformer has no explicit ordering
   signal).
2. A learned attention pooling over time for the segmentation heads, replacing the
   plain mean over time.

Everything else (dual-pass token concatenation, start/end timestamp heads, losses)
is inherited unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .model_20260611 import DualPassTemporalChangeModel


class TemporalPosEncAttnPoolChangeModel(DualPassTemporalChangeModel):
    """Dual-pass temporal model with temporal pos-enc and attention pooling."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize and add the positional embedding and time-pooling head."""
        super().__init__(*args, **kwargs)

        # Learned positional embedding over the T chronological tokens.
        self.temporal_pos = nn.Parameter(
            torch.randn(1, self.num_timesteps, self.temporal_dim) * 0.02
        )
        # Additive attention scorer for pooling over time.
        self.time_pool = nn.Linear(self.temporal_dim, 1)

    def _add_temporal_pos(self, x: torch.Tensor) -> torch.Tensor:
        """Add the learned per-index positional embedding.

        ``x`` is ``(b*h*w, T, temporal_dim)``; ``temporal_pos`` broadcasts over the
        batch dimension.
        """
        return x + self.temporal_pos

    def _pool_time(self, x: torch.Tensor) -> torch.Tensor:
        """Attention-pool the per-timestep features over time.

        ``x`` is ``(B, C, H, W, T)``. A learned linear scorer produces one weight
        per timestep, softmax-normalized over T, and the features are summed with
        those weights to give ``(B, C, H, W)``.
        """
        b, c, h, w, t = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).reshape(b * h * w, t, c)  # (N, T, C)
        weights = torch.softmax(self.time_pool(tokens), dim=1)  # (N, T, 1)
        pooled = (tokens * weights).sum(dim=1)  # (N, C)
        return pooled.reshape(b, h, w, c).permute(0, 3, 1, 2)
