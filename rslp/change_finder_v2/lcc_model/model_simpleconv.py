"""From-scratch simple conv change model for the LCC task.

This module only introduces ``SimpleConvChangeModel``. It reuses everything else
(the ``SinglePassSampler`` transform, ``SinglePassMultiTask``, the metrics, and the
five loss functions) from :mod:`rslp.change_finder_v2.lcc_model.model_singlepass`.

Instead of running the heavy OlmoEarth encoder, this model applies a small 2D or
3D conv backbone directly on the normalized ``sentinel2_l2a`` stack (at full
temporal and spatial resolution), then predicts binary/src/dst with per-task
per-pixel temporal heads (attention pooling or a GRU), and predicts the
start/end change boundaries with the same per-timestep linear heads used by the
single-pass model.

The whole thing is a single parameterized class so a sweep over
``conv_type`` x ``num_conv_layers`` x ``embedding_dim`` x ``head_type`` (plus an
optional temporal self-attention stack) can be driven entirely from config.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.train.model_context import ModelContext, ModelOutput

from .model_singlepass import (  # noqa: F401  (re-exported for config convenience)
    BalancedBinaryMetric,
    SinglePassChangeModel,
    SinglePassMultiTask,
    SinglePassPredictBuilder,
    SinglePassSampler,
    TimestampBoundaryAccuracy,
)

# Number of GroupNorm groups. All swept embedding dims (64/128/256) are divisible.
NUM_GROUPS = 8


class _ConvBackbone(nn.Module):
    """2D or 3D conv backbone that preserves temporal and spatial resolution.

    Takes ``(B, in_channels, T, H, W)`` and returns ``(B, dim, T, H, W)``. A stem
    conv maps to ``dim``; the remaining ``num_conv_layers - 1`` convs are residual
    blocks. The 2D variant folds the T dimension into the batch so its 3x3 convs
    are shared across timesteps (all temporal reasoning then happens in the head);
    the 3D variant uses 3x3x3 convs that also mix over time.
    """

    def __init__(
        self, conv_type: str, in_channels: int, dim: int, num_conv_layers: int
    ) -> None:
        """Initialize the backbone.

        Args:
            conv_type: "2d" or "3d".
            in_channels: number of input bands.
            dim: embedding dimension.
            num_conv_layers: total number of conv layers (>= 1); the first is the
                stem that maps in_channels -> dim.
        """
        super().__init__()
        if conv_type not in ("2d", "3d"):
            raise ValueError(f"conv_type must be '2d' or '3d', got {conv_type!r}")
        if num_conv_layers < 1:
            raise ValueError(f"num_conv_layers must be >= 1, got {num_conv_layers}")
        self.conv_type = conv_type

        conv_cls = nn.Conv3d if conv_type == "3d" else nn.Conv2d
        self.stem = conv_cls(in_channels, dim, kernel_size=3, padding=1)
        self.stem_norm = nn.GroupNorm(NUM_GROUPS, dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_conv_layers - 1):
            self.convs.append(conv_cls(dim, dim, kernel_size=3, padding=1))
            self.norms.append(nn.GroupNorm(NUM_GROUPS, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone.

        Args:
            x: input tensor of shape ``(B, in_channels, T, H, W)``.

        Returns:
            features of shape ``(B, dim, T, H, W)``.
        """
        b, _, t, h, w = x.shape
        if self.conv_type == "2d":
            # Fold time into the batch so convs are shared across timesteps.
            x = x.permute(0, 2, 1, 3, 4).reshape(b * t, x.shape[1], h, w)

        x = F.relu(self.stem_norm(self.stem(x)), inplace=True)
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(x + norm(conv(x)), inplace=True)

        if self.conv_type == "2d":
            dim = x.shape[1]
            x = x.reshape(b, t, dim, h, w).permute(0, 2, 1, 3, 4)
        return x


class _AttnHead(nn.Module):
    """Per-pixel temporal cross-attention (attention pooling) head.

    A single learned query per task attends over the T per-pixel temporal tokens
    and the pooled feature is classified with a 1x1 conv.
    """

    def __init__(self, dim: int, num_heads: int, num_classes: int) -> None:
        """Initialize the attention head."""
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.classifier = nn.Conv2d(dim, num_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        """Pool over time and classify.

        Args:
            tokens: ``(B*H*W, T, dim)`` per-pixel temporal tokens.
            b, h, w: batch/height/width to reshape the pooled feature back to a map.

        Returns:
            logits of shape ``(B, num_classes, H, W)``.
        """
        q = self.query.expand(tokens.shape[0], -1, -1)  # (N, 1, dim)
        pooled, _ = self.attn(q, tokens, tokens)  # (N, 1, dim)
        pooled = pooled.squeeze(1).reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return self.classifier(pooled)


class _RNNHead(nn.Module):
    """Per-pixel temporal GRU head.

    A GRU runs over the T per-pixel temporal tokens; the outputs are mean-pooled
    over time and classified with a 1x1 conv.
    """

    def __init__(self, dim: int, bidirectional: bool, num_classes: int) -> None:
        """Initialize the RNN head."""
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=bidirectional)
        out_dim = dim * 2 if bidirectional else dim
        self.classifier = nn.Conv2d(out_dim, num_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        """Run the GRU over time, pool, and classify.

        Args:
            tokens: ``(B*H*W, T, dim)`` per-pixel temporal tokens.
            b, h, w: batch/height/width to reshape the pooled feature back to a map.

        Returns:
            logits of shape ``(B, num_classes, H, W)``.
        """
        out, _ = self.gru(tokens)  # (N, T, out_dim)
        pooled = out.mean(dim=1)  # (N, out_dim)
        pooled = pooled.reshape(b, h, w, -1).permute(0, 3, 1, 2)
        return self.classifier(pooled)


def _make_head(
    head_type: str, dim: int, num_heads: int, bidirectional: bool, num_classes: int
) -> nn.Module:
    """Build a per-task temporal head of the requested type."""
    if head_type == "attn":
        return _AttnHead(dim, num_heads, num_classes)
    if head_type == "rnn":
        return _RNNHead(dim, bidirectional, num_classes)
    raise ValueError(f"head_type must be 'attn' or 'rnn', got {head_type!r}")


class SimpleConvChangeModel(SinglePassChangeModel):
    """Simple from-scratch conv change model with per-task temporal heads.

    Subclasses :class:`SinglePassChangeModel` purely to reuse its five loss
    functions (``_seg_loss``, ``_balanced_binary_loss``, ``_timestamp_ce``), which
    only depend on ``self.binary_loss_weight`` and torch ops. The parent
    ``__init__`` (which requires an encoder and builds upsampling decoders) is
    intentionally bypassed with a direct ``nn.Module.__init__`` call.
    """

    def __init__(
        self,
        conv_type: str,
        num_conv_layers: int,
        embedding_dim: int,
        head_type: str,
        num_temporal_selfattn_layers: int = 0,
        num_classes_binary: int = 3,
        num_classes_src: int = 13,
        num_classes_dst: int = 13,
        num_timesteps: int = 20,
        in_channels: int = 12,
        attn_num_heads: int = 4,
        rnn_bidirectional: bool = True,
        binary_loss_weight: float = 2.0,
    ) -> None:
        """Initialize the simple conv change model.

        Args:
            conv_type: "2d" or "3d" backbone.
            num_conv_layers: total conv layers in the backbone (4/8/12).
            embedding_dim: backbone/head feature dimension (64/128/256).
            head_type: per-task temporal head, "attn" or "rnn".
            num_temporal_selfattn_layers: number of temporal self-attention layers
                applied over the T tokens (per pixel) before the heads. Default 0.
            num_classes_binary: number of binary change classes.
            num_classes_src: number of source land cover classes.
            num_classes_dst: number of destination land cover classes.
            num_timesteps: expected number of input timesteps (e.g. 20).
            in_channels: number of input bands (12 for sentinel2_l2a here).
            attn_num_heads: number of heads for attention (self-attn and attn head).
            rnn_bidirectional: whether the GRU head is bidirectional.
            binary_loss_weight: multiplier applied to the binary change loss.
        """
        nn.Module.__init__(self)

        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.head_type = head_type
        self.binary_loss_weight = binary_loss_weight
        self.num_classes_binary = num_classes_binary
        self.num_classes_src = num_classes_src
        self.num_classes_dst = num_classes_dst

        self.backbone = _ConvBackbone(
            conv_type, in_channels, embedding_dim, num_conv_layers
        )

        # Learned temporal positional embedding over the T chronological tokens.
        self.temporal_pos = nn.Parameter(
            torch.randn(1, num_timesteps, embedding_dim) * 0.02
        )

        if num_temporal_selfattn_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attn_num_heads,
                dim_feedforward=embedding_dim * 4,
                batch_first=True,
            )
            self.temporal_selfattn: nn.Module | None = nn.TransformerEncoder(
                layer, num_layers=num_temporal_selfattn_layers
            )
        else:
            self.temporal_selfattn = None

        self.head_binary = _make_head(
            head_type,
            embedding_dim,
            attn_num_heads,
            rnn_bidirectional,
            num_classes_binary,
        )
        self.head_src = _make_head(
            head_type,
            embedding_dim,
            attn_num_heads,
            rnn_bidirectional,
            num_classes_src,
        )
        self.head_dst = _make_head(
            head_type,
            embedding_dim,
            attn_num_heads,
            rnn_bidirectional,
            num_classes_dst,
        )

        # Per-token timestamp heads producing one logit per timestep.
        self.start_head = nn.Linear(embedding_dim, 1)
        self.end_head = nn.Linear(embedding_dim, 1)

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Run the conv backbone and per-task temporal heads.

        Args:
            context: ModelContext with a ``sentinel2_l2a`` RasterImage per sample,
                each of shape ``(in_channels, num_timesteps, H, W)``.
            targets: optional target dicts with "binary", "src", "dst",
                "timestamps" keys.

        Returns:
            ModelOutput with per-task outputs and losses (same layout as the
            single-pass model).
        """
        device = self.temporal_pos.device
        images = torch.stack(
            [inp["sentinel2_l2a"].image for inp in context.inputs], dim=0
        ).to(device)  # (B, C, T, H, W)

        b, _, t, h, w = images.shape
        if t != self.num_timesteps:
            raise ValueError(f"Expected {self.num_timesteps} timesteps, got {t}")

        feature = self.backbone(images)  # (B, dim, T, H, W)

        # Per-pixel temporal tokens: (B*H*W, T, dim), with positional embedding.
        tokens = feature.permute(0, 3, 4, 2, 1).reshape(
            b * h * w, t, self.embedding_dim
        )
        tokens = tokens + self.temporal_pos
        if self.temporal_selfattn is not None:
            tokens = self.temporal_selfattn(tokens)

        logits_binary = self.head_binary(tokens, b, h, w)
        logits_src = self.head_src(tokens, b, h, w)
        logits_dst = self.head_dst(tokens, b, h, w)

        # Per-token timestamp logits over T (already at full resolution).
        xt = tokens.reshape(b, h, w, t, self.embedding_dim)
        start_logits = self.start_head(xt).squeeze(-1).permute(0, 3, 1, 2)  # (B,T,H,W)
        end_logits = self.end_head(xt).squeeze(-1).permute(0, 3, 1, 2)

        losses: dict[str, torch.Tensor] = {}
        if targets is not None:
            losses["binary_cls"] = self.binary_loss_weight * self._balanced_binary_loss(
                logits_binary, targets
            )
            losses["src_cls"] = self._seg_loss(logits_src, targets, "src")
            losses["dst_cls"] = self._seg_loss(logits_dst, targets, "dst")
            losses["start_ce"] = self._timestamp_ce(start_logits, targets, "start")
            losses["end_ce"] = self._timestamp_ce(end_logits, targets, "end")

        outputs: list[dict[str, Any]] = []
        for i in range(len(context.inputs)):
            outputs.append(
                {
                    "binary": F.softmax(logits_binary[i], dim=0),
                    "src": F.softmax(logits_src[i], dim=0),
                    "dst": F.softmax(logits_dst[i], dim=0),
                    "timestamps": {
                        "start": F.softmax(start_logits[i], dim=0),
                        "end": F.softmax(end_logits[i], dim=0),
                    },
                }
            )

        return ModelOutput(outputs=outputs, loss_dict=losses)
