"""TemporalChangeModel: single-pass OlmoEarth encoder with temporal-aware decoder.

Architecture (replaces the dual-pass DualPassChangeModel):
- Input: sentinel2_l2a with 12 timesteps (8 semi-annual + 4 recent), built by
  SemiAnnualOptionSampler.
- A single OlmoEarth forward pass with ``token_pooling=False`` and real
  timestamps, so per-token (per-timestep) features are preserved.
- The encoder output (B, C, H, W, T*S) is reshaped to (B, C, H, W, T, S) and
  mean-pooled over the S band sets to give per-timestep features (B, C, H, W, T).
- A small temporal transformer contextualizes the T tokens at each spatial
  location.
- Segmentation heads (binary/src/dst): mean-pool the tokens over time and run
  per-task decoders that upsample to full resolution. The decoder is either a
  conv stack (``decoder_stages``) or, when ``simple_decoder`` is set, a single
  1x1 conv followed by PixelShuffle.
- Timestamp heads (start/end): per-token linear logits over the T timesteps,
  upsampled to full resolution, trained with cross-entropy at change pixels.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, ModelOutput

from .model import StageSpec, _make_decoder


def _make_simple_decoder(
    in_dim: int, num_classes: int, patch_size: int
) -> nn.Sequential:
    """Single 1x1 conv to num_classes*patch_size^2 channels then PixelShuffle.

    Maps (B, in_dim, H, W) -> (B, num_classes, H*patch_size, W*patch_size).
    """
    return nn.Sequential(
        nn.Conv2d(in_dim, num_classes * patch_size * patch_size, kernel_size=1),
        nn.PixelShuffle(patch_size),
    )


class TemporalChangeModel(nn.Module):
    """Single-pass temporal change model with start/end timestamp prediction."""

    def __init__(
        self,
        encoder: OlmoEarth,
        num_classes_binary: int = 3,
        num_classes_src: int = 13,
        num_classes_dst: int = 13,
        num_timesteps: int = 12,
        embedding_dim: int = 768,
        temporal_dim: int | None = None,
        decoder_stages: list[StageSpec] | None = None,
        simple_decoder: bool = False,
        temporal_depth: int = 1,
        temporal_heads: int = 8,
        dim_feedforward: int = 2048,
        binary_loss_weight: float = 2.0,
    ):
        """Initialize the temporal LCC model.

        Args:
            encoder: the OlmoEarth encoder. Must be configured with
                ``token_pooling=False`` so per-token features are returned.
            num_classes_binary: number of classes for the binary change task.
            num_classes_src: number of source land cover classes.
            num_classes_dst: number of destination land cover classes.
            num_timesteps: expected number of input timesteps (12).
            embedding_dim: per-token encoder embedding size (768 for BASE).
            temporal_dim: temporal transformer / decoder embedding size; defaults
                to ``embedding_dim``. When different, a linear projection maps the
                encoder's ``embedding_dim`` features to ``temporal_dim``.
            decoder_stages: per-task conv decoder definition (see _make_decoder).
                The number of 2x upsamples (len - 1) must equal log2(patch_size).
                Required unless ``simple_decoder`` is set.
            simple_decoder: if set, use a single 1x1 conv + PixelShuffle decoder
                instead of the ``decoder_stages`` conv stack. Works better when
                labels are sparse point labels.
            temporal_depth: number of temporal transformer layers.
            temporal_heads: number of attention heads in the temporal transformer.
            dim_feedforward: hidden size of the temporal transformer FFN (PyTorch
                default is 2048, independent of ``temporal_dim``).
            binary_loss_weight: multiplier applied to the binary change loss.
        """
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.temporal_dim = temporal_dim if temporal_dim is not None else embedding_dim
        self.num_timesteps = num_timesteps
        self.binary_loss_weight = binary_loss_weight

        # Optional projection from encoder dim to temporal transformer dim.
        self.input_proj = (
            nn.Linear(embedding_dim, self.temporal_dim)
            if self.temporal_dim != embedding_dim
            else nn.Identity()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.temporal_dim,
            nhead=temporal_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=temporal_depth
        )

        # Segmentation decoders consume the mean-over-time feature.
        if simple_decoder:
            patch_size = encoder.patch_size
            self.decoder_binary = _make_simple_decoder(
                self.temporal_dim, num_classes_binary, patch_size
            )
            self.decoder_src = _make_simple_decoder(
                self.temporal_dim, num_classes_src, patch_size
            )
            self.decoder_dst = _make_simple_decoder(
                self.temporal_dim, num_classes_dst, patch_size
            )
        else:
            if decoder_stages is None:
                raise ValueError(
                    "decoder_stages must be specified when simple_decoder is False"
                )
            self.decoder_binary = _make_decoder(
                self.temporal_dim, decoder_stages, num_classes_binary
            )
            self.decoder_src = _make_decoder(
                self.temporal_dim, decoder_stages, num_classes_src
            )
            self.decoder_dst = _make_decoder(
                self.temporal_dim, decoder_stages, num_classes_dst
            )

        # Per-token timestamp heads producing one logit per timestep.
        self.start_head = nn.Linear(self.temporal_dim, 1)
        self.end_head = nn.Linear(self.temporal_dim, 1)

        self.num_classes_binary = num_classes_binary
        self.num_classes_src = num_classes_src
        self.num_classes_dst = num_classes_dst

    def _per_timestep_features(self, context: ModelContext) -> torch.Tensor:
        """Run the encoder and return per-timestep features (B, C, H, W, T).

        OlmoEarth v1.1 emits a single band-set token per modality per timestep,
        so the unpooled token dim is exactly the number of timesteps T.
        """
        token_feature_maps = self.encoder(context)
        feature = token_feature_maps.feature_maps[0]  # (B, C, H, W, T)
        return feature

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Single forward pass with temporal-aware decoders.

        Args:
            context: ModelContext with ``sentinel2_l2a`` RasterImage (12 timesteps).
            targets: optional target dicts with "binary", "src", "dst", "timestamps" keys.

        Returns:
            ModelOutput with per-task outputs and losses.
        """
        feature = self._per_timestep_features(context)  # (B, C, H, W, T)
        b, c, h, w, t = feature.shape

        # Temporal transformer over the T tokens at each spatial location.
        x = feature.permute(0, 2, 3, 4, 1).reshape(b * h * w, t, c)
        x = self.input_proj(x)  # (b*h*w, T, temporal_dim)
        x = self.temporal_encoder(x)
        x = x.reshape(b, h, w, t, self.temporal_dim).permute(0, 4, 1, 2, 3)

        # Segmentation aggregation: mean over time.
        seg_feat = x.mean(dim=-1)  # (B, C, H, W)

        logits_binary = self.decoder_binary(seg_feat)
        logits_src = self.decoder_src(seg_feat)
        logits_dst = self.decoder_dst(seg_feat)

        # Per-token timestamp logits over T, upsampled to full resolution.
        xt = x.permute(0, 2, 3, 4, 1)  # (B, H, W, T, C)
        start_logits = self.start_head(xt).squeeze(-1).permute(0, 3, 1, 2)  # (B,T,H,W)
        end_logits = self.end_head(xt).squeeze(-1).permute(0, 3, 1, 2)
        scale = self.encoder.patch_size
        start_logits = F.interpolate(
            start_logits, scale_factor=scale, mode="bilinear", align_corners=False
        )
        end_logits = F.interpolate(
            end_logits, scale_factor=scale, mode="bilinear", align_corners=False
        )

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

    def _seg_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
        task_name: str,
    ) -> torch.Tensor:
        """Compute masked cross-entropy loss for a segmentation task."""
        labels = torch.stack(
            [t[task_name]["classes"].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t[task_name]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, labels, reduction="none")
        return (loss * valid).sum() / valid.sum()

    def _balanced_binary_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
    ) -> torch.Tensor:
        """Balanced binary loss with per-sample balancing.

        For each sample, the loss is the mean over its change points plus the
        mean over its no-change points (each group divided by its own point
        count). Every sample with any valid points contributes equally.
        """
        labels = torch.stack(
            [t["binary"]["classes"].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t["binary"]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, labels, reduction="none")  # (B, H, W)

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

    def _timestamp_ce(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
        key: str,
    ) -> torch.Tensor:
        """Masked cross-entropy over the T timesteps for the start/end boundary.

        ``logits`` is (B, T, H, W); the target is the per-pixel timestep index
        (B, H, W). Loss is averaged over valid (change) pixels only.
        """
        idx = torch.stack(
            [t["timestamps"][key].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t["timestamps"]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()

        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss = F.cross_entropy(logits, idx, reduction="none")  # (B, H, W)
        return (loss * valid).sum() / valid.sum()
