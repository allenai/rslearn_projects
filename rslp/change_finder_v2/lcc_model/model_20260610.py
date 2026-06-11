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
- Segmentation heads (binary/src/dst): aggregate the tokens into
  [mean, baseline_mean, recent_mean, recent-baseline] difference features and
  run per-task conv decoders that upsample to full resolution.
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


class TemporalChangeModel(nn.Module):
    """Single-pass temporal change model with start/end timestamp prediction."""

    def __init__(
        self,
        encoder: OlmoEarth,
        num_classes_binary: int = 3,
        num_classes_src: int = 13,
        num_classes_dst: int = 13,
        num_timesteps: int = 12,
        num_recent: int = 4,
        embedding_dim: int = 768,
        decoder_stages: list[StageSpec] | None = None,
        temporal_depth: int = 1,
        temporal_heads: int = 8,
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
            num_recent: number of trailing "recent" timesteps used to form the
                recent-vs-baseline difference feature.
            embedding_dim: per-token encoder embedding size (768 for BASE).
            decoder_stages: per-task conv decoder definition (see _make_decoder).
                The number of 2x upsamples (len - 1) must equal log2(patch_size).
                Required.
            temporal_depth: number of temporal transformer layers.
            temporal_heads: number of attention heads in the temporal transformer.
            binary_loss_weight: multiplier applied to the binary change loss.
        """
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.num_recent = num_recent
        self.binary_loss_weight = binary_loss_weight

        if decoder_stages is None:
            raise ValueError("decoder_stages must be specified")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=temporal_heads,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=temporal_depth
        )

        # Segmentation decoders consume [mean, baseline, recent, diff] = 4 * dim.
        concat_dim = embedding_dim * 4
        self.decoder_binary = _make_decoder(
            concat_dim, decoder_stages, num_classes_binary
        )
        self.decoder_src = _make_decoder(concat_dim, decoder_stages, num_classes_src)
        self.decoder_dst = _make_decoder(concat_dim, decoder_stages, num_classes_dst)

        # Per-token timestamp heads producing one logit per timestep.
        self.start_head = nn.Linear(embedding_dim, 1)
        self.end_head = nn.Linear(embedding_dim, 1)

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
        x = self.temporal_encoder(x)
        x = x.reshape(b, h, w, t, c).permute(0, 4, 1, 2, 3)  # (B, C, H, W, T)

        # Segmentation aggregation: mean + baseline/recent means + difference.
        split = t - self.num_recent
        mean_t = x.mean(dim=-1)
        baseline = x[..., :split].mean(dim=-1)
        recent = x[..., split:].mean(dim=-1)
        diff = recent - baseline
        seg_feat = torch.cat([mean_t, baseline, recent, diff], dim=1)  # (B, 4C, H, W)

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
