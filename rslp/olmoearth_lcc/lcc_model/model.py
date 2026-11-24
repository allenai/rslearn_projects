"""LCC change model: OlmoEarth encoding with per-timestep token heads.

Input is ``sentinel2_l2a`` with ``num_timesteps`` images (16 quarterly + 6 frequent
= 22, built by ``transforms.StackSampler``). The encoder runs with
``token_pooling=False`` over the whole stack in one pass, so every token can attend
to every other and the output has one token per timestep: ``(B, C, H, W, T)``.

Downstream heads all operate on the per-timestep tokens:

- ``binary``: a learned changepoint scan. For every split t the before-mean A_t and
  after-mean B_t are compared via |B_t - A_t| by a shared scorer; evidence is
  max-pooled over splits, so "change" can only be expressed as before-vs-after
  dissimilarity at some breakpoint. The scan also yields split-attention-weighted
  before/after aggregates that the other heads consume.
- ``src`` / ``dst``: land cover decoders on the weighted before / after aggregate,
  so each looks at the correct side of the breakpoint.
- ``pre_change`` / ``post_change``: change-category decoders on the concatenated
  before and after aggregates, so the heads see both sides of the transition (a
  category is a before->after transition, so a single side cannot separate e.g.
  "new_building" from "none" in an already built-up area).
- ``ts_start`` / ``ts_end``: a per-token linear logit over the T timesteps, upsampled
  to full resolution, trained with cross-entropy at change pixels. These are
  segmentation-style outputs/targets (T classes = timestep indices), matching the
  ``SegmentationTask`` entries of the same names in the task config.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, ModelOutput, RasterImage

from .components import balanced_loss
from .output_bands import (
    NUM_LC_CLASSES,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
)
from .timestamp_encoding import timestamps_to_days
from .transforms import INPUT_KEY, TS_END_KEY, TS_START_KEY

# A stage is a list of (out_channels, kernel_size) conv specs.
StageSpec = list[tuple[int, int]]

# Fixed logit for the (never-supervised) nodata channel of the breakpoint binary head.
NODATA_LOGIT = -10.0

# Hidden width of the shared breakpoint split scorer and the conv stages of the
# 1-channel evidence decoder (which must upsample as many times as decoder_stages).
BREAKPOINT_HIDDEN = 256
BREAKPOINT_EVIDENCE_STAGES: list[StageSpec] = [[(256, 3)], [(128, 3)], [(64, 3)]]


def _make_decoder(
    in_dim: int, stages: list[StageSpec], num_classes: int
) -> nn.Sequential:
    """Build a per-task convolutional decoder from explicit stage specs.

    Each stage is a list of (out_channels, kernel_size) convs (each followed by
    ReLU). A 2x bilinear upsample precedes every stage after the first, so the
    output is at 2^(len(stages)-1) times the input resolution. A final 1x1 conv
    produces num_classes channels.

    Takes (B, in_dim, H, W) and produces (B, num_classes, H*2^(n-1), W*2^(n-1))
    where n = len(stages).
    """
    layers: list[nn.Module] = []
    prev = in_dim
    for i, stage in enumerate(stages):
        if i > 0:
            layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
        for out_ch, k in stage:
            layers.append(nn.Conv2d(prev, out_ch, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU(inplace=True))
            prev = out_ch
    layers.append(nn.Conv2d(prev, num_classes, kernel_size=1))
    return nn.Sequential(*layers)


class ChangeModel(nn.Module):
    """Single-pass OlmoEarth encoder with breakpoint-scan heads (see module docstring)."""

    def __init__(
        self,
        encoder: OlmoEarth,
        num_classes_src: int = NUM_LC_CLASSES,
        num_classes_dst: int = NUM_LC_CLASSES,
        num_classes_pre_change: int = len(PRE_CHANGE_CATEGORY_NAMES),
        num_classes_post_change: int = len(POST_CHANGE_CATEGORY_NAMES),
        num_timesteps: int = 22,
        embedding_dim: int = 768,
        decoder_stages: list[StageSpec] | None = None,
        binary_loss_weight: float = 2.0,
    ):
        """Initialize the LCC change model.

        Args:
            encoder: the OlmoEarth encoder. Must be configured with
                ``token_pooling=False`` so per-token features are returned.
            num_classes_src: number of source land cover classes.
            num_classes_dst: number of destination land cover classes.
            num_classes_pre_change: number of pre_change_category classes
                (including nodata and "none").
            num_classes_post_change: number of post_change_category classes
                (including nodata and "none").
            num_timesteps: number of input timesteps in ``sentinel2_l2a`` (22).
            embedding_dim: per-token encoder embedding size (768 for BASE).
            decoder_stages: per-task conv decoder definition (see _make_decoder).
                The number of 2x upsamples (len - 1) must equal log2(patch_size)
                so outputs are full resolution, and it must have as many stages as
                BREAKPOINT_EVIDENCE_STAGES. Required.
            binary_loss_weight: multiplier applied to the binary change loss.
        """
        super().__init__()
        if decoder_stages is None:
            raise ValueError("decoder_stages must be specified")
        if len(BREAKPOINT_EVIDENCE_STAGES) != len(decoder_stages):
            raise ValueError(
                f"decoder_stages must have {len(BREAKPOINT_EVIDENCE_STAGES)} stages, "
                f"got {len(decoder_stages)}"
            )

        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.binary_loss_weight = binary_loss_weight

        # Shared scorer applied to |after - before| at every split.
        self.split_proj = nn.Sequential(
            nn.Conv2d(embedding_dim, BREAKPOINT_HIDDEN, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(BREAKPOINT_HIDDEN, BREAKPOINT_HIDDEN, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # Scalar per-split score for the split-attention over breakpoints (used to
        # pick the before/after aggregates for the other heads).
        self.split_score = nn.Conv2d(BREAKPOINT_HIDDEN, 1, kernel_size=1)
        self.evidence_decoder = _make_decoder(
            BREAKPOINT_HIDDEN, BREAKPOINT_EVIDENCE_STAGES, 1
        )

        self.decoder_src = _make_decoder(embedding_dim, decoder_stages, num_classes_src)
        self.decoder_dst = _make_decoder(embedding_dim, decoder_stages, num_classes_dst)
        # The change-category decoders take concat(before, after).
        self.decoder_pre_change = _make_decoder(
            2 * embedding_dim, decoder_stages, num_classes_pre_change
        )
        self.decoder_post_change = _make_decoder(
            2 * embedding_dim, decoder_stages, num_classes_post_change
        )

        # Per-token timestamp heads producing one logit per timestep.
        self.start_head = nn.Linear(embedding_dim, 1)
        self.end_head = nn.Linear(embedding_dim, 1)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _per_timestep_features(self, context: ModelContext) -> torch.Tensor:
        """Encode the stack in one encoder call; return (B, C, H, W, T) tokens."""
        for inp in context.inputs:
            image: RasterImage = inp[INPUT_KEY]
            if image.image.shape[1] != self.num_timesteps:
                raise ValueError(
                    f"expected {self.num_timesteps} timesteps in {INPUT_KEY}, got "
                    f"{image.image.shape[1]}"
                )
        tokens = self.encoder(context).feature_maps[0]  # (B, C, H, W, T)
        return tokens[..., : self.num_timesteps]

    def _breakpoint_features(
        self, feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Changepoint scan over the T timesteps.

        Args:
            feature: (B, C, H, W, T) per-timestep tokens.

        Returns:
            (evidence_feat, src_feat, dst_feat): max-pooled per-split hidden
            features (B, hidden, H, W) and the split-attention-weighted
            before/after aggregates (B, C, H, W).
        """
        B, C, H, W, T = feature.shape
        S = T - 1
        cums = feature.cumsum(dim=-1)  # (B, C, H, W, T)
        total = cums[..., -1:]
        counts = torch.arange(1, T, device=feature.device, dtype=feature.dtype)
        before = cums[..., :-1] / counts  # (B, C, H, W, S) mean of images [0, t]
        after = (total - cums[..., :-1]) / (T - counts)  # mean of images (t, T)
        con = (after - before).abs()

        # Shared scorer over all splits: fold S into the batch dimension.
        con = con.permute(0, 4, 1, 2, 3).reshape(B * S, C, H, W)
        hidden = self.split_proj(con)  # (B*S, hidden, H, W)
        scores = self.split_score(hidden)  # (B*S, 1, H, W)
        hidden = hidden.reshape(B, S, -1, H, W)
        scores = scores.reshape(B, S, H, W)

        evidence_feat = hidden.max(dim=1).values
        w = F.softmax(scores, dim=1)  # (B, S, H, W)
        w = w.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H, W, S)
        src_feat = (before * w).sum(dim=-1)
        dst_feat = (after * w).sum(dim=-1)
        return evidence_feat, src_feat, dst_feat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        context: ModelContext,
        targets: list[dict[str, Any]] | None = None,
    ) -> ModelOutput:
        """Forward pass.

        Args:
            context: ModelContext with ``sentinel2_l2a`` RasterImage (num_timesteps).
            targets: optional target dicts with "binary", "src", "dst",
                "pre_change", "post_change", "ts_start", "ts_end" keys.

        Returns:
            ModelOutput with per-task outputs and losses.
        """
        feature = self._per_timestep_features(context)  # (B, C, H, W, T)

        bp_feat, before_feat, after_feat = self._breakpoint_features(feature)
        evidence = self.evidence_decoder(bp_feat)  # (B, 1, H', W')
        logits_binary = torch.cat(
            [
                torch.full_like(evidence, NODATA_LOGIT),
                torch.zeros_like(evidence),
                evidence,
            ],
            dim=1,
        )

        logits_src = self.decoder_src(before_feat)
        logits_dst = self.decoder_dst(after_feat)
        category_in = torch.cat([before_feat, after_feat], dim=1)  # (B, 2C, H, W)
        change_logits = {
            "pre_change": self.decoder_pre_change(category_in),
            "post_change": self.decoder_post_change(category_in),
        }

        # Per-token timestamp logits over T, upsampled to full resolution.
        xt = feature.permute(0, 2, 3, 4, 1)  # (B, H, W, T, C)
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
            losses["binary_cls"] = self.binary_loss_weight * self._balanced_loss(
                logits_binary, targets, "binary"
            )
            losses["src_cls"] = self._seg_loss(logits_src, targets, "src")
            losses["dst_cls"] = self._seg_loss(logits_dst, targets, "dst")
            for name, logits in change_logits.items():
                if name in targets[0]:
                    losses[f"{name}_cls"] = self._balanced_loss(logits, targets, name)
            # Masked CE over the T timesteps, averaged over valid (change) pixels.
            losses["start_ce"] = self._seg_loss(start_logits, targets, TS_START_KEY)
            losses["end_ce"] = self._seg_loss(end_logits, targets, TS_END_KEY)

        outputs: list[dict[str, Any]] = []
        for i in range(len(context.inputs)):
            timestamps = context.inputs[i][INPUT_KEY].timestamps
            if timestamps is None:
                raise ValueError(f"{INPUT_KEY} must have timestamps")
            timestep_days = timestamps_to_days(timestamps)
            outputs.append(
                {
                    "binary": F.softmax(logits_binary[i], dim=0),
                    "src": F.softmax(logits_src[i], dim=0),
                    "dst": F.softmax(logits_dst[i], dim=0),
                    **{
                        name: F.softmax(change_logits[name][i], dim=0)
                        for name in change_logits
                    },
                    TS_START_KEY: F.softmax(start_logits[i], dim=0),
                    TS_END_KEY: F.softmax(end_logits[i], dim=0),
                    "timestep_days": timestep_days,
                }
            )

        return ModelOutput(outputs=outputs, loss_dict=losses)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    @staticmethod
    def _stack_targets(
        targets: list[dict[str, Any]], task_name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack a task's per-sample labels and valid masks into (B, H, W) tensors."""
        labels = torch.stack(
            [t[task_name]["classes"].get_hw_tensor() for t in targets], dim=0
        ).long()
        valid = torch.stack(
            [t[task_name]["valid"].get_hw_tensor() for t in targets], dim=0
        ).bool()
        return labels, valid

    def _seg_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
        task_name: str,
    ) -> torch.Tensor:
        """Compute masked cross-entropy loss for a segmentation task."""
        labels, valid = self._stack_targets(targets, task_name)

        if not valid.any():
            # Zero loss that still depends on logits: under DDP, a rank whose
            # batch has no valid pixels must still produce (zero) gradients for
            # these parameters, or its gradient allreduces go missing and the
            # ranks hang.
            return logits.sum() * 0.0

        loss = F.cross_entropy(logits, labels, reduction="none")
        return (loss * valid).sum() / valid.sum()

    def _balanced_loss(
        self,
        logits: torch.Tensor,
        targets: list[dict[str, Any]],
        task_name: str,
    ) -> torch.Tensor:
        """Per-sample class-balanced cross-entropy (see balanced_loss)."""
        labels, valid = self._stack_targets(targets, task_name)
        return balanced_loss(logits, labels, valid)
