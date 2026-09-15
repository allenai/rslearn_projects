"""LCC change model: two-pass OlmoEarth encoding with per-timestep token heads.

Input is ``sentinel2_l2a`` with ``num_timesteps`` images (16 quarterly + 4 frequent
= 20, built by ``transforms.StackSampler``). The encoder runs with
``token_pooling=False`` so per-timestep tokens are preserved.

Encoding is always done in two passes: the stack is split at ``num_pass1`` into a
historical half and a recent half, both halves are encoded in a single batched
encoder call (``2B`` samples of ``num_pass1`` / ``num_timesteps - num_pass1``
images), and the resulting tokens are concatenated back along time to give
``(B, C, H, W, num_timesteps)`` features. Encoder attention is quadratic in the
token count, so two 10-image passes cost about half the attention FLOPs of one
20-image pass; the price is that tokens from the two halves do not attend to each
other inside the encoder.

Downstream heads all operate on the concatenated per-timestep tokens:

- ``season_embed``: optionally add a month-of-year (sin/cos) embedding to each
  timestep token so seasonal differences between mosaics are explainable by
  covariates instead of being read as change.
- ``temporal_depth > 0``: optionally contextualize the T tokens at each spatial
  location with a small temporal transformer (optionally with a learned
  positional embedding over the T chronological slots, ``temporal_pos_enc``).
- ``temporal_aggregation``: how tokens are pooled over time into the feature
  consumed by the pre/post change-category heads (and, in ``binary_mode="mean"``,
  the binary/src/dst heads): ``"mean"``, ``"diff"`` (last minus first timestep),
  or ``"attn"`` (learned attention pooling).
- ``binary_mode``:
    * ``"mean"``: the binary change head is a conv decoder on the pooled feature.
    * ``"breakpoint"``: a learned changepoint scan. For every split t the
      before-mean A_t and after-mean B_t are compared via |B_t - A_t| by a shared
      scorer; evidence is max-pooled over splits. src is decoded from the
      split-attention-weighted before feature and dst from the weighted after
      feature, so "change" can only be expressed as before-vs-after dissimilarity
      at some breakpoint and src/dst look at the correct sides of it.
- start/end timestamps: a per-token linear logit over the T timesteps, upsampled
  to full resolution, trained with cross-entropy at change pixels.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, ModelOutput, RasterImage

from .timestamp_encoding import timestamps_to_days
from .transforms import INPUT_KEY

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
    """Two-pass OlmoEarth encoder with per-task decoders over per-timestep tokens."""

    def __init__(
        self,
        encoder: OlmoEarth,
        num_classes_binary: int = 3,
        num_classes_src: int = 13,
        num_classes_dst: int = 13,
        num_classes_pre_change: int = 11,
        num_classes_post_change: int = 15,
        num_timesteps: int = 20,
        num_pass1: int = 10,
        embedding_dim: int = 768,
        decoder_stages: list[StageSpec] | None = None,
        binary_loss_weight: float = 2.0,
        binary_mode: str = "mean",
        temporal_aggregation: str = "mean",
        season_embed: bool = False,
        temporal_depth: int = 0,
        temporal_heads: int = 8,
        dim_feedforward: int = 2048,
        temporal_pos_enc: bool = False,
    ):
        """Initialize the LCC change model.

        Args:
            encoder: the OlmoEarth encoder. Must be configured with
                ``token_pooling=False`` so per-token features are returned.
            num_classes_binary: number of classes for the binary change task
                (3: nodata/no_change/change).
            num_classes_src: number of source land cover classes.
            num_classes_dst: number of destination land cover classes.
            num_classes_pre_change: number of pre_change_category classes
                (including nodata and "none", and the merged same_change
                categories; see tasks.MERGED_PRE_SAME_CATEGORY_NAMES).
            num_classes_post_change: number of post_change_category classes
                (including nodata and "none").
            num_timesteps: number of input timesteps in ``sentinel2_l2a`` (20).
            num_pass1: number of leading images encoded in the first pass; the
                remaining ``num_timesteps - num_pass1`` form the second pass. Equal
                halves are fastest (unequal halves force the encoder's masked
                path).
            embedding_dim: per-token encoder embedding size (768 for BASE).
            decoder_stages: per-task conv decoder definition (see _make_decoder).
                The number of 2x upsamples (len - 1) must equal log2(patch_size)
                so outputs are full resolution. Required.
            binary_loss_weight: multiplier applied to the binary change loss.
            binary_mode: ``"mean"`` or ``"breakpoint"`` (see module docstring).
            temporal_aggregation: ``"mean"``, ``"diff"``, or ``"attn"`` pooling of
                the per-timestep tokens into the segmentation feature.
            season_embed: add month-of-year embeddings to the tokens.
            temporal_depth: number of temporal transformer layers (0 disables).
            temporal_heads: attention heads of the temporal transformer.
            dim_feedforward: FFN hidden size of the temporal transformer.
            temporal_pos_enc: add a learned positional embedding over the T
                chronological slots before the temporal transformer. Requires
                ``temporal_depth > 0``.
        """
        super().__init__()
        if decoder_stages is None:
            raise ValueError("decoder_stages must be specified")
        if not 0 < num_pass1 < num_timesteps:
            raise ValueError(
                f"num_pass1 must be in (0, {num_timesteps}), got {num_pass1}"
            )
        if binary_mode not in ("mean", "breakpoint"):
            raise ValueError(f"unknown binary_mode {binary_mode!r}")
        if temporal_aggregation not in ("mean", "diff", "attn"):
            raise ValueError(f"unknown temporal_aggregation {temporal_aggregation!r}")
        if temporal_pos_enc and temporal_depth <= 0:
            raise ValueError("temporal_pos_enc requires temporal_depth > 0")

        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.num_pass1 = num_pass1
        self.binary_loss_weight = binary_loss_weight
        self.binary_mode = binary_mode
        self.temporal_aggregation = temporal_aggregation
        self.season_embed = season_embed

        self.num_classes_binary = num_classes_binary
        self.num_classes_src = num_classes_src
        self.num_classes_dst = num_classes_dst
        self.num_classes_pre_change = num_classes_pre_change
        self.num_classes_post_change = num_classes_post_change

        # Optional month-of-year embedding added to every token.
        if season_embed:
            self.month_mlp = nn.Sequential(
                nn.Linear(2, embedding_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim // 4, embedding_dim),
            )

        # Optional temporal transformer over the T tokens at each spatial location.
        if temporal_depth > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=temporal_heads,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            )
            self.temporal_encoder: nn.TransformerEncoder | None = nn.TransformerEncoder(
                encoder_layer, num_layers=temporal_depth
            )
            if temporal_pos_enc:
                self.temporal_pos: nn.Parameter | None = nn.Parameter(
                    torch.randn(1, num_timesteps, embedding_dim) * 0.02
                )
            else:
                self.temporal_pos = None
        else:
            self.temporal_encoder = None
            self.temporal_pos = None

        # Learned attention pooling over time (temporal_aggregation="attn").
        if temporal_aggregation == "attn":
            self.time_pool = nn.Linear(embedding_dim, 1)

        # Binary change pathway.
        if binary_mode == "mean":
            self.decoder_binary = _make_decoder(
                embedding_dim, decoder_stages, num_classes_binary
            )
        else:
            # Shared scorer applied to |after - before| at every split.
            self.split_proj = nn.Sequential(
                nn.Conv2d(embedding_dim, BREAKPOINT_HIDDEN, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(BREAKPOINT_HIDDEN, BREAKPOINT_HIDDEN, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            # Scalar per-split score for the split-attention over breakpoints
            # (used to pick the before/after aggregates for src/dst).
            self.split_score = nn.Conv2d(BREAKPOINT_HIDDEN, 1, kernel_size=1)
            if len(BREAKPOINT_EVIDENCE_STAGES) != len(decoder_stages):
                raise ValueError(
                    "decoder_stages must have "
                    f"{len(BREAKPOINT_EVIDENCE_STAGES)} stages for binary_mode="
                    f"'breakpoint', got {len(decoder_stages)}"
                )
            self.evidence_decoder = _make_decoder(
                BREAKPOINT_HIDDEN, BREAKPOINT_EVIDENCE_STAGES, 1
            )

        # Segmentation decoders on the (pooled or breakpoint-weighted) feature.
        self.decoder_src = _make_decoder(embedding_dim, decoder_stages, num_classes_src)
        self.decoder_dst = _make_decoder(embedding_dim, decoder_stages, num_classes_dst)
        self.decoder_pre_change = _make_decoder(
            embedding_dim, decoder_stages, num_classes_pre_change
        )
        self.decoder_post_change = _make_decoder(
            embedding_dim, decoder_stages, num_classes_post_change
        )

        # Per-token timestamp heads producing one logit per timestep.
        self.start_head = nn.Linear(embedding_dim, 1)
        self.end_head = nn.Linear(embedding_dim, 1)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _per_timestep_features(self, context: ModelContext) -> torch.Tensor:
        """Encode the stack in two passes and return (B, C, H, W, T) tokens.

        Each sample's ``sentinel2_l2a`` is split at ``num_pass1``; the historical
        halves of all samples followed by the recent halves of all samples are
        encoded in one batched encoder call, and the per-timestep tokens are
        concatenated back along time in chronological order.
        """
        n = self.num_pass1
        pass1_inputs: list[dict[str, Any]] = []
        pass2_inputs: list[dict[str, Any]] = []
        for inp in context.inputs:
            image: RasterImage = inp[INPUT_KEY]
            if image.image.shape[1] != self.num_timesteps:
                raise ValueError(
                    f"expected {self.num_timesteps} timesteps in {INPUT_KEY}, got "
                    f"{image.image.shape[1]}"
                )
            ts = image.timestamps
            pass1_inputs.append(
                {
                    INPUT_KEY: RasterImage(
                        image=image.image[:, :n], timestamps=ts[:n] if ts else None
                    )
                }
            )
            pass2_inputs.append(
                {
                    INPUT_KEY: RasterImage(
                        image=image.image[:, n:], timestamps=ts[n:] if ts else None
                    )
                }
            )

        batch_size = len(context.inputs)
        sub_context = ModelContext(
            inputs=pass1_inputs + pass2_inputs,
            metadatas=list(context.metadatas) * 2,
        )
        tokens = self.encoder(sub_context).feature_maps[0]  # (2B, C, H, W, T_max)
        # Preserve anything the encoder recorded (e.g. tokens-in-batch).
        context.context_dict.update(sub_context.context_dict)

        # With unequal halves the encoder pads the shorter half with missing
        # tokens at the end, so slice each half to its real length.
        feat1 = tokens[:batch_size, ..., :n]
        feat2 = tokens[batch_size:, ..., : self.num_timesteps - n]
        return torch.cat([feat1, feat2], dim=-1)

    def _month_embedding(
        self, context: ModelContext, T: int, ref: torch.Tensor
    ) -> torch.Tensor:
        """Build (B, C, 1, 1, T) month-of-year embeddings from input timestamps."""
        rows = []
        for input_dict in context.inputs:
            image = input_dict.get(INPUT_KEY)
            feats = torch.zeros(T, 2)
            if isinstance(image, RasterImage) and image.timestamps is not None:
                for t, (t0, t1) in enumerate(image.timestamps[:T]):
                    mid = t0 + (t1 - t0) / 2
                    frac = (mid.month - 1 + (mid.day - 1) / 31.0) / 12.0
                    feats[t, 0] = math.sin(2 * math.pi * frac)
                    feats[t, 1] = math.cos(2 * math.pi * frac)
            rows.append(feats)
        months = torch.stack(rows, dim=0).to(device=ref.device, dtype=ref.dtype)
        emb = self.month_mlp(months)  # (B, T, C)
        return emb.permute(0, 2, 1)[:, :, None, None, :]  # (B, C, 1, 1, T)

    def _apply_temporal_encoder(self, feature: torch.Tensor) -> torch.Tensor:
        """Run the temporal transformer over the T tokens at each location."""
        assert self.temporal_encoder is not None
        b, c, h, w, t = feature.shape
        x = feature.permute(0, 2, 3, 4, 1).reshape(b * h * w, t, c)
        if self.temporal_pos is not None:
            x = x + self.temporal_pos
        x = self.temporal_encoder(x)
        return x.reshape(b, h, w, t, c).permute(0, 4, 1, 2, 3)

    def _pool_time(self, feature: torch.Tensor) -> torch.Tensor:
        """Aggregate (B, C, H, W, T) tokens over time -> (B, C, H, W)."""
        if self.temporal_aggregation == "diff":
            return feature[..., -1] - feature[..., 0]
        if self.temporal_aggregation == "attn":
            b, c, h, w, t = feature.shape
            tokens = feature.permute(0, 2, 3, 4, 1).reshape(b * h * w, t, c)
            weights = torch.softmax(self.time_pool(tokens), dim=1)  # (N, T, 1)
            pooled = (tokens * weights).sum(dim=1)  # (N, C)
            return pooled.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return feature.mean(dim=-1)

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
                "pre_change", "post_change", "timestamps" keys.

        Returns:
            ModelOutput with per-task outputs and losses.
        """
        feature = self._per_timestep_features(context)  # (B, C, H, W, T)

        if self.season_embed:
            feature = feature + self._month_embedding(
                context, feature.shape[-1], feature
            )
        if self.temporal_encoder is not None:
            feature = self._apply_temporal_encoder(feature)

        pooled = self._pool_time(feature)  # (B, C, H, W)

        if self.binary_mode == "mean":
            logits_binary = self.decoder_binary(pooled)
            src_in = pooled
            dst_in = pooled
        else:
            bp_feat, src_in, dst_in = self._breakpoint_features(feature)
            evidence = self.evidence_decoder(bp_feat)  # (B, 1, H', W')
            logits_binary = torch.cat(
                [
                    torch.full_like(evidence, NODATA_LOGIT),
                    torch.zeros_like(evidence),
                    evidence,
                ],
                dim=1,
            )

        logits_src = self.decoder_src(src_in)
        logits_dst = self.decoder_dst(dst_in)
        change_logits = {
            "pre_change": self.decoder_pre_change(pooled),
            "post_change": self.decoder_post_change(pooled),
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
            losses["binary_cls"] = self.binary_loss_weight * self._balanced_binary_loss(
                logits_binary, targets
            )
            losses["src_cls"] = self._seg_loss(logits_src, targets, "src")
            losses["dst_cls"] = self._seg_loss(logits_dst, targets, "dst")
            for name, logits in change_logits.items():
                if name in targets[0]:
                    losses[f"{name}_cls"] = self._seg_loss(logits, targets, name)
            losses["start_ce"] = self._timestamp_ce(start_logits, targets, "start")
            losses["end_ce"] = self._timestamp_ce(end_logits, targets, "end")

        outputs: list[dict[str, Any]] = []
        for i in range(len(context.inputs)):
            ts_image = context.inputs[i].get(INPUT_KEY)
            timestep_days = (
                timestamps_to_days(ts_image.timestamps)
                if isinstance(ts_image, RasterImage) and ts_image.timestamps is not None
                else None
            )
            outputs.append(
                {
                    "binary": F.softmax(logits_binary[i], dim=0),
                    "src": F.softmax(logits_src[i], dim=0),
                    "dst": F.softmax(logits_dst[i], dim=0),
                    **{
                        name: F.softmax(change_logits[name][i], dim=0)
                        for name in change_logits
                    },
                    "timestamps": {
                        "start": F.softmax(start_logits[i], dim=0),
                        "end": F.softmax(end_logits[i], dim=0),
                    },
                    "timestep_days": timestep_days,
                }
            )

        return ModelOutput(outputs=outputs, loss_dict=losses)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

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
