"""LCC change model: OlmoEarth encoding with per-timestep token heads.

Input is ``sentinel2_l2a`` with ``num_timesteps`` images (16 quarterly + 6 frequent
= 22, built by ``transforms.StackSampler``). The encoder runs with
``token_pooling=False`` so per-timestep tokens are preserved.

By default encoding is done in two passes: the stack is split at ``num_pass1``
into a historical half and a recent half, both halves are encoded in a single
batched encoder call (``2B`` samples of ``num_pass1`` /
``num_timesteps - num_pass1`` images), and the resulting tokens are concatenated
back along time to give ``(B, C, H, W, num_timesteps)`` features. Encoder
attention is quadratic in the token count, so two 11-image passes cost about half
the attention FLOPs of one 22-image pass; the price is that tokens from the two
halves do not attend to each other inside the encoder. With ``num_pass1=None``
the whole stack is encoded in one pass instead, so every token can attend to
every other, at roughly double the encoder attention cost and memory.

Downstream heads all operate on the per-timestep tokens:

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
      scorer; evidence is max-pooled over splits, so "change" can only be
      expressed as before-vs-after dissimilarity at some breakpoint. The scan
      also yields split-attention-weighted before/after aggregates that the
      other heads may consume (below).
- ``src_dst_input``: which feature the src/dst land cover decoders read.
  ``"pooled"`` uses the time-pooled feature for both; ``"breakpoint"`` decodes
  src from the weighted before aggregate and dst from the weighted after
  aggregate so each looks at the correct side of the breakpoint. ``"auto"``
  (default) picks ``"breakpoint"`` iff ``binary_mode == "breakpoint"``.
- ``change_category_input``: which feature the pre/post change-category
  decoders read. ``"pooled"`` (default) uses the time-pooled feature;
  ``"breakpoint"`` concatenates the weighted before and after aggregates so the
  heads see both sides of the transition (a category is a before->after
  transition, so a single side cannot separate e.g. "new_building" from "none"
  in an already built-up area).
- ``ts_start`` / ``ts_end``: a per-token linear logit over the T timesteps, upsampled
  to full resolution, trained with cross-entropy at change pixels. These are
  segmentation-style outputs/targets (T classes = timestep indices), matching the
  ``SegmentationTask`` entries of the same names in the task config.
- ``ts_input``: ``"token"`` (default) uses the per-token linear heads above.
  ``"breakpoint"`` derives the timestep logits from the breakpoint scan instead:
  the start/end logit of split t (between images t and t+1) is the scan's split
  score plus a learned offset from the split's scorer hidden feature and the two
  tokens either side of it. ts_start index t (last image before the change)
  maps to split t, ts_end index t (first image after it) to split t-1; the one
  index without a split (start = last image, end = first image) gets a logit
  from that edge token. The timestep labels then supervise the same split
  attention that picks the src/dst and change-category aggregates.

Annual embedding history (``history_years`` K > 0): the ``v13_embeddings`` input
holds K annual OlmoEarth v1.3 embeddings (built by the stack transforms). Each year
is dequantized, averaged over the patch_size x patch_size pixels of each token
(ignoring nodata pixels) and projected by a small MLP plus a learned per-slot
embedding. They can be used in three ways, independently:

- ``history_context``: a per-location history vector (masked mean and mean absolute
  deviation of the K year features, through an MLP) is concatenated onto the input
  of the breakpoint split scorer and of the src/dst and change-category decoders,
  so change evidence can depend on what the pixel normally looks like. The scan
  itself and the timestep heads still see only the image tokens.
- ``history_memory``: the K year features become extra tokens of the temporal
  transformer (requires ``temporal_depth > 0``), with nodata years masked out as
  keys. By default they are dropped after the transformer, so the scan, decoders
  and timestep heads are unchanged; the image tokens have read the history.
- ``history_in_scan`` (requires ``history_memory``): the year tokens are kept after
  the transformer as the earliest K timesteps of the breakpoint scan and of the
  timestep heads (so the heads have K + T classes; the transform must index the
  targets over the same timeline).

``temporal_time_enc`` replaces the learned per-slot positional embedding of the
temporal transformer with an embedding of each token's actual time: sin/cos of its
age (days before the last image) at several periods plus month-of-year. Year
tokens use the year's midpoint and also get a learned type embedding.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.models.olmoearth_pretrain.model import OlmoEarth
from rslearn.train.model_context import ModelContext, ModelOutput, RasterImage

from .components import balanced_loss
from .timestamp_encoding import timestamps_to_days
from .transforms import (
    EMBEDDING_KEY,
    EMBEDDING_NODATA,
    INPUT_KEY,
    TS_END_KEY,
    TS_START_KEY,
)

# A stage is a list of (out_channels, kernel_size) conv specs.
StageSpec = list[tuple[int, int]]

# Fixed logit for the (never-supervised) nodata channel of the breakpoint binary head.
NODATA_LOGIT = -10.0

# Hidden width of the shared breakpoint split scorer and the conv stages of the
# 1-channel evidence decoder (which must upsample as many times as decoder_stages).
BREAKPOINT_HIDDEN = 256
BREAKPOINT_EVIDENCE_STAGES: list[StageSpec] = [[(256, 3)], [(128, 3)], [(64, 3)]]

# Periods (days) of the sin/cos features of a token's age used by temporal_time_enc;
# two month-of-year features are added to these.
TIME_ENC_PERIODS = (30.0, 91.0, 365.0, 730.0, 1461.0)
TIME_ENC_DIM = 2 * len(TIME_ENC_PERIODS) + 2


def _midpoints(timestamps: list[tuple[Any, Any]]) -> list[Any]:
    return [t0 + (t1 - t0) / 2 for t0, t1 in timestamps]


def _time_features(timestamps: list[tuple[Any, Any]], ref: Any) -> torch.Tensor:
    """(T, TIME_ENC_DIM) features of each timestep's midpoint relative to ``ref``."""
    rows = []
    for mid in _midpoints(timestamps):
        age = (ref - mid).total_seconds() / 86400.0
        feats = []
        for period in TIME_ENC_PERIODS:
            feats += [math.sin(2 * math.pi * age / period), math.cos(2 * math.pi * age / period)]
        frac = (mid.month - 1 + (mid.day - 1) / 31.0) / 12.0
        feats += [math.sin(2 * math.pi * frac), math.cos(2 * math.pi * frac)]
        rows.append(feats)
    return torch.tensor(rows, dtype=torch.float32)


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
        num_classes_src: int = 14,
        num_classes_dst: int = 14,
        num_classes_pre_change: int = 8,
        num_classes_post_change: int = 21,
        num_timesteps: int = 22,
        num_pass1: int | None = 11,
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
        src_dst_input: str = "auto",
        change_category_input: str = "pooled",
        temporal_time_enc: bool = False,
        history_years: int = 0,
        history_context: bool = False,
        history_memory: bool = False,
        history_in_scan: bool = False,
        history_input_dim: int = 128,
        history_hidden: int = 256,
        ts_input: str = "token",
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
                (including nodata and "none").
            num_classes_post_change: number of post_change_category classes
                (including nodata and "none").
            num_timesteps: number of input timesteps in ``sentinel2_l2a`` (22).
            num_pass1: number of leading images encoded in the first pass; the
                remaining ``num_timesteps - num_pass1`` form the second pass. Equal
                halves are fastest (unequal halves force the encoder's masked
                path). ``None`` encodes all ``num_timesteps`` images in a single
                pass (see module docstring).
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
            src_dst_input: ``"auto"``, ``"pooled"``, or ``"breakpoint"`` (see
                module docstring). ``"breakpoint"`` requires
                ``binary_mode="breakpoint"``.
            change_category_input: ``"pooled"`` or ``"breakpoint"`` (see module
                docstring). ``"breakpoint"`` requires ``binary_mode="breakpoint"``.
            temporal_time_enc: embed each token's actual time before the temporal
                transformer (see module docstring). Requires ``temporal_depth > 0``;
                exclusive with ``temporal_pos_enc``.
            history_years: number of annual embedding years K in the
                ``v13_embeddings`` input (0 = not used).
            history_context: feed a per-location history vector to the split
                scorer and decoders.
            history_memory: add the year tokens to the temporal transformer.
            history_in_scan: keep the year tokens in the breakpoint scan and
                timestep heads. Requires ``history_memory``.
            history_input_dim: dimensions of each annual embedding (128).
            history_hidden: width of the year features and history vector.
            ts_input: ``"token"`` or ``"breakpoint"`` (see module docstring).
                ``"breakpoint"`` requires ``binary_mode="breakpoint"``.
        """
        super().__init__()
        if decoder_stages is None:
            raise ValueError("decoder_stages must be specified")
        if num_pass1 is not None and not 0 < num_pass1 < num_timesteps:
            raise ValueError(
                f"num_pass1 must be None or in (0, {num_timesteps}), got {num_pass1}"
            )
        if binary_mode not in ("mean", "breakpoint"):
            raise ValueError(f"unknown binary_mode {binary_mode!r}")
        if temporal_aggregation not in ("mean", "diff", "attn"):
            raise ValueError(f"unknown temporal_aggregation {temporal_aggregation!r}")
        if temporal_pos_enc and temporal_depth <= 0:
            raise ValueError("temporal_pos_enc requires temporal_depth > 0")
        if src_dst_input not in ("auto", "pooled", "breakpoint"):
            raise ValueError(f"unknown src_dst_input {src_dst_input!r}")
        if change_category_input not in ("pooled", "breakpoint"):
            raise ValueError(f"unknown change_category_input {change_category_input!r}")
        if src_dst_input == "auto":
            src_dst_input = "breakpoint" if binary_mode == "breakpoint" else "pooled"
        if src_dst_input == "breakpoint" and binary_mode != "breakpoint":
            raise ValueError(
                "src_dst_input='breakpoint' requires binary_mode='breakpoint'"
            )
        if change_category_input == "breakpoint" and binary_mode != "breakpoint":
            raise ValueError(
                "change_category_input='breakpoint' requires binary_mode='breakpoint'"
            )
        if temporal_time_enc and (temporal_depth <= 0 or temporal_pos_enc):
            raise ValueError(
                "temporal_time_enc requires temporal_depth > 0 and no temporal_pos_enc"
            )
        uses_history = history_context or history_memory or history_in_scan
        if uses_history != (history_years > 0):
            raise ValueError(
                "history_years > 0 requires (and is required by) history_context, "
                "history_memory or history_in_scan"
            )
        if history_memory and temporal_depth <= 0:
            raise ValueError("history_memory requires temporal_depth > 0")
        if history_in_scan and not history_memory:
            raise ValueError("history_in_scan requires history_memory")
        if ts_input not in ("token", "breakpoint"):
            raise ValueError(f"unknown ts_input {ts_input!r}")
        if ts_input == "breakpoint" and binary_mode != "breakpoint":
            raise ValueError("ts_input='breakpoint' requires binary_mode='breakpoint'")

        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_timesteps = num_timesteps
        self.num_pass1 = num_pass1
        self.binary_loss_weight = binary_loss_weight
        self.binary_mode = binary_mode
        self.temporal_aggregation = temporal_aggregation
        self.season_embed = season_embed
        self.src_dst_input = src_dst_input
        self.change_category_input = change_category_input
        self.ts_input = ts_input
        self.temporal_time_enc = temporal_time_enc
        self.history_years = history_years
        self.history_context = history_context
        self.history_memory = history_memory
        self.history_in_scan = history_in_scan

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
            # No nested tensors: with a key padding mask they would zero the outputs
            # of masked (nodata) year tokens in eval mode only.
            self.temporal_encoder: nn.TransformerEncoder | None = nn.TransformerEncoder(
                encoder_layer, num_layers=temporal_depth, enable_nested_tensor=False
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

        if temporal_time_enc:
            self.time_mlp = nn.Sequential(
                nn.Linear(TIME_ENC_DIM, embedding_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embedding_dim // 4, embedding_dim),
            )

        # Annual embedding history.
        ctx_dim = 0
        if history_years > 0:
            self.hist_proj = nn.Sequential(
                nn.Linear(history_input_dim, history_hidden),
                nn.GELU(),
                nn.Linear(history_hidden, history_hidden),
                nn.LayerNorm(history_hidden),
            )
            self.hist_slot = nn.Parameter(
                torch.randn(history_years, history_hidden) * 0.02
            )
        if history_context:
            self.hist_ctx = nn.Sequential(
                nn.Linear(2 * history_hidden, history_hidden),
                nn.GELU(),
                nn.Linear(history_hidden, history_hidden),
            )
            # Used at locations with no valid year.
            self.hist_null = nn.Parameter(torch.zeros(history_hidden))
            ctx_dim = history_hidden
        if history_memory:
            self.hist_to_token = nn.Linear(history_hidden, embedding_dim)
            self.hist_type = nn.Parameter(torch.randn(embedding_dim) * 0.02)

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
                nn.Conv2d(embedding_dim + ctx_dim, BREAKPOINT_HIDDEN, kernel_size=1),
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
        self.decoder_src = _make_decoder(
            embedding_dim + ctx_dim, decoder_stages, num_classes_src
        )
        self.decoder_dst = _make_decoder(
            embedding_dim + ctx_dim, decoder_stages, num_classes_dst
        )
        # The change-category decoders take concat(before, after) in breakpoint mode.
        category_in_dim = (
            2 * embedding_dim
            if change_category_input == "breakpoint"
            else embedding_dim
        ) + ctx_dim
        self.decoder_pre_change = _make_decoder(
            category_in_dim, decoder_stages, num_classes_pre_change
        )
        self.decoder_post_change = _make_decoder(
            category_in_dim, decoder_stages, num_classes_post_change
        )

        if ts_input == "token":
            # Per-token timestamp heads producing one logit per timestep.
            self.start_head = nn.Linear(embedding_dim, 1)
            self.end_head = nn.Linear(embedding_dim, 1)
        else:
            # Start/end offsets added to each split's score, from the split's scorer
            # hidden feature and the tokens either side of it.
            self.ts_split_offset = nn.Conv2d(
                BREAKPOINT_HIDDEN + 2 * embedding_dim, 2, kernel_size=1
            )
            # Logits of the index without a split: start at the last image, end at
            # the first image.
            self.ts_start_last = nn.Linear(embedding_dim, 1)
            self.ts_end_first = nn.Linear(embedding_dim, 1)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _per_timestep_features(self, context: ModelContext) -> torch.Tensor:
        """Encode the stack and return (B, C, H, W, T) per-timestep tokens.

        With ``num_pass1=None`` the full stack is encoded in one encoder call.
        Otherwise each sample's ``sentinel2_l2a`` is split at ``num_pass1``; the
        historical halves of all samples followed by the recent halves of all
        samples are encoded in one batched encoder call, and the per-timestep
        tokens are concatenated back along time in chronological order.
        """
        for inp in context.inputs:
            image: RasterImage = inp[INPUT_KEY]
            if image.image.shape[1] != self.num_timesteps:
                raise ValueError(
                    f"expected {self.num_timesteps} timesteps in {INPUT_KEY}, got "
                    f"{image.image.shape[1]}"
                )

        if self.num_pass1 is None:
            tokens = self.encoder(context).feature_maps[0]  # (B, C, H, W, T)
            return tokens[..., : self.num_timesteps]

        n = self.num_pass1
        pass1_inputs: list[dict[str, Any]] = []
        pass2_inputs: list[dict[str, Any]] = []
        for inp in context.inputs:
            image = inp[INPUT_KEY]
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

    def _time_embedding(
        self,
        timestamps: list[list[tuple[Any, Any]]],
        refs: list[Any],
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """(B, T, C) time embeddings of per-sample timestamps (temporal_time_enc)."""
        feats = torch.stack(
            [_time_features(ts, ref) for ts, ref in zip(timestamps, refs)], dim=0
        ).to(device=ref_tensor.device)
        return self.time_mlp(feats).to(ref_tensor.dtype)

    @staticmethod
    def _input_timestamps(
        context: ModelContext, key: str
    ) -> list[list[tuple[Any, Any]]]:
        out = []
        for inp in context.inputs:
            image = inp[key]
            if not isinstance(image, RasterImage) or image.timestamps is None:
                raise ValueError(f"{key} must have timestamps")
            out.append(list(image.timestamps))
        return out

    def _history_features(
        self, context: ModelContext, h: int, w: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-year token features of the annual embeddings.

        Returns:
            (features, valid): (B, K, hidden, h, w) year features at the token grid
            and the (B, K, h, w) mask of tokens with at least one valid pixel.
        """
        q = torch.stack(
            [inp[EMBEDDING_KEY].image for inp in context.inputs], dim=0
        ).float()  # (B, D, K, H, W)
        B, D, K, H, W = q.shape
        if K != self.history_years:
            raise ValueError(f"expected {self.history_years} embedding years, got {K}")
        if H % h or W % w:
            raise ValueError(f"embedding size {(H, W)} not a multiple of {(h, w)}")
        valid_px = (q[:, 0] != EMBEDDING_NODATA).float()  # (B, K, H, W)
        # Signed power-2 dequantization (OlmoEarth embedding store convention).
        deq = torch.sign(q) * (q.abs() / 127.5) ** 2 * valid_px[:, None]
        p = (H // h, W // w)
        summed = F.avg_pool2d(deq.permute(0, 2, 1, 3, 4).reshape(B * K, D, H, W), p)
        frac = F.avg_pool2d(valid_px.reshape(B * K, 1, H, W), p)
        mean = (summed / frac.clamp_min(1e-6)).reshape(B, K, D, h, w)
        valid = (frac > 0).reshape(B, K, h, w)
        feat = self.hist_proj(mean.permute(0, 1, 3, 4, 2))  # (B, K, h, w, hidden)
        feat = feat + self.hist_slot[None, :, None, None, :]
        return feat.permute(0, 1, 4, 2, 3), valid

    def _history_context_vector(
        self, feat: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """(B, hidden, h, w) history vector from the masked year features."""
        v = valid[:, :, None].to(feat.dtype)  # (B, K, 1, h, w)
        n = v.sum(dim=1)  # (B, 1, h, w)
        mean = (feat * v).sum(dim=1) / n.clamp_min(1)
        dev = ((feat - mean[:, None]).abs() * v).sum(dim=1) / n.clamp_min(1)
        ctx = self.hist_ctx(torch.cat([mean, dev], dim=1).permute(0, 2, 3, 1))
        ctx = ctx.permute(0, 3, 1, 2)
        null = self.hist_null[None, :, None, None].to(ctx.dtype).expand_as(ctx)
        return torch.where(n > 0, ctx, null)

    def _apply_temporal_encoder(
        self,
        feature: torch.Tensor,
        context: ModelContext,
        history: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Run the temporal transformer over the T tokens at each location.

        With ``history_memory`` the K year tokens are prepended (nodata years
        masked as keys) and, unless ``history_in_scan``, dropped from the output.
        """
        assert self.temporal_encoder is not None
        b, c, h, w, t = feature.shape
        # The encoder may return bfloat16 tokens; the transformer runs in float32.
        x = feature.float().permute(0, 2, 3, 4, 1).reshape(b * h * w, t, c)
        if self.temporal_pos is not None:
            x = x + self.temporal_pos
        refs = None
        if self.temporal_time_enc:
            raw_ts = self._input_timestamps(context, INPUT_KEY)
            refs = [_midpoints(ts)[-1] for ts in raw_ts]
            x = x + self._time_embedding(raw_ts, refs, x).repeat_interleave(
                h * w, dim=0
            )
        mask = None
        k = 0
        if self.history_memory:
            assert history is not None
            feat, valid = history
            k = feat.shape[1]
            tok = self.hist_to_token(feat.permute(0, 3, 4, 1, 2).reshape(b * h * w, k, -1))
            tok = tok + self.hist_type.to(tok.dtype)
            if self.temporal_time_enc:
                assert refs is not None
                emb_ts = self._input_timestamps(context, EMBEDDING_KEY)
                tok = tok + self._time_embedding(emb_ts, refs, tok).repeat_interleave(
                    h * w, dim=0
                )
            x = torch.cat([tok, x], dim=1)
            pad = ~valid.permute(0, 2, 3, 1).reshape(b * h * w, k)
            mask = torch.cat(
                [pad, torch.zeros(b * h * w, t, dtype=torch.bool, device=pad.device)],
                dim=1,
            )
        x = self.temporal_encoder(x, src_key_padding_mask=mask)
        if self.history_memory and not self.history_in_scan:
            x = x[:, k:]
        tt = x.shape[1]
        return x.reshape(b, h, w, tt, c).permute(0, 4, 1, 2, 3)

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
        self,
        feature: torch.Tensor,
        ctx: torch.Tensor | None = None,
        return_splits: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Changepoint scan over the T timesteps.

        Args:
            feature: (B, C, H, W, T) per-timestep tokens.
            ctx: optional (B, ctx_dim, H, W) history vector, concatenated onto the
                split scorer input at every split.

            return_splits: also return the per-split scores (B, S, H, W) and
                hidden features (B, S, hidden, H, W).

        Returns:
            (evidence_feat, src_feat, dst_feat): max-pooled per-split hidden
            features (B, hidden, H, W) and the split-attention-weighted
            before/after aggregates (B, C, H, W), plus (scores, hidden) when
            return_splits.
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
        if ctx is not None:
            con = torch.cat([con, ctx.repeat_interleave(S, dim=0)], dim=1)
        hidden = self.split_proj(con)  # (B*S, hidden, H, W)
        scores = self.split_score(hidden)  # (B*S, 1, H, W)
        hidden = hidden.reshape(B, S, -1, H, W)
        scores = scores.reshape(B, S, H, W)

        evidence_feat = hidden.max(dim=1).values
        w = F.softmax(scores, dim=1)  # (B, S, H, W)
        w = w.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H, W, S)
        src_feat = (before * w).sum(dim=-1)
        dst_feat = (after * w).sum(dim=-1)
        if return_splits:
            return evidence_feat, src_feat, dst_feat, scores, hidden
        return evidence_feat, src_feat, dst_feat

    def _breakpoint_ts_logits(
        self, feature: torch.Tensor, scores: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Timestep logits from the breakpoint scan (ts_input="breakpoint").

        Args:
            feature: (B, C, H, W, T) per-timestep tokens.
            scores: (B, S, H, W) split scores, S = T - 1.
            hidden: (B, S, hidden, H, W) split scorer features.

        Returns:
            (start_logits, end_logits), each (B, T, H, W) at token resolution.
        """
        B, C, H, W, T = feature.shape
        S = T - 1
        x = feature.permute(0, 4, 1, 2, 3)  # (B, T, C, H, W)
        con = torch.cat([hidden, x[:, :-1].to(hidden.dtype), x[:, 1:].to(hidden.dtype)], dim=2)
        offset = self.ts_split_offset(con.reshape(B * S, -1, H, W)).reshape(B, S, 2, H, W)
        start_split = scores + offset[:, :, 0]
        end_split = scores + offset[:, :, 1]
        tokens = feature.permute(0, 2, 3, 4, 1)  # (B, H, W, T, C)
        start_last = self.ts_start_last(tokens[:, :, :, -1]).permute(0, 3, 1, 2)
        end_first = self.ts_end_first(tokens[:, :, :, 0]).permute(0, 3, 1, 2)
        start_logits = torch.cat([start_split, start_last.to(start_split.dtype)], dim=1)
        end_logits = torch.cat([end_first.to(end_split.dtype), end_split], dim=1)
        return start_logits, end_logits

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

        if self.season_embed:
            feature = feature + self._month_embedding(
                context, feature.shape[-1], feature
            )
        history = None
        ctx = None
        if self.history_years > 0:
            history = self._history_features(context, feature.shape[2], feature.shape[3])
            if self.history_context:
                ctx = self._history_context_vector(*history)
        if self.temporal_encoder is not None:
            feature = self._apply_temporal_encoder(feature, context, history)

        pooled = self._pool_time(feature)  # (B, C, H, W)

        if self.binary_mode == "mean":
            logits_binary = self.decoder_binary(pooled)
            before_feat = after_feat = None
            split_scores = split_hidden = None
        else:
            bp_feat, before_feat, after_feat, split_scores, split_hidden = (
                self._breakpoint_features(feature, ctx, return_splits=True)
            )
            evidence = self.evidence_decoder(bp_feat)  # (B, 1, H', W')
            logits_binary = torch.cat(
                [
                    torch.full_like(evidence, NODATA_LOGIT),
                    torch.zeros_like(evidence),
                    evidence,
                ],
                dim=1,
            )

        # Route the pooled or breakpoint-weighted features to the segmentation heads.
        if self.src_dst_input == "breakpoint":
            assert before_feat is not None and after_feat is not None
            src_in, dst_in = before_feat, after_feat
        else:
            src_in = dst_in = pooled
        if self.change_category_input == "breakpoint":
            assert before_feat is not None and after_feat is not None
            category_in = torch.cat([before_feat, after_feat], dim=1)  # (B, 2C, H, W)
        else:
            category_in = pooled
        if ctx is not None:
            src_in = torch.cat([src_in, ctx], dim=1)
            dst_in = torch.cat([dst_in, ctx], dim=1)
            category_in = torch.cat([category_in, ctx], dim=1)

        logits_src = self.decoder_src(src_in)
        logits_dst = self.decoder_dst(dst_in)
        change_logits = {
            "pre_change": self.decoder_pre_change(category_in),
            "post_change": self.decoder_post_change(category_in),
        }

        # Timestamp logits over T, upsampled to full resolution.
        if self.ts_input == "breakpoint":
            assert split_scores is not None and split_hidden is not None
            start_logits, end_logits = self._breakpoint_ts_logits(
                feature, split_scores, split_hidden
            )
        else:
            xt = feature.permute(0, 2, 3, 4, 1)  # (B, H, W, T, C)
            start_logits = self.start_head(xt).squeeze(-1).permute(0, 3, 1, 2)
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
            ts_image = context.inputs[i].get(INPUT_KEY)
            timestep_days = (
                timestamps_to_days(ts_image.timestamps)
                if isinstance(ts_image, RasterImage) and ts_image.timestamps is not None
                else None
            )
            if self.history_in_scan and timestep_days is not None:
                # The timestep heads also cover the K year tokens (oldest first).
                emb_image = context.inputs[i][EMBEDDING_KEY]
                timestep_days = timestamps_to_days(emb_image.timestamps) + timestep_days
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
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

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
