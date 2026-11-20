"""Strata (allenai/olmoearth_strata) encoder giving per-timestep tokens for ChangeModel.

Strata's own forward pools time away (one feature per pixel), and its space-time
exchange blocks and spatial path group tokens by (cell, day-of-year bin), which
merges the same season of different years. Change detection needs before/after, so
this wrapper runs the pretrained modules with one bin per stack slot instead:

- pixel path: one token per (pixel, slot) with the pretrained observation, sensor
  and time (day-of-year harmonics + days since the first slot) embeddings; the
  pixel blocks attend over the slots of a pixel, as in pretraining.
- exchange blocks and spatial path: (cell, slot) tokens instead of (cell, doy bin).
  Each slot's bin embedding is the pretrained embedding of the doy bin its date
  falls in, and the st_rope time angle is the slot's day of year (the pretrained
  rotary uses the doy of the bin centre), so the pretrained weights keep their
  meaning while different years stay separate tokens.

Per-timestep output at ``patch_size`` (4): the spatial-path (cell, slot) tokens
through a learned stride ``patch_size / cfg.patch`` conv plus the pixel-path tokens
through a learned stride ``patch_size`` conv, summed and layer-normed ->
``(B, embedding_dim, H/4, W/4, T)``, the same layout as the OlmoEarth wrapper with
``token_pooling=False``.

Supports the configuration of strata_g1n_nd_t_sfpx_rope (spatial_time "bins",
spatial_in "concat", st_rope, no time RoPE, no location input, no pooler depth).
The Strata package is imported from ``strata_src`` (the training image does not
include it). Inputs are raw harmonized Sentinel-2 L2A values in the LCC band order
(B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09); B01/B09 are dropped and the
rest normalized with Strata's statistics, so the config must not apply
OlmoEarthNormalize.
"""

import math
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from rslearn.models.component import FeatureMaps
from rslearn.train.model_context import ModelContext
from rslearn.train.optimizer import OptimizerFactory

from .transforms import INPUT_KEY

# Strata's 10 S2 bands are the first 10 of the LCC input bands, in the same order.
NUM_S2_BANDS = 10


def _slot_st_rope(
    gh: int, gw: int, doy: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strata's st_rope with a per-sample day of year per slot.

    Args:
        gh: cell grid rows.
        gw: cell grid columns.
        doy: (B, T) day of year of each slot.
        head_dim: attention head dimension.

    Returns:
        (cos, sin), each (B, 1, gh*gw*T, head_dim), tokens ordered (row, col, slot).
    """
    b, t = doy.shape
    half = head_dim // 2
    n_t = half // 3
    n_y = (half - n_t) // 2
    n_x = half - n_t - n_y
    dev = doy.device
    r = torch.arange(gh, device=dev, dtype=torch.float32)
    c = torch.arange(gw, device=dev, dtype=torch.float32)
    rr, cc = torch.meshgrid(r, c, indexing="ij")
    fy = 100.0 ** (-torch.arange(n_y, device=dev, dtype=torch.float32) / max(n_y, 1))
    fx = 100.0 ** (-torch.arange(n_x, device=dev, dtype=torch.float32) / max(n_x, 1))
    ks = torch.arange(1, n_t + 1, device=dev, dtype=torch.float32) * (2 * math.pi / 365.25)
    ang_rc = torch.cat([rr.reshape(-1, 1) * fy, cc.reshape(-1, 1) * fx], dim=-1)
    ang_t = doy.float()[..., None] * ks  # (B, T, n_t)
    ang = torch.cat(
        [
            ang_rc[None, :, None].expand(b, -1, t, -1),
            ang_t[:, None].expand(-1, gh * gw, -1, -1),
        ],
        dim=-1,
    ).reshape(b, gh * gw * t, half)
    ang = torch.cat([ang, ang], dim=-1)[:, None]
    return ang.cos(), ang.sin()


class StrataTemporalEncoder(nn.Module):
    """Strata encoder returning per-timestep tokens (see module docstring)."""

    def __init__(
        self,
        checkpoint_path: str,
        strata_src: str,
        patch_size: int = 4,
        embedding_dim: int = 768,
        grad_checkpoint: bool = True,
        autocast_dtype: str | None = "bfloat16",
    ):
        """Create a new StrataTemporalEncoder.

        Args:
            checkpoint_path: Strata encoder checkpoint ({model, step, config}).
            strata_src: directory containing the ``olmoearth_strata`` package.
            patch_size: output token patch size (pixels per side).
            embedding_dim: output channels per token.
            grad_checkpoint: checkpoint the Strata blocks in training.
            autocast_dtype: autocast dtype for the Strata modules, or None.
        """
        super().__init__()
        if strata_src not in sys.path:
            sys.path.insert(0, strata_src)
        from olmoearth_strata.data.normalize import CLIP, STATS
        from olmoearth_strata.model import strata as strata_mod

        st = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        enc_cfg = dict(st["config"]["enc"])
        enc_cfg["sensors"] = tuple(enc_cfg["sensors"])
        cfg = strata_mod.StrataConfig(**enc_cfg)
        if st["config"].get("data", {}).get("s2_log"):
            raise ValueError("s2_log checkpoints are not supported")
        # Plain padded attention (no flash_attn packing, no torch.compile).
        cfg.pixel_pack = False
        cfg.compile_pixel_blocks = False
        cfg.grad_checkpoint = grad_checkpoint
        expected = {
            "spatial_time": "bins",
            "spatial_in": "concat",
            "st_rope": True,
            "time_rope": False,
            "pool_depth": 0,
            "ex_dim": 0,
            "loc_dim": 0,
            "time_days_feature": True,
        }
        for k, v in expected.items():
            if getattr(cfg, k) != v:
                raise ValueError(f"unsupported Strata config {k}={getattr(cfg, k)!r}")
        if cfg.st_exchange_every <= 0 or cfg.spatial_depth <= 0:
            raise ValueError("Strata config needs exchange blocks and a spatial path")
        if patch_size % cfg.patch != 0:
            raise ValueError(f"patch_size {patch_size} not a multiple of {cfg.patch}")

        self.strata = strata_mod.build_encoder(cfg)
        sd = st["model"]
        if any(k.startswith("encoder.") for k in sd):
            sd = {k[len("encoder.") :]: v for k, v in sd.items() if k.startswith("encoder.")}
        self.strata.load_state_dict(sd)
        self.cfg = cfg
        self.time_features = strata_mod.time_features
        self.patch_size = patch_size
        self.clip = CLIP
        mean, std = STATS["s2"]
        self.register_buffer("s2_mean", torch.tensor(mean), persistent=False)
        self.register_buffer("s2_std", torch.tensor(std), persistent=False)
        self.autocast_dtype = (
            getattr(torch, autocast_dtype) if autocast_dtype is not None else None
        )

        k = patch_size // cfg.patch
        self.spatial_proj = nn.Conv2d(
            cfg.spatial_dim, embedding_dim, kernel_size=k, stride=k
        )
        self.pixel_proj = nn.Conv2d(
            cfg.pixel_dim, embedding_dim, kernel_size=patch_size, stride=patch_size
        )
        self.out_norm = nn.LayerNorm(embedding_dim)

    def _inputs(
        self, context: ModelContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalized values (B,T,H,W,10), mask (B,T,H,W), doy and days (B,T)."""
        images, doys, days = [], [], []
        for inp in context.inputs:
            image = inp[INPUT_KEY]
            images.append(image.image[:NUM_S2_BANDS])  # (10, T, H, W)
            if not image.timestamps:
                raise ValueError("StrataTemporalEncoder needs input timestamps")
            mids = [s + (e - s) / 2 for s, e in image.timestamps]
            doys.append([m.timetuple().tm_yday for m in mids])
            days.append([(m - mids[0]).total_seconds() / 86400.0 for m in mids])
        raw = torch.stack(images).float()  # (B, 10, T, H, W)
        dev = raw.device
        mask = (raw > 0).any(dim=1)  # (B, T, H, W); S2 nodata is 0 in every band
        v = raw.permute(0, 2, 3, 4, 1) / 10000.0  # (B, T, H, W, 10)
        v = ((v - self.s2_mean) / self.s2_std).clamp(-self.clip, self.clip)
        doy = torch.tensor(doys, device=dev, dtype=torch.float32)
        day = torch.tensor(days, device=dev, dtype=torch.float32)
        return v, mask, doy, day

    def _cell_slots(
        self, x: torch.Tensor, m: torch.Tensor, b: int, h: int, w: int, t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean of the valid pixel tokens per (cell, slot): (B, C*T, d), has-token (B, C*T)."""
        p = self.cfg.patch
        gh, gw = h // p, w // p
        d = x.shape[-1]
        wgt = m.to(x.dtype)[..., None]  # (B*H*W, T, 1)
        y = (x * wgt).view(b, gh, p, gw, p, t, d).sum((2, 4))  # (B, gh, gw, T, d)
        c = wgt.view(b, gh, p, gw, p, t, 1).sum((2, 4))
        z = (y / c.clamp(min=1.0)).reshape(b, gh * gw * t, d)
        return z, (c[..., 0] > 0).reshape(b, gh * gw * t)

    def _concat_cells(
        self, x: torch.Tensor, m: torch.Tensor, b: int, h: int, w: int, t: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Strata's spatial_in="concat" with one bin per slot: (B, C*T, p*p*cb), (B, C*T)."""
        s = self.strata
        p = self.cfg.patch
        gh, gw = h // p, w // p
        y = s.sp_bottleneck(x)  # (B*H*W, T, cb)
        y = torch.where(m[..., None], y, s.sp_empty.to(y.dtype))
        cb = y.shape[-1]
        y = y.view(b, gh, p, gw, p, t, cb).permute(0, 1, 3, 5, 2, 4, 6)
        valid = m.view(b, gh, p, gw, p, t).any(4).any(2).reshape(b, gh * gw * t)
        return y.reshape(b, gh * gw * t, p * p * cb), valid

    def _attend(
        self,
        block: nn.Module,
        z: torch.Tensor,
        valid: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        vs = valid.clone()
        vs[~valid.any(1), 0] = True
        # No key mask when every token is valid, so SDPA can use flash attention.
        key_mask = None if bool(vs.all()) else vs
        return self.strata._run(block, z, key_mask, rope=rope)

    def _strata_tokens(
        self, context: ModelContext
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
        s, cfg = self.strata, self.cfg
        v, mask, doy, day = self._inputs(context)
        b, t, h, w, _ = v.shape
        p = cfg.patch
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(f"input {h}x{w} not divisible by {self.patch_size}")
        gh, gw = h // p, w // p
        n_bins = cfg.time_bins
        dbin = (doy * (n_bins / 366.0)).long().clamp(0, n_bins - 1)  # (B, T)

        # Pixel path tokens, one per (pixel, slot).
        tok = s.obs_embed["s2"](v) + s.sensor_embed["s2"]
        tf = self.time_features(
            doy[:, :, None, None].expand(b, t, h, w),
            day[:, :, None, None].expand(b, t, h, w),
            cfg.time_harmonics,
        )
        tok = tok + s.time_embed(tf)
        x = tok.permute(0, 2, 3, 1, 4).reshape(b * h * w, t, -1)
        m = mask.permute(0, 2, 3, 1).reshape(b * h * w, t)
        m_safe = m.clone()
        m_safe[~m.any(1), 0] = True
        pix_mask = None if bool(m.all()) else m_safe

        ex = 0
        for i, blk in enumerate(s.pixel_blocks):
            x = s._run(blk, x, pix_mask)
            if (i + 1) % cfg.st_exchange_every == 0:
                z, valid = self._cell_slots(x, m, b, h, w, t)
                d = z.shape[-1]
                pos = s.ex_bin_embed.to(z.dtype)[dbin]  # (B, T, d)
                z = z + pos[:, None].expand(b, gh * gw, t, d).reshape(b, -1, d)
                heads = cfg.ex_heads or cfg.pixel_heads
                rope = _slot_st_rope(gh, gw, doy, d // heads)
                z = self._attend(s.ex_blocks[ex], z, valid, rope)
                back = s.ex_out[ex](z).view(b, gh, gw, t, -1)
                back = back[:, :, None, :, None].expand(b, gh, p, gw, p, t, back.shape[-1])
                x = x + back.reshape(b * h * w, t, -1) * m[..., None].to(back.dtype)
                ex += 1
        x = s.pixel_norm(x)  # (B*H*W, T, pixel_dim)

        # Spatial path over (cell, slot) tokens.
        z, valid = self._concat_cells(x, m, b, h, w, t)
        z = s.to_spatial(z)
        ds = z.shape[-1]
        pos = s.bin_embed.to(z.dtype)[dbin]  # (B, T, ds)
        z = z + pos[:, None].expand(b, gh * gw, t, ds).reshape(b, -1, ds)
        rope = _slot_st_rope(gh, gw, doy, ds // cfg.spatial_heads)
        for blk in s.spatial_blocks:
            z = self._attend(blk, z, valid, rope)
        z = s.spatial_norm(z)  # (B, gh*gw*T, ds)
        x = x * m[..., None].to(x.dtype)
        return x, z, m, b, t, h, w

    def forward(self, context: ModelContext) -> FeatureMaps:
        """Per-timestep tokens (B, embedding_dim, H/patch, W/patch, T)."""
        dev = context.inputs[0][INPUT_KEY].image.device
        if self.autocast_dtype is not None:
            ctx: Any = torch.autocast(device_type=dev.type, dtype=self.autocast_dtype)
        else:
            ctx = torch.autocast(device_type=dev.type, enabled=False)
        with ctx:
            x, z, m, b, t, h, w = self._strata_tokens(context)
        p = self.cfg.patch
        gh, gw = h // p, w // p
        zs = z.float().view(b, gh, gw, t, -1).permute(0, 3, 4, 1, 2)
        zs = self.spatial_proj(zs.reshape(b * t, -1, gh, gw))
        xs = x.float().view(b, h, w, t, -1).permute(0, 3, 4, 1, 2)
        xs = self.pixel_proj(xs.reshape(b * t, -1, h, w))
        out = (zs + xs).view(b, t, -1, zs.shape[-2], zs.shape[-1])
        out = self.out_norm(out.permute(0, 3, 4, 1, 2))  # (B, H', W', T, C)
        return FeatureMaps([out.permute(0, 4, 1, 2, 3)])


class EncoderLRAdamW(OptimizerFactory):
    """AdamW with the encoder at ``encoder_lr_scale`` times the base learning rate."""

    def __init__(
        self,
        lr: float = 1e-4,
        encoder_lr_scale: float = 0.1,
        encoder_prefix: str = "model.encoder.strata.",
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
    ):
        """Create a new EncoderLRAdamW.

        Args:
            lr: base learning rate (heads and the new projection layers).
            encoder_lr_scale: multiplier for parameters under ``encoder_prefix``.
            encoder_prefix: parameter name prefix of the pretrained encoder.
            betas: AdamW betas.
            weight_decay: AdamW weight decay.
        """
        self.lr = lr
        self.encoder_lr_scale = encoder_lr_scale
        self.encoder_prefix = encoder_prefix
        self.betas = betas
        self.weight_decay = weight_decay

    def build(self, lm: Any) -> torch.optim.Optimizer:
        """Build the optimizer."""
        enc, rest = [], []
        for name, param in lm.named_parameters():
            (enc if name.startswith(self.encoder_prefix) else rest).append(param)
        if not enc:
            raise ValueError(f"no parameters under {self.encoder_prefix!r}")
        return torch.optim.AdamW(
            [
                {"params": enc, "lr": self.lr * self.encoder_lr_scale},
                {"params": rest, "lr": self.lr},
            ],
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
