"""Torch helpers that turn per-timestep timestamp predictions into two day bands.

The LCC change models predict change timing per timestep, either as separate
start/end distributions or as a per-timestep "in change window" membership map.
For the ``output_change`` raster we collapse those into exactly two per-pixel
bands -- the predicted pre-change and post-change dates, encoded as integer days
since :data:`~rslp.change_finder_v2.lcc_model.timestamp_encoding.TIMESTAMP_EPOCH`.

The per-timestep day values themselves come from the model output
(``raw_output["timestep_days"]``), which the model forward passes through from
its input image timestamps (``process_output``/``SampleMetadata`` have no access
to the per-timestep acquisition dates).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .timestamp_encoding import MAX_DAY_VALUE


def _days_tensor(timestep_days: Sequence[int], reference: torch.Tensor) -> torch.Tensor:
    """Build a (T,) float tensor of per-timestep day-values on ``reference``'s device."""
    return torch.tensor(
        list(timestep_days), dtype=torch.float32, device=reference.device
    )


def _stack_day_bands(pre_days: torch.Tensor, post_days: torch.Tensor) -> torch.Tensor:
    """Clamp and stack per-pixel pre/post day maps into a (2, H, W) tensor."""
    bands = torch.stack([pre_days, post_days], dim=0)
    return bands.clamp(0, MAX_DAY_VALUE).round()


def start_end_day_bands(
    start_probs: torch.Tensor,
    end_probs: torch.Tensor,
    timestep_days: Sequence[int],
) -> torch.Tensor:
    """Two day bands from start/end timestep distributions via per-pixel argmax.

    Args:
        start_probs: (T, H, W) softmax over timesteps for the change-start boundary.
        end_probs: (T, H, W) softmax over timesteps for the change-end boundary.
        timestep_days: per-timestep day-values (length T).

    Returns:
        (2, H, W) tensor of clamped day-values (pre, post).
    """
    days = _days_tensor(timestep_days, start_probs)
    pre_idx = start_probs.argmax(dim=0)  # (H, W)
    post_idx = end_probs.argmax(dim=0)  # (H, W)
    return _stack_day_bands(days[pre_idx], days[post_idx])


def membership_day_bands(
    membership: torch.Tensor,
    timestep_days: Sequence[int],
    threshold: float = 0.5,
) -> torch.Tensor:
    """Two day bands from a per-timestep membership map.

    The pre-change date is the earliest timestep whose membership exceeds
    ``threshold`` and the post-change date is the latest such timestep. Pixels
    with no timestep above threshold fall back to the argmax timestep.

    Args:
        membership: (T, H, W) per-timestep "in change window" probability.
        timestep_days: per-timestep day-values (length T).
        threshold: membership threshold for the pre/post boundaries.

    Returns:
        (2, H, W) tensor of clamped day-values (pre, post).
    """
    days = _days_tensor(timestep_days, membership)
    t = membership.shape[0]
    idx = torch.arange(t, device=membership.device).view(t, 1, 1)
    over = membership > threshold  # (T, H, W)
    any_over = over.any(dim=0)  # (H, W)

    earliest = torch.where(over, idx, torch.full_like(idx, t)).amin(dim=0)  # (H, W)
    latest = torch.where(over, idx, torch.full_like(idx, -1)).amax(dim=0)  # (H, W)
    fallback = membership.argmax(dim=0)  # (H, W)

    pre_idx = torch.where(any_over, earliest, fallback)
    post_idx = torch.where(any_over, latest, fallback)
    return _stack_day_bands(days[pre_idx], days[post_idx])
