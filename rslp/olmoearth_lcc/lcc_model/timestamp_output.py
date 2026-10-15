"""Torch helpers that turn per-timestep timestamp predictions into two day bands.

The LCC change model predicts change timing as separate start/end distributions
over the input timesteps. For the ``output_change`` raster we collapse those into
exactly two per-pixel bands -- the predicted pre-change and post-change dates,
encoded as integer days since
:data:`~rslp.olmoearth_lcc.lcc_model.timestamp_encoding.TIMESTAMP_EPOCH`.

The per-timestep day values themselves come from the model output
(``raw_output["timestep_days"]``), which the model forward passes through from
its input image timestamps (``process_output``/``SampleMetadata`` have no access
to the per-timestep acquisition dates).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .timestamp_encoding import MAX_DAY_VALUE


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
    days = torch.tensor(
        list(timestep_days), dtype=torch.float32, device=start_probs.device
    )
    pre_idx = start_probs.argmax(dim=0)  # (H, W)
    post_idx = end_probs.argmax(dim=0)  # (H, W)
    bands = torch.stack([days[pre_idx], days[post_idx]], dim=0)
    return bands.clamp(0, MAX_DAY_VALUE).round()
