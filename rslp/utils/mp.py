"""Multi-processing utilities."""

import multiprocessing
import os
from typing import Optional


DEFAULT_MP_CONTEXT = "forkserver"
MP_CONTEXT_ENV_VAR = "RSLEARN_MULTIPROCESSING_CONTEXT"
MP_SHARING_STRATEGY_ENV_VAR = "RSLEARN_TORCH_MP_SHARING_STRATEGY"


def init_mp(context: str | None = None, sharing_strategy: Optional[str] = None) -> None:
    """Set start method for multiprocessing.

    Uses RSLEARN_MULTIPROCESSING_CONTEXT if provided, else defaults to forkserver.
    """
    if sharing_strategy is None:
        sharing_strategy = os.environ.get(MP_SHARING_STRATEGY_ENV_VAR)
    if sharing_strategy:
        try:
            import torch.multiprocessing as torch_mp

            torch_mp.set_sharing_strategy(sharing_strategy)
        except Exception:
            # Avoid failing to start if torch is not available yet.
            pass
    if context is None:
        context = os.environ.get(MP_CONTEXT_ENV_VAR, DEFAULT_MP_CONTEXT)
    multiprocessing.set_start_method(context, force=True)
