"""Multi-processing utilities."""

import multiprocessing
import os


DEFAULT_MP_CONTEXT = "forkserver"
MP_CONTEXT_ENV_VAR = "RSLEARN_MULTIPROCESSING_CONTEXT"


def init_mp(context: str | None = None) -> None:
    """Set start method for multiprocessing.

    Uses RSLEARN_MULTIPROCESSING_CONTEXT if provided, else defaults to forkserver.
    """
    if context is None:
        context = os.environ.get(MP_CONTEXT_ENV_VAR, DEFAULT_MP_CONTEXT)
    multiprocessing.set_start_method(context, force=True)
