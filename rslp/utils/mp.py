"""Multi-processing utilities."""

import multiprocessing

import torch.multiprocessing


def init_mp() -> None:
    """Set start method to spawn and enforce file_system sharing."""
    # torch.multiprocessing.set_sharing_strategy("file_system")
    multiprocessing.set_start_method("spawn", force=True)
