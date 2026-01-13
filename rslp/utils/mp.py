"""Multi-processing utilities."""

import multiprocessing

import torch.multiprocessing


def init_mp() -> None:
    """Set start method to preload and configure forkserver preload."""
    # Use file_system sharing to avoid passing 256+ FDs to forkserver
    torch.multiprocessing.set_sharing_strategy("file_system")
    multiprocessing.set_start_method("forkserver", force=True)
    multiprocessing.set_forkserver_preload(
        [
            "pickle",
            "fiona",
            "gcsfs",
            "jsonargparse",
            "numpy",
            "PIL",
            "torch",
            "torch.multiprocessing",
            "torchvision",
            "upath",
            "wandb",
            "rslearn.main",
            "rslearn.train.dataset",
            "rslearn.train.data_module",
        ]
    )
