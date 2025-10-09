"""Load model and optimizer state in-place from a checkpoint saved via `save_model_and_optim_state()`.

Copied from https://olmo-core.readthedocs.io/en/stable/_modules/olmo_core/distributed/checkpoint.html#load_model_and_optim_state,
except that we allow for missing weights.

Goal is eventually to get something like this merged into olmo_core and not have to maintain this.
"""

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from olmo_core.aliases import PathOrStr
from olmo_core.distributed.checkpoint import (
    RemoteFileSystemReader,
    _prepare_state_dict,
    swap_param_keys,
)
from olmo_core.io import normalize_path
from olmo_core.utils import gc_cuda


@torch.no_grad()
def load_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: torch.optim.Optimizer | None = None,
    *,
    process_group: dist.ProcessGroup | None = None,
    key_mapping: dict[str, str] | None = None,
    pre_download: bool = False,
    work_dir: PathOrStr | None = None,
    strict: bool = True,
    flatten_optimizer_state: bool = False,
    thread_count: int | None = None,
    planner: dist_cp.DefaultLoadPlanner | None = None,
) -> None:
    """Load model and optimizer state in-place from a checkpoint saved via :func:`save_model_and_optim_state()`.

    This method is agnostic to the distributed topology in that it can load checkpoints saved with a different
    distributed topology (e.g. FSDP/FSDP2, DDP).

    .. seealso::
        - :func:`save_model_and_optim_state()`
        - :func:`unshard_checkpoint()`

    .. tip::
        With :class:`~torch.distributed.fsdp.FullyShardedDataParallel` models it's not necessary
        to set the state dict type before calling this (or :func:`save_model_and_optim_state()`) via
        :meth:`~torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type()` or other methods.
        This function handles that internally.

    .. warning::
        Due to the way :mod:`torch.distributed.checkpoint` works, if you have keys in the checkpoint
        dict that are not present in the current state of the model or optimizer, those keys won't
        be loaded.

        For example, if you added a custom field to one of your optimizer's param groups
        before saving the checkpoint, but don't have that field in the param group of the optimizer
        you're loading into, it won't be added.

        This can cause unexpected behavior if you're not careful. In this case the best thing to do
        is to ensure all keys are in present param groups when you initialize the optimizer, before saving
        or loading a checkpoint.

    :param dir: Path/URL to the checkpoint saved via :func:`save_model_and_optim_state()`.
    :param model: The model to load the state into.
    :param optim: The optimizer to load the state into.
    :param process_group: The process group to use for distributed collectives.
    :param key_mapping: Can be used to load a checkpoint where certain parameter have different names.
        This dictionary should map current keys to keys in the checkpoint to be loaded.
    :param pre_download: Download and cache relevant remote checkpoint files before trying to read from them.
    :param work_dir: A working directory for caching files/directories.
    :param strict: Load keys strictly.
    :param flatten_optimizer_state: Flatten the optimizer state when loading. This should match
        the setting used when saving the state dict and is needed in a distributed setting when
        the params in some param groups may differ between ranks, such as with pipeline parallelism.
    :param thread_count: Set the number of threads used for certain operations.
    :param planner: The planner to use for loading the checkpoint.
    """
    dir = normalize_path(dir)
    state_dict = _prepare_state_dict(
        model,
        optim,
        process_group=process_group,
        flatten_optimizer_state=flatten_optimizer_state,
    )
    reader = RemoteFileSystemReader(
        dir, thread_count=thread_count, pre_download=pre_download, work_dir=work_dir
    )
    metadata = reader.read_metadata()

    if key_mapping is not None:
        swap_param_keys(state_dict, key_mapping, metadata=metadata)

    dist_cp.load(  # nosec
        state_dict,
        checkpoint_id=dir,
        storage_reader=reader,
        process_group=process_group,
        planner=planner,
    )

    if key_mapping is not None:
        swap_param_keys(state_dict, key_mapping, reverse=True, quiet=True)

    dist_cp_sd.set_model_state_dict(
        model, state_dict["model"], options=dist_cp_sd.StateDictOptions(strict=strict)
    )
    gc_cuda()

    if optim is not None:
        dist_cp_sd.set_optimizer_state_dict(
            model,
            optim,
            state_dict["optim"],
            options=dist_cp_sd.StateDictOptions(
                strict=strict, flatten_optimizer_state_dict=flatten_optimizer_state
            ),
        )
        gc_cuda()
