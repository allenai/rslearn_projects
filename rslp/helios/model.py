"""Helios model wrapper for fine-tuning in rslearn."""

import json
import os
import re
from contextlib import nullcontext
from typing import Any

import torch
from einops import rearrange
from helios.data.constants import Modality
from helios.nn.flexihelios import TokensAndMasks
from helios.train.masking import MaskedHeliosSample, MaskValue
from olmo_core.config import Config
from torch.distributed.checkpoint import DefaultLoadPlanner

from rslp.helios.checkpoint import load_model_and_optim_state
from rslp.log_utils import get_logger

logger = get_logger(__name__)

MODALITY_NAMES = [
    "sentinel2_l2a",
    "sentinel1",
    "worldcover",
    "openstreetmap_raster",
    "landsat",
]

AUTOCAST_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries in place, override a keys with b keys.

    Args:
        a: first dictionary
        b: second dictionary
    Returns:
        merged dictionary
    """
    out = dict(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Helios(torch.nn.Module):
    """A wrapper to support the Helios model."""

    def __init__(
        self,
        checkpoint_path: str,
        selector: list[str | int] = [],
        forward_kwargs: dict[str, Any] = {},
        random_initialization: bool = False,
        embedding_size: int | None = None,
        patch_size: int | None = None,
        autocast_dtype: str | None = "bfloat16",
    ):
        """Create a new Helios model.

        Args:
            checkpoint_path: the checkpoint directory to load. It should contain
                config.json file as well as model_and_optim folder.
            selector: an optional sequence of attribute names or list indices to select
                the sub-module that should be applied on the input images.
            forward_kwargs: additional arguments to pass to forward pass besides the
                MaskedHeliosSample.
            random_initialization: whether to skip loading the checkpoint so the
                weights are randomly initialized. In this case, the checkpoint is only
                used to define the model architecture.
            embedding_size: optional embedding size to report via
                get_backbone_channels.
            patch_size: optional patch size to report via get_backbone_channels.
            autocast_dtype: which dtype to use for autocasting, or set None to disable.
        """
        super().__init__()
        self.forward_kwargs = forward_kwargs
        self.embedding_size = embedding_size
        self.patch_size = patch_size

        if autocast_dtype is not None:
            self.autocast_dtype = AUTOCAST_DTYPE_MAP[autocast_dtype]
        else:
            self.autocast_dtype = None

        self.load_model(checkpoint_path, selector, random_initialization)

    def load_model(
        self,
        checkpoint_path: str,
        selector: list[str | int] = [],
        random_initialization: bool = False,
        model_overrides: dict[str, Any] = {},
        planner: DefaultLoadPlanner | None = None,
    ) -> None:
        """Load the model from a checkpoint.

        Args:
            checkpoint_path: the checkpoint directory to load. It should contain
                config.json file as well as model_and_optim folder.
            selector: an optional sequence of attribute names or list indices to select
                the sub-module that should be applied on the input images.
            random_initialization: whether to skip loading the checkpoint so the
                weights are randomly initialized. In this case, the checkpoint is only
                used to define the model architecture.
            model_overrides: overrides for the model building.
            planner: the planner to use for loading the checkpoint.
        """
        # Load the model config and initialize it.
        # We avoid loading the train module here because it depends on running within
        # olmo_core.
        with open(f"{checkpoint_path}/config.json") as f:
            config_dict = json.load(f)
            if model_overrides is not None:
                config_dict["model"] = deep_merge(config_dict["model"], model_overrides)
            model_config = Config.from_dict(config_dict["model"])

        model = model_config.build()

        # Load the checkpoint.
        if not random_initialization:
            train_module_dir = os.path.join(checkpoint_path, "model_and_optim")
            if os.path.exists(train_module_dir):
                load_model_and_optim_state(train_module_dir, model, planner=planner)
                logger.info(f"loaded helios encoder from {train_module_dir}")
            else:
                logger.info(f"could not find helios encoder at {train_module_dir}")
        else:
            logger.info("skipping loading helios encoder")

        # Select just the portion of the model that we actually want to use.
        for part in selector:
            if isinstance(part, str):
                model = getattr(model, part)
            else:
                model = model[part]
        self.model = model

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute feature maps from the Helios backbone.

        Inputs:
            inputs: input dicts. It should include keys corresponding to the modalities
                that should be passed to the Helios model.
        """
        kwargs = {}
        present_modalities = []
        device = None
        # Handle the case where some modalities are multitemporal and some are not.
        # We assume all multitemporal modalities have the same number of timesteps.
        max_timesteps = 1
        for modality in MODALITY_NAMES:
            if modality not in inputs[0]:
                continue
            present_modalities.append(modality)
            cur = torch.stack([inp[modality] for inp in inputs], dim=0)
            device = cur.device
            # Check if it's single or multitemporal, and reshape accordingly
            num_bands = Modality.get(modality).num_bands
            num_timesteps = cur.shape[1] // num_bands
            max_timesteps = max(max_timesteps, num_timesteps)
            cur = rearrange(cur, "b (t c) h w -> b h w t c", t=num_timesteps)
            kwargs[modality] = cur
            # Create mask array which is BHWTS (without channels but with band sets).
            num_band_sets = len(Modality.get(modality).band_sets)
            mask_shape = cur.shape[0:4] + (num_band_sets,)
            mask = (
                torch.ones(mask_shape, dtype=torch.int32, device=device)
                * MaskValue.ONLINE_ENCODER.value
            )
            kwargs[f"{modality}_mask"] = mask

        # Timestamps is required.
        # Note that only months (0 to 11) are used in Helios position encoding.
        # For now, we assign same timestamps to all inputs, but later we should handle varying timestamps per input.
        timestamps = torch.zeros(
            (len(inputs), max_timesteps, 3), dtype=torch.int32, device=device
        )
        timestamps[:, :, 0] = 1  # day
        timestamps[:, :, 1] = torch.arange(max_timesteps, device=device)[
            None, :
        ]  # month
        timestamps[:, :, 2] = 2024  # year
        kwargs["timestamps"] = timestamps

        sample = MaskedHeliosSample(**kwargs)

        # Decide context based on self.autocast_dtype.
        if self.autocast_dtype is None:
            context = nullcontext()
        else:
            assert device is not None
            context = torch.amp.autocast(
                device_type=device.type, dtype=self.autocast_dtype
            )

        with context:
            # Currently we assume the provided model always returns a TokensAndMasks object.
            tokens_and_masks: TokensAndMasks = self.model(
                sample, always_pass_none_mask_to_transformer=True, **self.forward_kwargs
            )["tokens_and_masks"]

        # Apply temporal/modality pooling so we just have one feature per patch.
        features = []
        for modality in present_modalities:
            modality_features = getattr(tokens_and_masks, modality)
            # Pool over band sets and timesteps (BHWTSC -> BHWC).
            pooled = modality_features.mean(dim=[3, 4])
            # We want BHWC -> BCHW.
            pooled = rearrange(pooled, "b h w c -> b c h w")
            features.append(pooled)
        # Pool over the modalities, so we get one BCHW feature map.
        pooled = torch.stack(features, dim=0).mean(dim=0)
        return [pooled]

    def get_backbone_channels(self) -> list:
        """Returns the output channels of this model when used as a backbone.

        The output channels is a list of (downsample_factor, depth) that corresponds
        to the feature maps that the backbone returns. For example, an element [2, 32]
        indicates that the corresponding feature map is 1/2 the input resolution and
        has 32 channels.

        Returns:
            the output channels of the backbone as a list of (downsample_factor, depth)
            tuples.
        """
        return [(self.patch_size, self.embedding_size)]


class TaskConditionedHelios(Helios):
    """A wrapper to support task-conditioned Helios models via embeddings for each task."""

    def __init__(
        self,
        checkpoint_path: str,
        selector: list[str | int] = [],
        forward_kwargs: dict[str, Any] = {},
        random_initialization: bool = False,
        embedding_size: int | None = None,
        patch_size: int | None = None,
        autocast_dtype: str | None = "bfloat16",
        task_embed_opts: dict[str, Any] | None = None,
        model_overrides: dict[str, Any] | None = None,
    ) -> None:
        """Initialize task-conditioned helios model, loading task embeddings.

        Args:
            checkpoint_path: the checkpoint directory to load. It should contain
                config.json file as well as model_and_optim folder.
            selector: optional attribute names or list indices to select a submodule.
            forward_kwargs: extra args for forward() besides MaskedHeliosSample.
            random_initialization: if True, skip loading checkpoint weights.
            embedding_size: optional embedding size reported via get_backbone_channels.
            patch_size: optional patch size reported via get_backbone_channels.
            autocast_dtype: dtype to use for autocasting (string), or None to disable.
            task_embed_opts: configuration for task embeddings
                - type: "learned" (embedding table) | "precomputed" (from file)
                - path: str (required for "precomputed"; optional as init for "learned")
                - dim: int, dimension of task embeddings (overridden by precomputed embeds)
                - tasks: list[str], sorted list of tasks (overridden by precomputed embeds)
            model_overrides: overrides for the model module. Use this to set task conditioning
                options like encoder_config.use_task_lora, etc.
                task_dim will automatically be passed through.
        """
        # Need to manually initialize the superclass to avoid re-initializing the model
        torch.nn.Module.__init__(self)

        self.forward_kwargs = forward_kwargs
        self.embedding_size = embedding_size
        self.patch_size = patch_size

        if autocast_dtype is not None:
            self.autocast_dtype = AUTOCAST_DTYPE_MAP[autocast_dtype]
        else:
            self.autocast_dtype = None

        # Load the task embeddings (we need the task dimension to initialize helios)
        self.load_task_embeds(task_embed_opts or {})

        if model_overrides is None:
            model_overrides = {}

        if "encoder_config" not in model_overrides:
            logger.warning(
                "No model overrides provided, behavior is the same as Helios"
            )
            model_overrides["encoder_config"] = {}

        # Add task_dim to all task-conditioned kwargs, checking that it matches with
        # any preconfigured task_dim values
        for k, v in model_overrides["encoder_config"].items():
            if re.match(r"task.*kwargs", k):
                if "task_dim" not in v:
                    v["task_dim"] = self.task_embed_dim
                elif v["task_dim"] != self.task_embed_dim:
                    raise ValueError(
                        f"task_dim in model_overrides must match task_embed_dim, "
                        f"got {v['task_dim']} != {self.task_embed_dim}"
                    )

        # Load the model from the checkpoint
        self.load_model(
            checkpoint_path,
            selector,
            random_initialization,
            model_overrides,
            planner=DefaultLoadPlanner(allow_partial_load=True),
        )

    def load_task_embeds(self, task_embed_opts: dict[str, Any]) -> None:
        """Load the task embeddings.

        Args:
            task_embed_opts: configuration for task embeddings
                - type: "learned" (embedding table) | "precomputed" (from file)
                - path: str (required for "precomputed"; optional as init for "learned")
                - dim: int, dimension of task embeddings (overridden by precomputed embeds)
                - tasks: list[str], sorted list of tasks (overridden by precomputed embeds)
        """
        self.task_embed_type: str = str(task_embed_opts["type"])
        self.task_embed_path: str | None = task_embed_opts.get("path", None)
        if self.task_embed_type not in ("learned", "precomputed"):
            raise ValueError(
                "task_embed_opts['type'] must be 'learned' or 'precomputed'"
            )

        self.pretrained_task_embeds: torch.Tensor | None = None
        if self.task_embed_type == "precomputed":
            if self.task_embed_path is None:
                raise ValueError(
                    "task_embed_opts['path'] must be provided for precomputed embeddings"
                )
            self.load_pretrained_embeds()

        elif self.task_embed_type == "learned" and self.task_embed_path is not None:
            # Use provided embeddings just to initialize
            self.load_pretrained_embeds()
            logger.info(
                "Using pretrained task embeds to initialize the embedding table"
            )

        # Some attributes are set by precomputed embeds if they're specified
        if not hasattr(self, "task_embed_dim"):
            self.task_embed_dim: int = int(task_embed_opts["dim"])
        if not hasattr(self, "tasks"):
            self.tasks: list[str] = list(task_embed_opts["tasks"])

        # Build the embedding table
        self.task_embed_table = torch.nn.Embedding(
            num_embeddings=len(self.tasks),
            embedding_dim=self.task_embed_dim,
        )

        # If we have pretrained vectors (either for precomputed or init), copy them in
        if self.pretrained_task_embeds is not None:
            with torch.no_grad():
                self.task_embed_table.weight.copy_(
                    self.pretrained_task_embeds.to(self.task_embed_table.weight.device)
                )

        # If precomputed, freeze the table to avoid accidental training
        if self.task_embed_type == "precomputed":
            for p in self.task_embed_table.parameters():
                p.requires_grad = False

    def load_pretrained_embeds(self, drop: tuple[str] = ("code_hash",)) -> None:
        """Load task embeddings from a file expected to contain a dict.

        Dict format is { task_name (str): 1D torch.Tensor }.
        Resulting pretrained embeds are [num_tasks, dim].

        Args:
            drop: list of keys to drop from the dict.
        """
        obj = torch.load(self.task_embed_path, map_location="cpu")
        obj = {k: v for k, v in obj.items() if k not in drop}

        self.tasks = list(obj.keys())
        logger.info(f"Loaded task embeddings: {self.tasks}")

        dim = obj[self.tasks[0]].shape[0]
        logger.info(f"Using task embed dim {dim}")

        rows = []
        for t in self.tasks:
            v = obj[t].detach().flatten().to(torch.float32)
            if v.dim() != 1 or v.numel() != dim:
                raise ValueError(
                    f"Embedding for task '{t}' must be 1D of length {dim}, "
                    f"got shape {tuple(v.shape)}"
                )
            rows.append(v)

        self.pretrained_task_embeds = torch.stack(rows, dim=0)
        self.task_embed_dim = dim

    def compute_task_embeds(self, task_id: torch.Tensor) -> torch.Tensor:
        """Compute or retrieve task embeddings given the task identifiers.

        Args:
            task_id: Integer tensor of shape (B,) with ids in [0, num_tasks).

        Returns:
            Tensor of shape (B, dim) with the task embeddings.
        """
        idx = task_id.long().to(self.task_embed_table.weight.device)
        return self.task_embed_table(idx)

    def forward(self, inputs: list[dict[str, Any]]) -> list[torch.Tensor]:
        """Compute feature maps from the Helios backbone with task conditioning.

        Inputs:
            inputs: list of input dicts. It should include keys corresponding to the
                    modalities that should be passed to the Helios model.
        """
        # Assume dataset_source is consistent across the batch
        dataset_source = inputs[0]["dataset_source"]
        ids = torch.tensor([self.tasks.index(dataset_source)] * len(inputs))
        task_embeds = self.compute_task_embeds(ids)

        # Feed in task embedding and restore previous value if it exists
        key = "task_emb"
        self.forward_kwargs[key], prev = (task_embeds, self.forward_kwargs.get(key))
        out = super().forward(inputs)
        self.forward_kwargs[key] = prev
        return out
