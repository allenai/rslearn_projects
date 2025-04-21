"""Helios model wrapper for fine-tuning in rslearn."""

import json
from typing import Any

import torch
from einops import rearrange
from helios.data.constants import Modality
from helios.nn.flexihelios import TokensAndMasks
from helios.train.masking import MaskedHeliosSample, MaskValue
from olmo_core.config import Config
from olmo_core.distributed.checkpoint import load_model_and_optim_state

from rslp.log_utils import get_logger

logger = get_logger(__name__)

MODALITY_NAMES = [
    "sentinel2_l2a",
    "sentinel1",
    "worldcover",
    "openstreetmap_raster",
]


class Helios(torch.nn.Module):
    """A wrapper to support the Helios model."""

    def __init__(
        self,
        checkpoint_path: str,
        selector: list[str | int] = [],
        forward_kwargs: dict[str, Any] = {},
        random_initialization: bool = False,
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
        """
        super().__init__()
        self.forward_kwargs = forward_kwargs

        # Load the model config and initialize it.
        # We avoid loading the train module here because it depends on running within
        # olmo_core.
        with open(f"{checkpoint_path}/config.json") as f:
            config_dict = json.load(f)
            model_config = Config.from_dict(config_dict["model"])

        model = model_config.build()

        # Load the checkpoint.
        if not random_initialization:
            train_module_dir = f"{checkpoint_path}/model_and_optim"
            load_model_and_optim_state(train_module_dir, model)

        # Select just the portion of the model that we actually want to use.
        for part in selector:
            if isinstance(part, str):
                model = getattr(model, part)
            else:
                model = model[part]
        self.model = model

    def get_backbone_channels(self) -> list[list[int]]:
        """Returns the output channels of the encoder at different scales.

        Returns:
            List[List[int]]: List of downsample factors and corresponding channel counts at each scale.
        """
        # Mainly for rslearn.models.simple_time_series.SimpleTimeSeries
        return [[1, self.model.embedding_size]]

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
        timestamps = torch.zeros((len(inputs), max_timesteps, 3), dtype=torch.int32, device=device)
        timestamps[:, :, 0] = 1  # day
        timestamps[:, :, 1] = torch.arange(max_timesteps, device=device)[None, :]  # month
        timestamps[:, :, 2] = 2024  # year
        kwargs["timestamps"] = timestamps

        sample = MaskedHeliosSample(**kwargs)

        # Currently we assume the provided model always returns a TokensAndMasks
        # object.
        tokens_and_masks: TokensAndMasks = self.model(sample, **self.forward_kwargs)

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
