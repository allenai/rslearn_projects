"""Helios model wrapper for fine-tuning in rslearn."""

from typing import Any

import torch
from einops import rearrange
from helios.data.constants import Modality
from helios.nn.flexihelios import TokensAndMasks
from helios.nn.latent_mim import LatentMIMConfig
from helios.train.masking import MaskedHeliosSample, MaskValue

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
        # I tried to use generic Config here but jsonargparse struggled to
        # initialize it with a subclass. So currently it has to be a concrete type
        # instead of generic superclass.
        model_config: LatentMIMConfig,
        selector: list[str | int] = [],
        forward_kwargs: dict[str, Any] = {},
    ):
        """Create a new Helios model.

        Args:
            model_config: the configuration object that specifies the model.
            selector: an optional sequence of attribute names or list indices to select
                the sub-module that should be applied on the input images.
            forward_kwargs: additional arguments to pass to forward pass besides the
                MaskedHeliosSample.
        """
        super().__init__()
        self.forward_kwargs = forward_kwargs

        model = model_config.build()
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
        for modality in MODALITY_NAMES:
            if modality not in inputs[0]:
                continue
            present_modalities.append(modality)
            cur = torch.stack([inp[modality] for inp in inputs], dim=0)
            device = cur.device
            # Reshape BCHW to BHWTC, currently we assume one timestep.
            cur = rearrange(cur, "b c h w -> b h w 1 c")
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
        # For now we assume one timestep and assign it an arbitrary value.
        timestamps = torch.zeros((len(inputs), 1, 3), dtype=torch.int32, device=device)
        timestamps[:, :, 0] = 1  # day
        timestamps[:, :, 1] = 7  # month
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
