"""Normalization transforms."""

import json
from typing import Any

from rslearn.train.transforms.transform import Transform


class HeliosNormalize(Transform):
    """Normalize using Helios JSON config."""

    def __init__(
        self,
        config_fname: str,
        band_names: dict[str, list[str]],
        std_multiplier: float | None = 2,
    ) -> None:
        """Initialize a new HeliosNormalize."""
        super().__init__()
        with open(config_fname) as f:
            self.norm_config = json.load(f)
        self.band_names = band_names
        self.std_multiplier = std_multiplier

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply normalization over the inputs and targets.

        Args:
            input_dict: the input
            target_dict: the target

        Returns:
            normalized (input_dicts, target_dicts) tuple
        """
        for modality_name, cur_band_names in self.band_names.items():
            band_norms = self.norm_config[modality_name]
            image = input_dict[modality_name]
            # Keep a set of indices to make sure that we normalize all of them.
            needed_band_indices = set(range(image.shape[0]))

            for band, norm_dict in band_norms.items():
                band_idx = cur_band_names.index(band)
                min_val = norm_dict["mean"] - self.std_multiplier * norm_dict["std"]
                max_val = norm_dict["mean"] + self.std_multiplier * norm_dict["std"]
                image[band_idx] = (image[band_idx] - min_val) / (max_val - min_val)
                needed_band_indices.remove(band_idx)

            if len(needed_band_indices) > 0:
                raise ValueError(
                    f"for modality {modality_name}, bands {needed_band_indices} were unexpectedly not normalized"
                )

        return input_dict, target_dict
