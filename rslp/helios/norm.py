"""Normalization transforms."""

from typing import Any

from helios.data.normalize import load_computed_config
from helios.data.utils import convert_to_db
from rslearn.train.transforms.transform import Transform


class HeliosNormalize(Transform):
    """Normalize using Helios JSON config."""

    def __init__(
        self,
        band_names: dict[str, list[str]],
        std_multiplier: float | None = 2,
        config_fname: str | None = None,
    ) -> None:
        """Initialize a new HeliosNormalize.

        Args:
            band_names: map from modality name to the list of bands in that modality in
                the order they are being loaded. Note that this order must match the
                expected order for the Helios model.
            std_multiplier: the std multiplier matching the one used for the model
                training in Helios.
            config_fname: override the normalization configuration filename. By default
                we use "norm_configs/computed.json" in helios.resources.
        """
        super().__init__()
        self.norm_config = load_computed_config()
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
            if modality_name == "sentinel1":
                image = convert_to_db(image)
            # Keep a set of indices to make sure that we normalize all of them.
            needed_band_indices = set(range(image.shape[0]))
            num_timesteps = image.shape[0] // len(cur_band_names)

            for band, norm_dict in band_norms.items():
                # If multitemporal, normalize each timestep separately.
                for t in range(num_timesteps):
                    band_idx = cur_band_names.index(band) + t * len(cur_band_names)
                    min_val = norm_dict["mean"] - self.std_multiplier * norm_dict["std"]
                    max_val = norm_dict["mean"] + self.std_multiplier * norm_dict["std"]
                    image[band_idx] = (image[band_idx] - min_val) / (max_val - min_val)
                    needed_band_indices.remove(band_idx)

            if len(needed_band_indices) > 0:
                raise ValueError(
                    f"for modality {modality_name}, bands {needed_band_indices} were unexpectedly not normalized"
                )

        return input_dict, target_dict
