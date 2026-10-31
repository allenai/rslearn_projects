"""Transform to test round 3 runs on the round 2 test_history datasets.

In the round 2 test_history datasets, the quarterly layer (infreq_test) overlaps the
weekly layer (freq_test_{E}), and mosaic timestamps are the time of the first scene in
the mosaic, so the latest quarterly mosaic can have the same scene and timestamp as the
first weekly mosaic. The round 2 sampler only used quarterly mosaics ending strictly
before the first weekly mosaic, but ChangeTimeSeriesSampler also uses the ones ending at
the same time (since its layers are adjacent rather than overlapping), which would add
the scene twice (and OlmoEarth rejects images with the same timestamp). This transform
drops those quarterly mosaics before the sampler, so the time series matches round 2.
"""

from typing import Any

from rslearn.train.model_context import RasterImage
from rslearn.train.transforms.transform import Transform


class DropOverlappingInfrequent(Transform):
    """Drop the infrequent images that do not end before the first frequent image."""

    def __init__(self, frequent_key: str, infrequent_key: str) -> None:
        """Create a new DropOverlappingInfrequent.

        Args:
            frequent_key: the input key of the frequent input.
            infrequent_key: the input key of the infrequent input.
        """
        super().__init__()
        self.frequent_key = frequent_key
        self.infrequent_key = infrequent_key

    def forward(
        self, input_dict: dict[str, Any], target_dict: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Drop the infrequent images.

        Args:
            input_dict: the input dict, containing the frequent and infrequent inputs.
            target_dict: the target dict.

        Returns:
            the updated (input_dict, target_dict).
        """
        frequent = input_dict.get(self.frequent_key)
        infrequent = input_dict.get(self.infrequent_key)
        if frequent is None or not frequent.timestamps or infrequent is None:
            return input_dict, target_dict
        first_start = frequent.timestamps[0][0]
        keep = [
            idx
            for idx, (_, end) in enumerate(infrequent.timestamps)
            if end < first_start
        ]
        input_dict[self.infrequent_key] = RasterImage(
            infrequent.image[:, keep],
            timestamps=[infrequent.timestamps[idx] for idx in keep],
        )
        return input_dict, target_dict
