"""Transforms for the Landsat vessel models."""

from typing import Any

from rslearn.train.transforms.crop import Crop


class CenterCrop(Crop):
    """Crop the centre of each input, whatever size the input is.

    The round-1 annotation windows are 512 px at 15 m so an annotator can see 7.68 km of
    context, while the classifier trains on the 64 px window around the detection. The
    older groups (``selected_copy``, ``feedback_20260325``, ...) are already 64 px. A
    training run that mixes them therefore sees two window sizes, and
    :class:`rslearn.train.transforms.crop.Crop` cannot serve both: with ``offset`` set it
    is a fixed pixel offset that overruns a 64 px window, and without ``offset`` it is a
    random crop, which on a 512 px window usually misses the vessel entirely.

    This crops each input around its own centre and clamps to the input size, so a 512 px
    window yields its central 64 px and a 64 px window passes through untouched. The
    detection sits at the window centre by construction (see
    ``scripts/create_round1_windows.py``), so the centre crop is the detection crop.

    Changing the trained window size is then one number here, with no re-acquisition:
    ``crop_size: 128`` reads the central 128 px of the same windows.

    One constraint when mixing groups: because inputs smaller than ``crop_size`` pass
    through at their own size, a ``crop_size`` above the smallest group's window size
    yields samples of two different sizes in one batch. rslearn's own collation tolerates
    that — ``collate_fn`` builds lists of per-sample dicts and never stacks — but
    ``OlmoEarth`` takes height and width from the first sample of the batch and assigns
    every sample into a buffer of that shape, so a mismatch raises there instead
    (``RuntimeError: The expanded size of the tensor (512) must match the existing size
    (64)``). Training on ``round1_20260803`` alongside the 64 px groups therefore means
    ``crop_size: 64``; to train at 128 or 256, restrict ``groups`` to the 512 px round-1
    windows.
    """

    def __init__(
        self,
        crop_size: int,
        image_selectors: list[str] = ["image"],
        box_selectors: list[str] = [],
        skip_missing: bool = False,
    ) -> None:
        """Initialize a new CenterCrop.

        Args:
            crop_size: the size to crop to. Inputs smaller than this are left alone.
            image_selectors: image items to transform.
            box_selectors: boxes items to transform.
            skip_missing: if True, skip selectors that are not present.
        """
        super().__init__(
            crop_size=crop_size,
            image_selectors=image_selectors,
            box_selectors=box_selectors,
            skip_missing=skip_missing,
        )
        self.target_size = crop_size

    def sample_state(self, image_shape: tuple[int, int]) -> dict[str, Any]:
        """Centre the crop in the input rather than sampling a position.

        Args:
            image_shape: the (height, width) of the images to transform, at the lowest
                resolution present.

        Returns:
            dict of the (deterministic) choices, in the form the base class applies.
        """
        size = min(self.target_size, image_shape[0], image_shape[1])
        return {
            "image_shape": image_shape,
            "crop_size": size,
            "remove_from_left": (image_shape[1] - size) // 2,
            "remove_from_top": (image_shape[0] - size) // 2,
        }
