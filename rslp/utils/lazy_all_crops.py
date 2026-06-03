"""Lazy (non-caching) all-crops dataset + data module for memory-safe prediction.

rslearn's default ``load_all_crops`` prediction path wraps a ``ModelDataset`` in
``IterableAllCropsDataset``, which reads the *entire* window into memory once and
then slices each crop out of that in-memory array. For a full Sentinel-1 GRD scene
the window is one ~25000x16700 px, 6-channel float32 array (~10 GB raw, ~28 GB after
transform copies), which OOM-kills the worker on the first batch fetch.

``LazyAllCropsDataset`` mirrors ``rslearn.train.in_memory_dataset.InMemoryAllCropsDataset``
(map-style: enumerates every sliding-window crop, exposes ``__len__``/``__getitem__``)
but does NOT cache the window. Each ``__getitem__`` reads only the requested crop's
pixel bounds from disk via rslearn's ``read_data_input``, so peak memory is one crop
(~6 MB at 512x512x6 float32) instead of the whole scene.

This lives in rslearn_projects (rather than modifying rslearn core) and reuses only
importable, public rslearn helpers. It is wired in via the model config's
``data.class_path`` -> ``LazyAllCropsDataModule``.
"""

import random
from typing import Any

import torch

from rslearn.dataset import Window
from rslearn.log_utils import get_logger
from rslearn.train.all_crops_dataset import (
    IterableAllCropsDataset,
    get_window_crop_options,
)
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.dataset import (
    ModelDataset,
    RetryDataset,
    is_data_input_available,
    read_data_input,
)
from rslearn.train.model_context import SampleMetadata

logger = get_logger(__name__)


class LazyAllCropsDataset(torch.utils.data.Dataset):
    """Map-style all-crops dataset that reads each crop from disk on demand.

    Equivalent in output to ``InMemoryAllCropsDataset`` / ``IterableAllCropsDataset``
    but without ever materializing a full window in memory. Because each crop's
    bounds are read directly from the (tiled) source rasters, no in-memory slicing
    or edge padding is needed: ``read_data_input`` returns exactly the crop-sized
    array.
    """

    def __init__(
        self,
        dataset: ModelDataset,
        crop_size: tuple[int, int],
        overlap_pixels: int = 0,
    ):
        """Create a new LazyAllCropsDataset.

        Args:
            dataset: the ModelDataset to wrap.
            crop_size: the size of the crops to extract.
            overlap_pixels: number of pixels shared between adjacent crops.
        """
        super().__init__()
        self.dataset = dataset
        self.crop_size = crop_size
        self.overlap_size = (overlap_pixels, overlap_pixels)
        self.windows = dataset.get_dataset_examples()
        self.inputs = dataset.inputs

        # Enumerate (window_id, crop_bounds, (crop_idx, num_crops)) using only
        # window-bounds metadata. No pixel data is read here. Crops within a
        # window are contiguous and in increasing crop_idx order, which
        # RslearnWriter relies on to reassemble outputs.
        self.crops: list[tuple[int, tuple[int, int, int, int], tuple[int, int]]] = []
        for window_id, window in enumerate(self.windows):
            crop_bounds_list = get_window_crop_options(
                crop_size, self.overlap_size, window.bounds
            )
            for i, crop_bounds in enumerate(crop_bounds_list):
                self.crops.append((window_id, crop_bounds, (i, len(crop_bounds_list))))

    def __len__(self) -> int:
        """Return the total number of crops across all windows."""
        return len(self.crops)

    def __getitem__(
        self, index: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Read a single crop directly from disk and return a finalized sample."""
        window_id, crop_bounds, (crop_idx, num_crops) = self.crops[index]
        window = self.windows[window_id]
        rng = random.Random(window_id if self.dataset.fix_crop_pick else None)

        # Read ONLY the crop bounds for each input. This is the key difference
        # from the default path: read_data_input issues a windowed read of the
        # crop bbox rather than the full window.
        raw_inputs: dict[str, Any] = {}
        passthrough_inputs: dict[str, Any] = {}
        for name, data_input in self.inputs.items():
            if not data_input.required and not is_data_input_available(
                data_input, window
            ):
                continue
            raw_inputs[name] = read_data_input(
                self.dataset.dataset, window, crop_bounds, data_input, rng
            )
            if data_input.passthrough:
                passthrough_inputs[name] = raw_inputs[name]

        metadata = SampleMetadata(
            window_group=window.group,
            window_name=window.name,
            window_bounds=window.bounds,
            crop_bounds=crop_bounds,
            crop_idx=crop_idx,
            num_crops_in_window=num_crops,
            time_range=window.time_range,
            projection=window.projection,
            dataset_source=self.dataset.name,
        )

        input_dict, target_dict = self.dataset.task.process_inputs(
            raw_inputs,
            metadata=metadata,
            load_targets=not self.dataset.split_config.get_skip_targets(),
        )
        input_dict.update(passthrough_inputs)
        input_dict, target_dict = self.dataset.transforms(input_dict, target_dict)
        return input_dict, target_dict, metadata

    def get_dataset_examples(self) -> list[Window]:
        """Returns the list of windows in this dataset."""
        return self.dataset.get_dataset_examples()

    def set_name(self, name: str) -> None:
        """Sets the dataset name.

        Args:
            name: the dataset name.
        """
        self.dataset.set_name(name)


class LazyAllCropsDataModule(RslearnDataModule):
    """Drop-in RslearnDataModule that uses LazyAllCropsDataset for prediction.

    Behaves identically to RslearnDataModule except that, when a split is set up
    as an ``IterableAllCropsDataset`` (the ``load_all_crops`` non-in-memory path),
    it is replaced with a ``LazyAllCropsDataset`` so crops are read on demand
    instead of materializing the full window. All other splits/paths are
    untouched.
    """

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> None:
        """Set up datasets, then swap any IterableAllCropsDataset for the lazy one.

        Uses ``*args``/``**kwargs`` to forward to ``RslearnDataModule.setup`` so we
        stay compatible across rslearn versions (the second positional/keyword arg
        was renamed from ``use_in_memory_all_crops_dataset`` to
        ``use_in_memory_dataset`` between releases). Lightning calls ``setup(stage)``
        positionally, so nothing extra is normally passed.

        Args:
            stage: the lightning stage (fit/validate/test/predict).
            *args: forwarded to RslearnDataModule.setup.
            **kwargs: forwarded to RslearnDataModule.setup.
        """
        super().setup(stage, *args, **kwargs)

        for split, dataset in list(self.datasets.items()):
            retry_wrapper = None
            inner = dataset
            if isinstance(inner, RetryDataset):
                retry_wrapper = inner
                inner = inner.dataset

            if not isinstance(inner, IterableAllCropsDataset):
                continue

            lazy = LazyAllCropsDataset(
                dataset=inner.dataset,
                crop_size=inner.crop_size,
                overlap_pixels=inner.overlap_size[0],
            )
            logger.info(
                "using LazyAllCropsDataset for split '%s' (%d crops, on-demand reads)",
                split,
                len(lazy),
            )

            if retry_wrapper is not None:
                self.datasets[split] = RetryDataset(
                    lazy,
                    retries=retry_wrapper.retries,
                    delay=retry_wrapper.delay,
                )
            else:
                self.datasets[split] = lazy
