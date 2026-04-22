"""Load rslearn-compatible data from OlmoEarth dataset folder."""

import hashlib
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import rasterio
import torch
from einops import rearrange
from rasterio.crs import CRS
from rslearn.train.data_module import collate_fn
from rslearn.train.model_context import RasterImage, SampleMetadata
from rslearn.train.tasks import Task
from rslearn.utils.geometry import Projection
from torch.utils.data import DataLoader, DistributedSampler

from rslp.log_utils import get_logger

logger = get_logger(__file__)


@dataclass
class ModalityInfo:
    """Info about a modality."""

    # Filename suffixes and the number of bands in that suffix.
    # If there are multiple suffixes, the bands should be stacked.
    suffixes: list[tuple[str, int]]

    # Number of overall bands.
    # If num_bands = 1 but there are multiple bands across the suffixes, then it will
    # be converted from one-hot encoding to integer class label.
    # This corresponds to the number of classes for categorical modalities.
    num_bands: int

    # Normalization applied as: (value - norm_offset) / norm_factor.
    # norm_factor != None signals that the modality is float-valued.
    norm_factor: float | None = None
    norm_offset: float = 0.0

    is_multitemporal: bool = False

    # Spatial resolution divisor relative to TILE_SIZE.
    # 1 = full resolution (256x256), 4 = quarter resolution (64x64), etc.
    resolution_div: int = 1


MODALITIES = {
    "10_sentinel2_l2a_monthly": ModalityInfo(
        suffixes=[
            ("_10.tif", 4),
            ("_20.tif", 6),
            ("_40.tif", 2),
        ],
        num_bands=12,
        norm_factor=10000,
        is_multitemporal=True,
    ),
    "10_openstreetmap_raster": ModalityInfo(
        suffixes=[("_2.5.tif", 30)],
        num_bands=1,
    ),
    "10_cdl": ModalityInfo(
        suffixes=[("_10.tif", 1)],
        num_bands=1,
    ),
    "10_worldcover": ModalityInfo(
        suffixes=[("_10.tif", 1)],
        num_bands=1,
    ),
    "10_wri_canopy_height_map": ModalityInfo(
        suffixes=[("_10.tif", 1)],
        num_bands=1,
    ),
    "10_srtm": ModalityInfo(
        suffixes=[("_10.tif", 1)],
        num_bands=1,
    ),
    "10_worldcereal": ModalityInfo(
        suffixes=[("_10.tif", 8)],
        num_bands=1,
    ),
    "10_olmoearth_v1_base_embedding": ModalityInfo(
        suffixes=[("_40.tif", 768)],
        num_bands=768,
        norm_factor=128.0,
        norm_offset=128.0,
        resolution_div=4,
    ),
}
TILE_SIZE = 256


class CollateFunction:
    """Collate function for OlmoEarth dataset."""

    def __init__(
        self,
        randomize: bool = True,
        min_size: int = 256,
        max_size: int = 256,
        patch_size: int = 32,
    ):
        """Create a new CollateFunction.

        Args:
            randomize: whether to randomize the selection of options like number of
                timesteps or height/width for cropping. Should be true for training and
                false for validation.
            min_size: minimum size to crop the input.
            max_size: maximum size to crop the input.
            patch_size: ensure the cropped input is a multiple of this amount.
        """
        self.randomize = randomize
        self.min_size = min_size
        self.max_size = max_size
        self.patch_size = patch_size

    def __call__(
        self, batch: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]
    ) -> tuple:
        """Collate batch of training examples.

        We just make list of the inputs and another of the targets.

        Args:
            batch: list of input/target/metadata for each example

        Returns:
            a tuple (inputs, targets, metadatas)
        """
        inputs, targets, metadatas = collate_fn(batch)

        # Find minimum number of available timesteps.
        # Input modalities are now RasterImage with shape (C, T, H, W).
        multitemporal_modalities = [
            (modality, info)
            for modality, info in MODALITIES.items()
            if info.is_multitemporal
        ]
        minimum_available_timesteps: int | None = None
        for input_dict in inputs:
            for modality, info in multitemporal_modalities:
                if modality not in input_dict:
                    continue
                cur_timesteps = input_dict[modality].image.shape[1]
                if minimum_available_timesteps is None:
                    minimum_available_timesteps = cur_timesteps
                else:
                    minimum_available_timesteps = min(
                        minimum_available_timesteps, cur_timesteps
                    )

        # Randomly pick a subset of timesteps, along with spatial crop size.
        assert minimum_available_timesteps is not None
        if self.randomize:
            num_timesteps = random.randint(1, minimum_available_timesteps)
            crop_size = random.randint(self.min_size, self.max_size)
            crop_size = (crop_size // self.patch_size) * self.patch_size
            crop_h_start = random.randint(0, TILE_SIZE - crop_size)
            crop_w_start = random.randint(0, TILE_SIZE - crop_size)
        else:
            rng = np.random.default_rng(hash(metadatas[0].window_name) % 65536)
            num_timesteps = rng.integers(minimum_available_timesteps) + 1
            crop_size = self.min_size + rng.integers(self.max_size - self.min_size + 1)
            crop_size = (crop_size // self.patch_size) * self.patch_size
            crop_h_start = 0
            crop_w_start = 0

        # Temporal subset.
        for input_dict in inputs:
            for modality, info in multitemporal_modalities:
                if modality not in input_dict:
                    continue
                img = input_dict[modality].image  # (C, T, H, W)
                available_timesteps = list(range(img.shape[1]))
                if self.randomize:
                    selected = sorted(random.sample(available_timesteps, num_timesteps))
                else:
                    selected = available_timesteps[:num_timesteps]
                input_dict[modality] = RasterImage(image=img[:, selected, :, :])

        # Spatial crop. Scale coordinates for tensors at different resolutions.
        def _scaled_crop(
            tensor: torch.Tensor,
        ) -> tuple[int, int, int]:
            scale = tensor.shape[-2] / TILE_SIZE
            return (
                int(crop_h_start * scale),
                int(crop_w_start * scale),
                int(crop_size * scale),
            )

        for input_dict in inputs:
            for modality in list(input_dict.keys()):
                value = input_dict[modality]
                if isinstance(value, RasterImage):
                    sh, sw, sc = _scaled_crop(value.image)
                    input_dict[modality] = RasterImage(
                        image=value.image[:, :, sh : sh + sc, sw : sw + sc],
                        timestamps=value.timestamps,
                    )
                elif isinstance(value, torch.Tensor):
                    sh, sw, sc = _scaled_crop(value)
                    input_dict[modality] = value[:, sh : sh + sc, sw : sw + sc]
                # input_dict also contains task name for each of the tasks mapping to empty dicts.
            # Set "image" alias after cropping so it refers to the cropped version.
            input_dict["image"] = input_dict["10_sentinel2_l2a_monthly"]
        for target_dict in targets:
            for task_name in list(target_dict.keys()):
                for sub_name in list(target_dict[task_name].keys()):
                    image = target_dict[task_name][sub_name]
                    if isinstance(image, RasterImage):
                        sh, sw, sc = _scaled_crop(image.image)
                        image = RasterImage(
                            image=image.image[:, :, sh : sh + sc, sw : sw + sc],
                            timestamps=image.timestamps,
                        )
                    elif len(image.shape) == 2:
                        sh, sw, sc = _scaled_crop(image)
                        image = image[sh : sh + sc, sw : sw + sc]
                    else:
                        sh, sw, sc = _scaled_crop(image)
                        image = image[:, sh : sh + sc, sw : sw + sc]
                    target_dict[task_name][sub_name] = image

        return (inputs, targets, metadatas)


class OlmoEarthDataset(torch.utils.data.Dataset):
    """A dataset for OlmoEarth data."""

    def __init__(
        self,
        ds_path: Path,
        task: Task,
        input_modalities: list[str],
        target_modalities: list[str],
        limit: int | None = None,
        skip: int = 0,
    ):
        """Create a new OlmoEarthDataset.

        Args:
            ds_path: the path to the OlmoEarth dataset folder.
            task: the task to train on.
            input_modalities: list of modalities to input.
            target_modalities: list of modalities to use as targets.
            limit: limit to this many samples
            skip: skip this many initial samples.
        """
        self.ds_path = ds_path
        self.task = task
        self.input_modalities = input_modalities
        self.target_modalities = target_modalities

        # Get the unique tiles that have at least one of the input modalities.
        tile_set = set()
        for modality in self.input_modalities:
            modality_info = MODALITIES[modality]
            suffix, _ = modality_info.suffixes[0]
            modality_dir = ds_path / modality
            logger.info(f"Getting tiles in {modality_dir}")
            for fname in modality_dir.iterdir():
                if not fname.name.endswith(suffix):
                    continue
                tile_name = fname.name[: -len(suffix)]
                tile_set.add(tile_name)

        logger.info(f"Discovered {len(tile_set)} tiles total")
        self.tile_list = list(tile_set)
        self.tile_list.sort(
            key=lambda tile_name: hashlib.sha256(tile_name.encode()).hexdigest()
        )

        self.tile_list = self.tile_list[skip:]
        if limit is not None:
            self.tile_list = self.tile_list[0:limit]

        logger.info(f"Finishing setup with {len(self.tile_list)} tiles")

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.tile_list)

    def _load_modality(self, tile_name: str, modality: str) -> torch.Tensor | None:
        info = MODALITIES[modality]
        target_size = TILE_SIZE // info.resolution_div

        # Get list of images across suffixes.
        # Each image is TxCxHxW for multitemporal, otherwise CxHxW.
        image_list = []

        for suffix, suffix_bands in info.suffixes:
            fname = self.ds_path / modality / (tile_name + suffix)
            if not fname.exists():
                return None

            with rasterio.open(fname) as src:
                array = torch.from_numpy(src.read())

            # Resize to target_size.
            if array.shape[1] < target_size:
                factor = target_size // array.shape[1]
                if (
                    array.shape[1] * factor != target_size
                    or array.shape[2] * factor != target_size
                ):
                    raise ValueError(f"bad array shape {array.shape}")
                array = torch.repeat_interleave(array, repeats=factor, dim=1)
                array = torch.repeat_interleave(array, repeats=factor, dim=2)
            elif array.shape[1] > target_size:
                factor = array.shape[1] // target_size
                if (
                    array.shape[1] != target_size * factor
                    or array.shape[2] != target_size * factor
                ):
                    raise ValueError(f"bad array shape {array.shape}")
                # Use max pool since it works better for categorical modalities (to not
                # lose the finer-grained detail).
                array = torch.nn.functional.max_pool2d(
                    array, kernel_size=factor, stride=factor
                )

            # Convert stacked timesteps.
            if info.is_multitemporal:
                if array.shape[0] % suffix_bands != 0:
                    raise ValueError(
                        f"array has {array.shape[0]} bands but that is not a multiple of {suffix_bands}"
                    )
                num_timesteps = array.shape[0] // suffix_bands
                array = array.reshape(
                    (num_timesteps, suffix_bands, array.shape[1], array.shape[2])
                )

            elif array.shape[0] != suffix_bands:
                raise ValueError(
                    f"non-multi-temporal array {array.shape[0]} != {suffix_bands}"
                )

            image_list.append(array)

        # Stack the bands on channel axis.
        if info.is_multitemporal:
            image = torch.cat(image_list, dim=1)
            # Change timesteps and bands back to being combined.
            image = image.reshape((-1, image.shape[2], image.shape[3]))
        else:
            image = torch.cat(image_list, dim=0)

        # Convert single-band via one-hot encoding if needed.
        # This only works for non-multi-temporal data.
        if image.shape[0] > 1 and info.num_bands == 1:
            image = image.argmax(dim=0, keepdim=True)

        if info.norm_factor is not None:
            image = ((image - info.norm_offset) / info.norm_factor).to(
                dtype=torch.float32
            )
        else:
            image = image.to(dtype=torch.long)

        return image

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, Any], dict[str, Any], SampleMetadata]:
        """Get the dataset item at the specified index."""
        tile_name = self.tile_list[idx]
        raw_inputs = {}
        passthrough_inputs = {}

        # Currently all the inputs probably need to be multi-temporal while all the
        # others need to not be. This may not be ideal.
        # TODO: sub-sample from available input modalities. Need to adjust model to
        # accept different subsets of inputs.
        for modality in self.input_modalities:
            image = self._load_modality(tile_name, modality)
            if image is None:
                continue
            info = MODALITIES[modality]
            if info.is_multitemporal:
                # (T*C, H, W) -> (C, T, H, W)
                image = rearrange(image, "(t c) h w -> c t h w", c=info.num_bands)
            else:
                # (C, H, W) -> (C, 1, H, W)
                image = image[:, None, :, :]
            passthrough_inputs[modality] = RasterImage(image=image)
        for modality in self.target_modalities:
            # For the targets we add one if they are present, otherwise set to zero.
            image = self._load_modality(tile_name, modality)
            if image is None:
                info = MODALITIES[modality]
                tile_size = TILE_SIZE // info.resolution_div
                if info.norm_factor is not None:
                    image = torch.zeros(
                        (info.num_bands, tile_size, tile_size), dtype=torch.float32
                    )
                else:
                    image = torch.zeros(
                        (info.num_bands, tile_size, tile_size), dtype=torch.long
                    )
            else:
                image = image + 1
            raw_inputs[modality] = RasterImage(image[:, None, :, :])

        sample_metadata = SampleMetadata(
            window_group="fake",
            window_name=tile_name,
            window_bounds=(0, 0, TILE_SIZE, TILE_SIZE),
            crop_bounds=(0, 0, TILE_SIZE, TILE_SIZE),
            crop_idx=0,
            num_crops_in_window=1,
            time_range=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 2, 1, tzinfo=UTC),
            ),
            projection=Projection(CRS.from_epsg(32610), 10, -10),
            dataset_source=None,
        )

        input_dict, target_dict = self.task.process_inputs(
            raw_inputs,
            metadata=sample_metadata,
            load_targets=True,
        )
        input_dict.update(passthrough_inputs)
        # input_dict, target_dict = self.transforms(input_dict, target_dict)

        return input_dict, target_dict, sample_metadata


class OlmoEarthDataModule(L.LightningDataModule):
    """Data module for OlmoEarth data."""

    def __init__(
        self,
        ds_path: Path,
        task: Task,
        input_modalities: list[str],
        target_modalities: list[str],
        num_val_examples: int,
        batch_size: int,
        num_workers: int,
        min_size: int = 256,
        max_size: int = 256,
        patch_size: int = 32,
    ) -> None:
        """Initialize a new DataModule."""
        super().__init__()
        self.ds_path = ds_path
        self.task = task
        self.input_modalities = input_modalities
        self.target_modalities = target_modalities
        self.num_val_examples = num_val_examples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_size = min_size
        self.max_size = max_size
        self.patch_size = patch_size

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit' or 'validate'
        """
        # Setup training dataset only for fit command.
        if stage == "fit":
            self.train_dataset = OlmoEarthDataset(
                ds_path=self.ds_path,
                task=self.task,
                input_modalities=self.input_modalities,
                target_modalities=self.target_modalities,
                skip=self.num_val_examples,
            )

        # Setup validation dataset.
        self.val_dataset = OlmoEarthDataset(
            ds_path=self.ds_path,
            task=self.task,
            input_modalities=self.input_modalities,
            target_modalities=self.target_modalities,
            limit=self.num_val_examples,
        )

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Implement one or more PyTorch DataLoaders for training.

        Returns:
            A collection of data loaders specifying training samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=CollateFunction(
                randomize=True,
                min_size=self.min_size,
                max_size=self.max_size,
                patch_size=self.patch_size,
            ),
            persistent_workers=True,
        )
        if (
            self.trainer is not None
            and self.trainer.world_size is not None
            and self.trainer.world_size > 1
        ):
            kwargs["sampler"] = DistributedSampler(
                self.train_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
            )
        else:
            kwargs["shuffle"] = True

        return DataLoader(**kwargs)

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        """Implement one or more PyTorch DataLoaders for validation.

        Returns:
            A collection of data loaders specifying validation samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        kwargs = dict(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=CollateFunction(
                randomize=False,
                min_size=self.min_size,
                max_size=self.max_size,
                patch_size=self.patch_size,
            ),
            persistent_workers=True,
        )
        if (
            self.trainer is not None
            and self.trainer.world_size is not None
            and self.trainer.world_size > 1
        ):
            kwargs["sampler"] = DistributedSampler(
                self.val_dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=False,
            )

        return DataLoader(**kwargs)
