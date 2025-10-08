"""Evaluation adapter for Panopticon."""

import torch
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.panopticon import Panopticon
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.unet import UNetDecoder
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
from rslearn.train.transforms.concatenate import Concatenate
from rslearn.train.transforms.normalize import Normalize
from rslearn.train.transforms.select_bands import SelectBands

from rslp.nandi.train import SegmentationPoolingDecoder

from .constants import LANDSAT_BANDS, SENTINEL2_BANDS


def get_model(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate Panopticon model."""
    # Panopticon resizes to 256x256 and always has 16x16 output feature map.
    downsample_factor = input_size // 16
    if task_type == "segment":
        decoders = dict(
            eval_task=[
                UNetDecoder(
                    in_channels=[[downsample_factor, 768]],
                    out_channels=task_channels,
                    conv_layers_per_resolution=2,
                    num_channels={8: 512, 4: 512, 2: 256, 1: 128},
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "segment_small":
        decoders = dict(
            eval_task=[
                SegmentationPoolingDecoder(
                    in_channels=768,
                    out_channels=task_channels,
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "detect":
        decoders = dict(
            eval_task=[
                FasterRCNN(
                    downsample_factors=[downsample_factor],
                    num_channels=768,
                    num_classes=task_channels,
                    anchor_sizes=[[32]],
                )
            ]
        )
    elif task_type == "classify":
        decoders = dict(
            eval_task=[
                PoolingDecoder(
                    in_channels=768,
                    out_channels=task_channels,
                    num_conv_layers=1,
                    num_fc_layers=1,
                ),
                ClassificationHead(),
            ]
        )
    elif task_type == "regress":
        decoders = dict(
            eval_task=[
                PoolingDecoder(
                    in_channels=768,
                    out_channels=task_channels,
                    num_conv_layers=1,
                    num_fc_layers=1,
                ),
                RegressionHead(),
            ]
        )
    else:
        raise NotImplementedError

    band_order: dict = {}
    image_keys: dict = {}
    for modality in input_modalities:
        if modality == "sentinel2":
            if task_name == "pastis":
                # PASTIS is missing some bands and Panopticon can deal with this.
                band_order["sentinel2"] = [
                    "B02",
                    "B03",
                    "B04",
                    "B05",
                    "B06",
                    "B07",
                    "B08",
                    "B8A",
                    "B11",
                    "B12",
                ]
                image_keys["sentinel2"] = 10
            else:
                band_order["sentinel2"] = SENTINEL2_BANDS
                image_keys["sentinel2"] = 12
        elif modality == "sentinel1":
            band_order["sentinel1"] = ["VV", "VH"]
            image_keys["sentinel1"] = 2
        elif modality == "landsat":
            band_order["landsat8"] = LANDSAT_BANDS
            image_keys["landsat8"] = 11

    return MultiTaskModel(
        encoder=[
            SimpleTimeSeries(
                encoder=Panopticon(
                    band_order=band_order,
                ),
                image_keys=image_keys,
            ),
        ],
        decoders=decoders,
    )


def get_transform(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate Panopticon transform."""
    modules: list[torch.nn.Module] = []

    # Rename landsat to landsat8, and normalize, if it is present.
    if "landsat" in input_modalities:
        modules.append(
            Concatenate(
                selections={"landsat": []},
                output_selector="landsat8",
            )
        )
        modules.append(
            Normalize(
                mean=10000,
                std=8000,
                selectors=["landsat8"],
            )
        )

    if "sentinel2" in input_modalities:
        if task_name == "pastis":
            wanted_bands = [
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B8A",
                "B11",
                "B12",
            ]
        else:
            wanted_bands = SENTINEL2_BANDS

        band_indices = [SENTINEL2_BANDS.index(band) for band in wanted_bands]
        modules.append(
            SelectBands(
                band_indices=band_indices,
                num_bands_per_timestep=len(SENTINEL2_BANDS),
                input_selector="sentinel2",
                output_selector="sentinel2",
            )
        )
        modules.append(
            Normalize(
                mean=2000,
                std=1500,
                selectors=["sentinel2"],
            )
        )

    if "sentinel1" in input_modalities:
        # For Sentinel-1, we just need to normalize it.
        modules.append(
            Normalize(
                mean=-10,
                std=10,
                selectors=["sentinel1"],
            )
        )

    return Sequential(*modules)
