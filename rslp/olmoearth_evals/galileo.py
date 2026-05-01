"""Evaluation adapter for Galileo."""

import torch
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.galileo import GalileoModel, GalileoSize
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.unet import UNetDecoder
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
from rslearn.train.transforms.select_bands import SelectBands

from rslp.nandi.train import SegmentationPoolingDecoder

from .constants import SENTINEL1_BANDS, SENTINEL2_BANDS


def get_model(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate Galileo model."""
    if task_type == "segment":
        decoders = dict(
            eval_task=[
                UNetDecoder(
                    in_channels=[[4, 768]],
                    out_channels=task_channels,
                    conv_layers_per_resolution=2,
                    num_channels={4: 512, 2: 256, 1: 128},
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
                    downsample_factors=[4],
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

    return MultiTaskModel(
        encoder=[
            GalileoModel(
                size=GalileoSize.BASE,
                patch_size=4,
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
    """Get appropriate Galileo transform."""
    modules: list[torch.nn.Module] = []

    for input_modality in input_modalities:
        if input_modality == "sentinel2":
            wanted_sentinel2 = [
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
            sentinel2_indexes = [
                SENTINEL2_BANDS.index(band) for band in wanted_sentinel2
            ]
            modules.append(
                SelectBands(
                    band_indices=sentinel2_indexes,
                    num_bands_per_timestep=len(SENTINEL2_BANDS),
                    input_selector="sentinel2",
                    output_selector="s2",
                ),
            )

        elif input_modality == "sentinel1":
            wanted_sentinel1 = ["vv", "vh"]
            sentinel1_indexes = [
                SENTINEL1_BANDS.index(band) for band in wanted_sentinel1
            ]
            modules.append(
                SelectBands(
                    band_indices=sentinel1_indexes,
                    num_bands_per_timestep=len(SENTINEL1_BANDS),
                    input_selector="sentinel1",
                    output_selector="s1",
                ),
            )

        else:
            raise ValueError(f"galileo does not support modality {input_modality}")

    # For GalileoModel, the actual normalization is handled within the model forward
    # pass.

    return Sequential(*modules)
