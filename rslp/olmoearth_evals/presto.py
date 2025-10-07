"""Evaluation adapter for Presto."""

import torch
from rslearn.models.conv import Conv
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pick_features import PickFeatures
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.presto import Presto
from rslearn.models.resize_features import ResizeFeatures
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
from rslearn.train.transforms.select_bands import SelectBands

from .constants import SENTINEL1_BANDS, SENTINEL2_BANDS


def get_model(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate Presto model."""
    if task_type == "segment":
        decoders = dict(
            eval_task=[
                Conv(
                    in_channels=128,
                    out_channels=task_channels,
                    kernel_size=1,
                    activation=torch.nn.Identity(),
                ),
                PickFeatures(
                    indexes=[0],
                    collapse=True,
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "segment_small":
        decoders = dict(
            eval_task=[
                Conv(
                    in_channels=128,
                    out_channels=task_channels,
                    kernel_size=1,
                    activation=torch.nn.Identity(),
                ),
                PickFeatures(
                    indexes=[0],
                    collapse=True,
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "detect":
        decoders = dict(
            eval_task=[
                ResizeFeatures(out_sizes=[(input_size // 4, input_size // 4)]),
                FasterRCNN(
                    downsample_factors=[4],
                    num_channels=128,
                    num_classes=task_channels,
                    anchor_sizes=[[32]],
                ),
            ]
        )
    elif task_type == "classify":
        decoders = dict(
            eval_task=[
                PoolingDecoder(
                    in_channels=128,
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
                    in_channels=128,
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
        encoder=[Presto(pixel_batch_size=16384)],
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
    """Get appropriate Presto transform."""
    modules: list[torch.nn.Module] = []

    if "sentinel2" in input_modalities:
        # Rename to s2 and pick different bands.
        # Presto doesn't use B01 or B09.
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
        sentinel2_indexes = [SENTINEL2_BANDS.index(band) for band in wanted_sentinel2]
        modules.append(
            SelectBands(
                band_indices=sentinel2_indexes,
                num_bands_per_timestep=len(SENTINEL2_BANDS),
                input_selector="sentinel2",
                output_selector="s2",
            )
        )

    if "sentinel1" in input_modalities:
        # Rename to s1.
        wanted_sentinel1 = ["vv", "vh"]
        sentinel1_indexes = [SENTINEL1_BANDS.index(band) for band in wanted_sentinel1]
        modules.append(
            SelectBands(
                band_indices=sentinel1_indexes,
                num_bands_per_timestep=len(SENTINEL1_BANDS),
                input_selector="sentinel1",
                output_selector="s1",
            )
        )

    return Sequential(*modules)
