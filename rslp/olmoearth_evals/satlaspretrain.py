"""Evaluation adapter for SatlasPretrain."""

import torch
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.fpn import Fpn
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.swin import Swin
from rslearn.models.unet import UNetDecoder
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
from rslearn.train.transforms.normalize import Normalize
from rslearn.train.transforms.select_bands import SelectBands

from .constants import SENTINEL2_BANDS


def get_model(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate SatlasPretrain model."""
    if task_type == "segment":
        decoders = dict(
            eval_task=[
                UNetDecoder(
                    in_channels=[[4, 128], [8, 256], [16, 512], [32, 1024]],
                    out_channels=task_channels,
                    conv_layers_per_resolution=2,
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "segment_small":
        raise NotImplementedError
    elif task_type == "detect":
        decoders = dict(
            eval_task=[
                FasterRCNN(
                    downsample_factors=[4, 8, 16, 32],
                    num_channels=128,
                    num_classes=task_channels,
                    anchor_sizes=[[32], [64], [128], [256]],
                )
            ],
        )
    elif task_type == "classify":
        decoders = dict(
            eval_task=[
                PoolingDecoder(
                    in_channels=1024,
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
                    in_channels=1024,
                    out_channels=task_channels,
                    num_conv_layers=1,
                    num_fc_layers=1,
                ),
                RegressionHead(),
            ]
        )
    else:
        raise NotImplementedError

    if task_type == "detect":
        return MultiTaskModel(
            encoder=[
                SimpleTimeSeries(
                    encoder=Swin(
                        pretrained=False,
                        input_channels=9,
                        output_layers=[1, 3, 5, 7],
                    ),
                    image_channels=9,
                ),
                Fpn(
                    in_channels=[128, 256, 512, 1024],
                    out_channels=128,
                ),
            ],
            decoders=decoders,
        )
    else:
        return MultiTaskModel(
            encoder=[
                SimpleTimeSeries(
                    encoder=Swin(
                        pretrained=False,
                        input_channels=9,
                        output_layers=[1, 3, 5, 7],
                    ),
                    image_channels=9,
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
    """Get appropriate SatlasPretrain transform."""
    if input_modalities != ["sentinel2"]:
        raise ValueError(
            "satlaspretrain evaluation is only designed to work with Sentinel-2 input"
        )

    modules: list[torch.nn.Module] = []

    wanted_bands = ["B04", "B03", "B02", "B05", "B06", "B07", "B08", "B11", "B12"]
    band_indices = [SENTINEL2_BANDS.index(band) for band in wanted_bands]
    modules.append(
        SelectBands(
            band_indices=band_indices,
            num_bands_per_timestep=len(SENTINEL2_BANDS),
            input_selector="sentinel2",
            output_selector="image",
        )
    )
    # Normalization for RGB bands.
    modules.append(
        Normalize(
            mean=0,
            std=3000,
            valid_range=[0, 1],
            bands=[0, 1, 2],
            num_bands=9,
        )
    )
    # Normalization for the other bands.
    modules.append(
        Normalize(
            mean=0,
            std=8160,
            valid_range=[0, 1],
            bands=[3, 4, 5, 6, 7, 8],
            num_bands=9,
        )
    )

    return Sequential(*modules)
