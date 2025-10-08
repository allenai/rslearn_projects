"""Evaluation adapter for DINOv3."""

import torch
from rslearn.models.dinov3 import DinoV3, DinoV3Normalize
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.unet import UNetDecoder
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
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
    """Get appropriate DINOv3 model."""
    # DINOv3 resizes to 256x256 and always has 16x16 output feature map.
    downsample_factor = input_size // 16
    if task_type == "segment":
        decoders = dict(
            eval_task=[
                UNetDecoder(
                    in_channels=[[downsample_factor, 1024]],
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
                    in_channels=1024,
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
                    num_channels=1024,
                    num_classes=task_channels,
                    anchor_sizes=[[32]],
                )
            ]
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

    if task_name == "forest_loss_driver":
        return MultiTaskModel(
            encoder=[
                SimpleTimeSeries(
                    encoder=SimpleTimeSeries(
                        encoder=DinoV3(
                            # Needs to be changed if you are outside Ai2, this is Ai2-internal path.
                            checkpoint_dir="/weka/dfive-default/helios/models/dinov3/checkpoints/",
                        ),
                        image_channels=3,
                    ),
                    image_channels=3 * 4,
                    image_key="image",
                    groups=[[0], [1]],
                ),
            ],
            decoders=dict(
                eval_task=[
                    PoolingDecoder(
                        in_channels=1024 * 2,
                        out_channels=task_channels,
                        num_conv_layers=1,
                        num_fc_layers=1,
                    ),
                    ClassificationHead(),
                ]
            ),
        )

    return MultiTaskModel(
        encoder=[
            SimpleTimeSeries(
                encoder=DinoV3(
                    # Needs to be changed if you are outside Ai2, this is Ai2-internal path.
                    checkpoint_dir="/weka/dfive-default/helios/models/dinov3/checkpoints/",
                ),
                image_channels=3,
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
    """Get appropriate DINOv3 transform."""
    if len(input_modalities) != 1:
        raise ValueError("dinov3 only supports one modality at a time")

    input_modality = input_modalities[0]
    modules: list[torch.nn.Module] = []

    # Convert the modality to RGB 0-255.
    # After normalizing to 0-255, DinoV3Normalize will do the rest.

    if input_modality == "landsat":
        wanted_bands = ["B4", "B3", "B2"]
        band_indices = [LANDSAT_BANDS.index(band) for band in wanted_bands]
        modules.append(
            SelectBands(
                band_indices=band_indices,
                num_bands_per_timestep=len(LANDSAT_BANDS),
                input_selector="landsat",
                output_selector="image",
            )
        )
        modules.append(
            Normalize(
                mean=5000,
                std=12000,
                valid_range=(0, 1),
                selectors=["image"],
            )
        )

    elif input_modality == "sentinel2":
        wanted_bands = ["B04", "B03", "B02"]
        band_indices = [SENTINEL2_BANDS.index(band) for band in wanted_bands]
        modules.append(
            SelectBands(
                band_indices=band_indices,
                num_bands_per_timestep=len(SENTINEL2_BANDS),
                input_selector="sentinel2",
                output_selector="image",
            )
        )
        modules.append(
            Normalize(
                mean=2000,
                std=1500,
                selectors=["image"],
            )
        )

    else:
        raise ValueError("dinov3 only supports sentinel2 or landsat")

    modules.append(DinoV3Normalize())
    return Sequential(*modules)
