"""Evaluation adapter for CROMA."""

import torch
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pick_features import PickFeatures
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.prithvi import PrithviNormalize, PrithviV2
from rslearn.models.resize_features import ResizeFeatures
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.unet import UNetDecoder
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
from rslearn.train.transforms.concatenate import Concatenate
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
    """Get appropriate Prithvi model."""
    if task_type == "segment":
        decoders = dict(
            eval_task=[
                PickFeatures(indexes=[10]),
                UNetDecoder(
                    in_channels=[[16, 1024]],
                    out_channels=task_channels,
                    conv_layers_per_resolution=2,
                    num_channels={8: 512, 4: 512, 2: 256, 1: 128},
                    original_size_to_interpolate=[input_size, input_size],
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "segment_small":
        decoders = dict(
            eval_task=[
                PickFeatures(indexes=[10]),
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
                PickFeatures(indexes=[10]),
                ResizeFeatures(out_sizes=[(input_size // 16, input_size // 16)]),
                FasterRCNN(
                    downsample_factors=[16],
                    num_channels=1024,
                    num_classes=task_channels,
                    anchor_sizes=[[32]],
                ),
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
                    encoder=PrithviV2(
                        num_frames=task_timesteps,
                    ),
                    image_channels=6 * 4,
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
            PrithviV2(
                num_frames=task_timesteps,
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
    """Get appropriate Prithvi transform."""
    modules: list[torch.nn.Module] = []
    concatenate_selections: dict[str, list] = {}

    if "sentinel2" in input_modalities:
        # We pick Sentinel-2 bands that best correspond to HLS bands 2, 3, 4, 5, 6, 7.
        # Ideally we would input HLS Sentinel-2 but we don't have that.
        wanted_bands = ["B02", "B03", "B04", "B08", "B11", "B12"]
        band_indices = [SENTINEL2_BANDS.index(band) for band in wanted_bands]
        modules.append(
            SelectBands(
                band_indices=band_indices,
                num_bands_per_timestep=len(SENTINEL2_BANDS),
                input_selector="sentinel2",
                output_selector="sentinel2",
            )
        )
        concatenate_selections["sentinel2"] = []

    if "landsat" in input_modalities:
        wanted_bands = ["B2", "B3", "B4", "B5", "B6", "B7"]
        band_indices = [LANDSAT_BANDS.index(band) for band in wanted_bands]
        modules.append(
            SelectBands(
                band_indices=band_indices,
                num_bands_per_timestep=len(LANDSAT_BANDS),
                input_selector="landsat",
                output_selector="landsat",
            )
        )
        concatenate_selections["landsat"] = []

    # We concatenate the Sentinel-2 where we picked HLS-ish bands, and Landsat.
    # And output the expected "image" key of stacked HLS images.
    modules.append(
        Concatenate(selections=concatenate_selections, output_selector="image")
    )

    modules.append(PrithviNormalize())
    return Sequential(*modules)
