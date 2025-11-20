"""Evaluation adapter for TerraMind."""

import os

import torch
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.resize_features import ResizeFeatures
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.terramind import Terramind, TerramindNormalize, TerramindSize
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
    """Get appropriate TerraMind model."""
    model_id = os.environ["EVAL_ADAPTER_MODEL_ID"]
    if model_id == "terramind":
        terramind_size = TerramindSize.BASE
        embedding_size = 768
    elif model_id == "terramind_large":
        terramind_size = TerramindSize.LARGE
        embedding_size = 1024
    else:
        raise ValueError(f"unknown terramind model ID {model_id}")

    if task_type == "segment":
        decoders = dict(
            eval_task=[
                UNetDecoder(
                    in_channels=[[16, embedding_size]],
                    out_channels=task_channels,
                    conv_layers_per_resolution=2,
                    num_channels={16: 512, 8: 512, 4: 512, 2: 256, 1: 128},
                    original_size_to_interpolate=[input_size, input_size],
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "segment_small":
        decoders = dict(
            eval_task=[
                SegmentationPoolingDecoder(
                    in_channels=embedding_size,
                    out_channels=task_channels,
                ),
                SegmentationHead(),
            ]
        )
    elif task_type == "detect":
        decoders = dict(
            eval_task=[
                # TerraMind patch_size = 16
                ResizeFeatures(out_sizes=[(input_size // 16, input_size // 16)]),
                FasterRCNN(
                    downsample_factors=[16],
                    num_channels=embedding_size,
                    num_classes=task_channels,
                    anchor_sizes=[[32]],
                ),
            ]
        )
    elif task_type == "classify":
        decoders = dict(
            eval_task=[
                PoolingDecoder(
                    in_channels=embedding_size,
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
                    in_channels=embedding_size,
                    out_channels=task_channels,
                    num_conv_layers=1,
                    num_fc_layers=1,
                ),
                RegressionHead(),
            ]
        )
    else:
        raise NotImplementedError

    modalities = []
    image_keys = {}
    if "sentinel2" in input_modalities:
        modalities.append("S2L2A")
        image_keys["S2L2A"] = 12
    if "sentinel1" in input_modalities:
        modalities.append("S1GRD")
        image_keys["S1GRD"] = 2

    if task_name == "forest_loss_driver":
        return MultiTaskModel(
            encoder=[
                SimpleTimeSeries(
                    encoder=SimpleTimeSeries(
                        encoder=Terramind(
                            model_size=terramind_size,
                            modalities=modalities,
                            do_resizing=True,
                        ),
                        image_keys=image_keys,
                    ),
                    image_channels=12 * 4,
                    image_key="S2L2A",
                    groups=[[0], [1]],
                ),
            ],
            decoders=dict(
                eval_task=[
                    PoolingDecoder(
                        in_channels=embedding_size * 2,
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
                encoder=Terramind(
                    model_size=terramind_size,
                    modalities=modalities,
                    do_resizing=True,
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
    """Get appropriate TerraMind transform."""
    modules: list[torch.nn.Module] = []

    if "sentinel2" in input_modalities:
        wanted_sentinel2 = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]
        sentinel2_indexes = [SENTINEL2_BANDS.index(band) for band in wanted_sentinel2]
        modules.append(
            SelectBands(
                band_indices=sentinel2_indexes,
                num_bands_per_timestep=len(SENTINEL2_BANDS),
                input_selector="sentinel2",
                output_selector="S2L2A",
            )
        )

    if "sentinel1" in input_modalities:
        wanted_sentinel1 = ["vv", "vh"]
        sentinel1_indexes = [SENTINEL1_BANDS.index(band) for band in wanted_sentinel1]
        modules.append(
            SelectBands(
                band_indices=sentinel1_indexes,
                num_bands_per_timestep=len(SENTINEL1_BANDS),
                input_selector="sentinel1",
                output_selector="S1GRD",
            )
        )

    modules.append(TerramindNormalize())

    return Sequential(*modules)
