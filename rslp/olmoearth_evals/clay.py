"""Evaluation adapter for Clay."""

import torch
from rslearn.models.clay.clay import Clay, ClayNormalize, ClaySize
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.models.unet import UNetDecoder
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
from rslearn.train.transforms.select_bands import SelectBands

from rslp.nandi.train import SegmentationPoolingDecoder

from .constants import LANDSAT_BANDS, SENTINEL1_BANDS, SENTINEL2_BANDS


def get_model(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate Clay model."""
    if task_type == "segment":
        decoders = dict(
            eval_task=[
                UNetDecoder(
                    in_channels=[[8, 1024]],
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
                    downsample_factors=[8],
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

    if len(input_modalities) != 1:
        raise ValueError("Clay only supports one input modality at a time")

    input_modality = input_modalities[0]
    image_keys = {}
    if input_modality == "sentinel2":
        clay_modality = "sentinel-2-l2a"
        image_keys[clay_modality] = 10
    elif input_modality == "sentinel1":
        clay_modality = "sentinel-1-rtc"
        image_keys[clay_modality] = 2
    elif input_modality == "landsat":
        clay_modality = "landsat-c2l1"
        image_keys[clay_modality] = 6

    if task_name == "forest_loss_driver":
        return MultiTaskModel(
            encoder=[
                SimpleTimeSeries(
                    encoder=SimpleTimeSeries(
                        encoder=Clay(
                            model_size=ClaySize.LARGE,
                            modality=clay_modality,
                        ),
                        image_keys=image_keys,
                    ),
                    image_channels=10 * 4,
                    image_key="sentinel-2-l2a",
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
                encoder=Clay(
                    model_size=ClaySize.LARGE,
                    modality=clay_modality,
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
    """Get appropriate Clay transform."""
    modules: list[torch.nn.Module] = []

    if "sentinel2" in input_modalities:
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
                output_selector="sentinel-2-l2a",
            ),
        )

    if "landsat" in input_modalities:
        # In metadata.yaml it says red, green, blue, nir08, swir16, swir22.
        wanted_landsat = ["B4", "B3", "B2", "B5", "B6", "B7"]
        landsat_indexes = [LANDSAT_BANDS.index(band) for band in wanted_landsat]
        modules.append(
            SelectBands(
                band_indices=landsat_indexes,
                num_bands_per_timestep=len(LANDSAT_BANDS),
                input_selector="landsat",
                output_selector="landsat-c2l1",
            ),
        )

    if "sentinel1" in input_modalities:
        wanted_sentinel1 = ["vv", "vh"]
        sentinel1_indexes = [SENTINEL1_BANDS.index(band) for band in wanted_sentinel1]
        modules.append(
            SelectBands(
                band_indices=sentinel1_indexes,
                num_bands_per_timestep=len(SENTINEL1_BANDS),
                input_selector="sentinel1",
                output_selector="sentinel-1-rtc",
            ),
        )

    modules.append(ClayNormalize())

    return Sequential(*modules)
