"""Evaluation adapter for AnySat."""

import torch
from rslearn.models.anysat import AnySat
from rslearn.models.conv import Conv
from rslearn.models.faster_rcnn import FasterRCNN
from rslearn.models.multitask import MultiTaskModel
from rslearn.models.pick_features import PickFeatures
from rslearn.models.pooling_decoder import PoolingDecoder
from rslearn.models.simple_time_series import SimpleTimeSeries
from rslearn.train.tasks.classification import ClassificationHead
from rslearn.train.tasks.regression import RegressionHead
from rslearn.train.tasks.segmentation import SegmentationHead
from rslearn.train.transforms import Sequential
from rslearn.train.transforms.normalize import Normalize
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
    """Get appropriate AnySat model."""
    output_mode = "dense"
    if task_type == "segment":
        decoders = dict(
            eval_task=[
                Conv(
                    in_channels=1536,
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
                SegmentationPoolingDecoder(
                    in_channels=1536,
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
        output_mode = "patch"
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
        output_mode = "patch"
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
        output_mode = "patch"
    else:
        raise NotImplementedError

    modalities = []
    for modality in input_modalities:
        if modality == "sentinel2":
            modalities.append("s2")
        elif modality == "sentinel1":
            modalities.append("s1-asc")
        elif modality == "landsat":
            modalities.append("l8")
        else:
            raise ValueError(f"unsupported modality {modality}")

    dates = list(range(0, 30 * task_timesteps, 30))

    if task_name == "forest_loss_driver":
        return MultiTaskModel(
            encoder=[
                SimpleTimeSeries(
                    encoder=AnySat(
                        modalities=modalities,
                        patch_size_meters=40,
                        output=output_mode,
                        dates={modality: dates for modality in modalities},
                        output_modality=modalities[0],
                    ),
                    image_channels=10 * 4,
                    image_key="s2",
                    groups=[[0], [1]],
                ),
            ],
            decoders=dict(
                eval_task=[
                    PoolingDecoder(
                        in_channels=768 * 2,
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
            AnySat(
                modalities=modalities,
                patch_size_meters=40,
                output=output_mode,
                dates={modality: dates for modality in modalities},
                output_modality=modalities[0],
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
    """Get appropriate AnySat transform."""
    modules: list[torch.nn.Module] = []

    if "sentinel2" in input_modalities:
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
        band_indices = [SENTINEL2_BANDS.index(band) for band in wanted_bands]
        modules.append(
            SelectBands(
                band_indices=band_indices,
                num_bands_per_timestep=len(SENTINEL2_BANDS),
                input_selector="sentinel2",
                output_selector="s2",
            )
        )
        modules.append(
            Normalize(
                mean=2000,
                std=1500,
                selectors=["s2"],
            )
        )

    if "sentinel1" in input_modalities:
        wanted_bands = ["vv", "vh"]
        band_indices = [SENTINEL1_BANDS.index(band) for band in wanted_bands]
        modules.append(
            SelectBands(
                band_indices=band_indices,
                num_bands_per_timestep=len(SENTINEL1_BANDS),
                input_selector="sentinel1",
                output_selector="s1-asc",
            )
        )
        modules.append(
            Normalize(
                mean=-10,
                std=10,
                selectors=["s1-asc"],
            )
        )

    if "landsat" in input_modalities:
        wanted_bands = [
            "B8",
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B9",
            "B10",
            "B11",
        ]
        band_indices = [LANDSAT_BANDS.index(band) for band in wanted_bands]
        modules.append(
            SelectBands(
                band_indices=band_indices,
                num_bands_per_timestep=len(LANDSAT_BANDS),
                input_selector="landsat",
                output_selector="l8",
            )
        )
        modules.append(
            Normalize(
                mean=10000,
                std=8000,
                selectors=["l8"],
            )
        )

    return Sequential(*modules)
