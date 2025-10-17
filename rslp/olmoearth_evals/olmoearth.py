"""Evaluation adapter for OlmoEarth."""

import torch
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
from rslp.olmoearth_pretrain.model import OlmoEarth
from rslp.olmoearth_pretrain.norm import OlmoEarthNormalize

from .constants import LANDSAT_BANDS, SENTINEL1_BANDS, SENTINEL2_BANDS


def get_model(
    input_size: int,
    input_modalities: list[str],
    task_type: str,
    task_name: str,
    task_channels: int = 1,
    task_timesteps: int = 1,
) -> torch.nn.Module:
    """Get appropriate OlmoEarth model."""
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

    if task_name == "forest_loss_driver":
        return MultiTaskModel(
            encoder=[
                SimpleTimeSeries(
                    encoder=OlmoEarth(
                        checkpoint_path="/weka/dfive-default/olmoearth_pretrain/checkpoints/henryh/base_v6.1_add_chm_cdl_worldcereal/step500000",
                        selector=["encoder"],
                        forward_kwargs=dict(patch_size=4),
                        patch_size=4,
                        embedding_size=768,
                    ),
                    image_channels=12 * 4,
                    image_key="sentinel2_l2a",
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
            OlmoEarth(
                checkpoint_path="/weka/dfive-default/olmoearth_pretrain/checkpoints/henryh/base_v6.1_add_chm_cdl_worldcereal/step500000",
                selector=["encoder"],
                forward_kwargs=dict(patch_size=4),
                patch_size=4,
                embedding_size=768,
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
    """Get appropriate OlmoEarth transform."""
    modules: list[torch.nn.Module] = []
    band_names: dict = {}

    if "sentinel2" in input_modalities:
        wanted_sentinel2 = [
            "B02",
            "B03",
            "B04",
            "B08",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "B01",
            "B09",
        ]
        sentinel2_indexes = [SENTINEL2_BANDS.index(band) for band in wanted_sentinel2]
        modules.append(
            SelectBands(
                band_indices=sentinel2_indexes,
                num_bands_per_timestep=len(SENTINEL2_BANDS),
                input_selector="sentinel2",
                output_selector="sentinel2_l2a",
            ),
        )
        band_names["sentinel2_l2a"] = wanted_sentinel2

    if "landsat" in input_modalities:
        wanted_landsat = [
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
        landsat_indexes = [LANDSAT_BANDS.index(band) for band in wanted_landsat]
        modules.append(
            SelectBands(
                band_indices=landsat_indexes,
                num_bands_per_timestep=len(LANDSAT_BANDS),
                input_selector="landsat",
                output_selector="landsat",
            ),
        )
        band_names["landsat"] = wanted_landsat

    if "sentinel1" in input_modalities:
        wanted_sentinel1 = ["vv", "vh"]
        sentinel1_indexes = [SENTINEL1_BANDS.index(band) for band in wanted_sentinel1]
        modules.append(
            SelectBands(
                band_indices=sentinel1_indexes,
                num_bands_per_timestep=len(SENTINEL1_BANDS),
                input_selector="sentinel1",
                output_selector="sentinel1",
            ),
        )
        band_names["sentinel1"] = wanted_sentinel1

    modules.append(
        OlmoEarthNormalize(
            band_names=band_names,
        )
    )

    return Sequential(*modules)
