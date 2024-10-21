"""Model training pipeline for Maldives ecosystem mapping project."""

import os
import shutil

from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
from upath import UPath

from rslp.lightning_cli import CustomLightningCLI


def maxar_predict_pipeline(out_dir: str) -> None:
    """Run the prediction pipeline.

    Args:
        out_dir: directory to write the output GeoTIFFs.
    """
    model_cfg_fname = "data/maldives_ecosystem_mapping/config.yaml"
    lightning_cli = CustomLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=[
            "predict",
            "--config",
            model_cfg_fname,
            "--autoresume=true",
        ],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )
    group_name = "images"
    path: UPath = lightning_cli.datamodule.path / "windows" / group_name
    for fname in path.glob("*/layers/output/output/geotiff.tif"):
        window_name = fname.parent.parent.parent.parent.name
        local_fname = os.path.join(out_dir, f"{window_name}.tif")
        with fname.open("rb") as src:
            with open(local_fname, "wb") as dst:
                shutil.copyfileobj(src, dst)


def sentinel2_predict_pipeline(out_dir: str) -> None:
    """Run the prediction pipeline.

    Args:
        out_dir: directory to write the output GeoTIFFs.
    """
    model_cfg_fname = "data/maldives_ecosystem_mapping/config_sentinel2.yaml"
    lightning_cli = CustomLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=[
            "predict",
            "--config",
            model_cfg_fname,
            "--autoresume=true",
        ],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )
    group_name = "images_sentinel2"
    path: UPath = lightning_cli.datamodule.path / "windows" / group_name
    for fname in path.glob("*/layers/output/output/geotiff.tif"):
        window_name = fname.parent.parent.parent.parent.name
        local_fname = os.path.join(out_dir, f"{window_name}.tif")
        with fname.open("rb") as src:
            with open(local_fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
