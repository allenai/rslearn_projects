"""Utilities for using rslearn datasets and models."""

from rslearn.dataset import Dataset
from rslearn.main import (
    IngestHandler,
    MaterializeHandler,
    PrepareHandler,
    apply_on_windows,
)
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
from upath import UPath

from rslp.lightning_cli import CustomLightningCLI


def materialize_dataset(
    ds_path: UPath, group: str | None = None, workers: int = 32
) -> None:
    """Materialize the specified dataset by running prepare/ingest/materialize.

    Args:
        ds_path: the dataset root.
        group: limit dataset actions to this group.
        workers: number of workers to use.
    """
    dataset = Dataset(ds_path)
    apply_on_windows(
        PrepareHandler(force=False),
        dataset,
        workers=workers,
        group=group,
    )
    apply_on_windows(
        IngestHandler(),
        dataset,
        workers=workers,
        group=group,
    )
    apply_on_windows(
        MaterializeHandler(),
        dataset,
        workers=workers,
        group=group,
    )


def run_model_predict(
    model_cfg_fname: str, ds_path: UPath, extra_args: list[str] = []
) -> None:
    """Call rslearn model predict.

    Args:
        model_cfg_fname: the model configuration file.
        ds_path: the dataset root path.
        extra_args: additional arguments to pass to model predict.
    """
    CustomLightningCLI(
        model_class=RslearnLightningModule,
        datamodule_class=RslearnDataModule,
        args=[
            "predict",
            "--config",
            model_cfg_fname,
            "--load_best=true",
            "--data.init_args.path",
            str(ds_path),
        ]
        + extra_args,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )
