"""Utilities for using rslearn datasets and models."""

from rslearn.dataset import Dataset

# Should wandb required from rslearn to run rslp?
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
from rslp.log_utils import get_logger

logger = get_logger(__name__)


def materialize_dataset(
    ds_path: UPath,
    ignore_errors: bool = False,
    disabled_layers: list[str] = [],
    group: str | None = None,
    workers: int = 32,
) -> None:
    """Materialize the specified dataset by running prepare/ingest/materialize.

    Args:
        ds_path: the dataset root.
        ignore_errors: whether to ignore errors, this allows us to ignore errors in the ingest step due to missing data, file corruption, etc.
        disabled_layers: a list of layers to disable.
        group: limit dataset actions to this group.
        workers: number of workers to use.
    """
    dataset = Dataset(ds_path, disabled_layers=disabled_layers)
    logger.info("Running prepare step")
    apply_on_windows(
        PrepareHandler(force=False),
        dataset,
        workers=workers,
        group=group,
    )
    logger.info("Running ingest step")
    apply_on_windows(
        IngestHandler(ignore_errors=ignore_errors),
        dataset,
        workers=workers,
        group=group,
        use_initial_job=False,
    )
    logger.info("Running materialize step")
    apply_on_windows(
        MaterializeHandler(),
        dataset,
        workers=workers,
        group=group,
        use_initial_job=False,
    )


def run_model_predict(
    model_cfg_fname: str,
    ds_path: UPath,
    groups: list[str] = [],
    extra_args: list[str] = [],
) -> None:
    """Call rslearn model predict.

    Args:
        model_cfg_fname: the model configuration file.
        ds_path: the dataset root path.
        groups: the groups to predict.
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
        + (["--data.init_args.predict_config.groups", str(groups)] if groups else [])
        + extra_args,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )
