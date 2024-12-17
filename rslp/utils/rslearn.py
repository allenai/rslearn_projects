"""Utilities for using rslearn datasets and models."""

from dataclasses import dataclass

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


# TODO: add an args model in rslearn instead of here
@dataclass
class PrepareIngestMaterializeApplyWindowsArgs:
    """Arguments for prepare/ingest/materialize/apply_on_windows."""

    workers: int = 0
    batch_size: int = 1
    use_initial_job: bool = False
    jobs_per_process: int | None = None
    group: str | None = None


def materialize_dataset(
    ds_path: UPath,
    ignore_errors: bool = False,
    disabled_layers: list[str] = [],
    apply_args: PrepareIngestMaterializeApplyWindowsArgs = PrepareIngestMaterializeApplyWindowsArgs(),
) -> None:
    """Materialize the specified dataset by running prepare/ingest/materialize.

    Args:
        ds_path: the dataset root.
        ignore_errors: whether to ignore errors, this allows us to ignore errors in the ingest step due to missing data, file corruption, etc.
        disabled_layers: a list of layers to disable.
        apply_args: arguments for prepare/ingest/materialize/apply_on_windows.
    """
    dataset = Dataset(ds_path, disabled_layers=disabled_layers)
    logger.debug(f"apply_args: {apply_args}")
    logger.info("Running prepare step")
    apply_on_windows(
        PrepareHandler(force=False),
        dataset,
        **apply_args,
    )
    logger.info("Running ingest step")
    apply_on_windows(
        IngestHandler(ignore_errors=ignore_errors),
        dataset,
        **apply_args,
    )
    logger.info("Running materialize step")
    apply_on_windows(
        MaterializeHandler(ignore_errors=ignore_errors),
        dataset,
        **apply_args,
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
