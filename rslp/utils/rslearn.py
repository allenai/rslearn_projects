"""Utilities for using rslearn datasets and models."""

from dataclasses import asdict, dataclass, field
from datetime import timedelta

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


@dataclass
class ApplyWindowsArgs:
    """Arguments for apply_on_windows."""

    workers: int = 0
    batch_size: int = 1
    use_initial_job: bool = False  # TODO: mathc no use_initial_job
    jobs_per_process: int | None = None
    group: str | None = None
    names: list[str] | None = None


@dataclass
class PrepareArgs:
    """Arguments for prepare operation."""

    apply_windows_args: ApplyWindowsArgs = field(
        default_factory=lambda: ApplyWindowsArgs()
    )
    retry_max_attempts: int = 0
    retry_backoff: timedelta = timedelta(minutes=1)


@dataclass
class IngestArgs:
    """Arguments for ingest operation."""

    ignore_errors: bool = False
    apply_windows_args: ApplyWindowsArgs = field(
        default_factory=lambda: ApplyWindowsArgs()
    )
    retry_max_attempts: int = 0
    retry_backoff: timedelta = timedelta(minutes=1)


@dataclass
class MaterializeArgs:
    """Arguments for materialize operation."""

    ignore_errors: bool = False
    apply_windows_args: ApplyWindowsArgs = field(
        default_factory=lambda: ApplyWindowsArgs()
    )
    retry_max_attempts: int = 0
    retry_backoff: timedelta = timedelta(minutes=1)


@dataclass
class MaterializePipelineArgs:
    """Arguments for materialize_dataset."""

    disabled_layers: list[str]
    prepare_args: PrepareArgs
    ingest_args: IngestArgs
    materialize_args: MaterializeArgs


def materialize_dataset(
    ds_root: UPath,
    materialize_pipeline_args: MaterializePipelineArgs,
) -> None:
    """Materialize the specified dataset by running prepare/ingest/materialize.

    Args:
        ds_root: the root path to the dataset.
        materialize_pipeline_args: arguments for materialize_dataset.
    """
    dataset = Dataset(
        ds_root,
        disabled_layers=materialize_pipeline_args.disabled_layers,
    )
    logger.debug("materialize_pipeline_args: %s", materialize_pipeline_args)
    logger.info("Running prepare step")
    apply_on_windows(
        PrepareHandler(
            force=False,
            retry_max_attempts=materialize_pipeline_args.prepare_args.retry_max_attempts,
            retry_backoff=materialize_pipeline_args.prepare_args.retry_backoff,
        ),
        dataset,
        **asdict(materialize_pipeline_args.prepare_args.apply_windows_args),
    )
    logger.info("Running ingest step")
    apply_on_windows(
        IngestHandler(
            ignore_errors=materialize_pipeline_args.ingest_args.ignore_errors,
            retry_max_attempts=materialize_pipeline_args.ingest_args.retry_max_attempts,
            retry_backoff=materialize_pipeline_args.ingest_args.retry_backoff,
        ),
        dataset,
        **asdict(materialize_pipeline_args.ingest_args.apply_windows_args),
    )
    logger.info("Running materialize step")
    apply_on_windows(
        MaterializeHandler(
            ignore_errors=materialize_pipeline_args.materialize_args.ignore_errors,
            retry_max_attempts=materialize_pipeline_args.materialize_args.retry_max_attempts,
            retry_backoff=materialize_pipeline_args.materialize_args.retry_backoff,
        ),
        dataset,
        **asdict(materialize_pipeline_args.materialize_args.apply_windows_args),
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
        # We allow environment var overrides for flexibility in deployment.
        parser_kwargs={"default_env": True},
    )
