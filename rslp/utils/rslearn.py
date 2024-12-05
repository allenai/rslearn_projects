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
    ds_path: UPath,
    group: str | None = None,
    workers: int = 32,
    initial_prepare_job: bool = False,
    prepare_workers: int | None = None,
    ingest_workers: int | None = None,
    materialize_workers: int | None = None,
) -> None:
    """Materialize the specified dataset by running prepare/ingest/materialize.

    Args:
        ds_path: the dataset root.
        group: limit dataset actions to this group.
        workers: number of workers to use.
        initial_prepare_job: set True if initial job during prepare is needed, e.g. if
            the data source creates an index first.
        prepare_workers: use this many workers for prepare stage (overrides workers
            argument)
        ingest_workers: use this many workers for ingest stage (overrides workers
            argument)
        materialize_workers: use this many workers for materialize stage (overrides
            workers argument)
    """
    dataset = Dataset(ds_path)

    if prepare_workers is None:
        prepare_workers = workers
    if ingest_workers is None:
        ingest_workers = workers
    if materialize_workers is None:
        materialize_workers = workers

    apply_on_windows(
        PrepareHandler(force=False),
        dataset,
        workers=prepare_workers,
        group=group,
        use_initial_job=initial_prepare_job,
    )
    apply_on_windows(
        IngestHandler(),
        dataset,
        workers=ingest_workers,
        group=group,
        use_initial_job=False,
    )
    apply_on_windows(
        MaterializeHandler(),
        dataset,
        workers=materialize_workers,
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
