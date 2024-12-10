"""Materialize the dataset for the forest loss driver inference pipeline."""

from upath import UPath

from rslp.utils.rslearn import materialize_dataset

# Eventually this should be moved to the config file.
VISUALIZATION_ONLY_LAYERS = [
    "planet_post_0",
    "planet_post_1",
    "planet_post_2",
    "planet_pre_0",
    "planet_pre_1",
    "planet_pre_2",
]

GROUP = None

WORKERS = 8


def materialize_forest_loss_driver_dataset(
    ds_path: UPath,
    ignore_errors: bool = False,
    disabled_layers: list[str] = VISUALIZATION_ONLY_LAYERS,
    group: str | None = GROUP,
    workers: int = WORKERS,
) -> None:
    """Materialize the forest loss driver dataset.

    Wrapper function specific to the forest loss driver inference pipeline.

    Args:
        ds_path: the dataset root path,
        ignore_errors: whether to ignore errors, this allows us to ignore errors in the ingest step due to missing data, file corruption, etc.
            For this task, as we don't end up using all the ingested data, it's okay to ignore the occasional errors. (we select the best images later)
        disabled_layers: layers to disable for prepare/ingest/materialize,
        group: the group to use for prepare/ingest/materialize,
        workers: the number of workers to use for prepare/ingest/materialize,
    Outputs:
        Steps:
            prepare: items.json file for each layer
            ingest:
            materialize
    """
    # TODO: Add step to validate the directory has all the required files.
    materialize_dataset(
        ds_path,
        ignore_errors=ignore_errors,
        disabled_layers=disabled_layers,
        group=group,
        workers=workers,
    )
    # We should clearly log anytime a file is written as part of this process
