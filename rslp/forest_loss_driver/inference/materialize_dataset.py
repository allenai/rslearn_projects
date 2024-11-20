"""Materialize the dataset for the forest loss driver inference pipeline."""

from rslp.utils.rslearn import materialize_dataset
from upath import UPath

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

WORKERS = 32


def materialize_forest_loss_driver_dataset(
    ds_path: UPath,
    disabled_layers: list[str] = VISUALIZATION_ONLY_LAYERS,
    group: str | None = GROUP,
    workers: int = WORKERS,
) -> None:
    """Materialize the forest loss driver dataset.

    Wrapper function specific to the forest loss driver inference pipeline.
    Args:
        ds_path: the dataset root path,

    Outputs:
        Steps:
            prepare: items.json file for each layer
            ingest:
            materialize
    """
    # TODO: Add step to validate the directory has all the required files.
    materialize_dataset(
        ds_path,
        disabled_layers=disabled_layers,
        group=group,
        workers=workers,
    )
    # We should clearly log anytime a file is written as part of this process