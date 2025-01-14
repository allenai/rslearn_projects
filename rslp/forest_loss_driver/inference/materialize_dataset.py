"""Materialize the dataset for the forest loss driver inference pipeline."""

from upath import UPath

from rslp.forest_loss_driver.inference.config import ForestLossDriverMaterializeArgs
from rslp.utils.rslearn import materialize_dataset


def materialize_forest_loss_driver_dataset(
    ds_root: UPath,
    materialize_pipeline_args: ForestLossDriverMaterializeArgs,
) -> None:
    """Materialize the forest loss driver dataset.

    Wrapper function specific to the forest loss driver inference pipeline.

    Args:
        ds_root: the path to the dataset.
        materialize_pipeline_args: arguments for materialize_dataset.
    """
    materialize_dataset(ds_root, materialize_pipeline_args)
