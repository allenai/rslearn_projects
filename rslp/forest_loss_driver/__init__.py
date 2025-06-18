"""Forest loss driver classification project."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from upath import UPath

from .copy import copy_dataset
from .extract_dataset import (
    ExtractAlertsArgs,
    InferenceLayerMaterializeArgs,
    SelectLeastCloudyImagesArgs,
    VisLayerMaterializeArgs,
    extract_alerts,
    extract_dataset,
    select_least_cloudy_images_pipeline,
)
from .predict_pipeline import predict_pipeline
from .webapp.index_windows import index_windows
from .webapp.make_tiles import MakeTilesArgs, make_tiles


@dataclass
class IntegratedConfig:
    """Integrated inference config for forest loss driver classification.

    The arguments are combined so they can be passed together to the integrated
    pipeline, which runs the steps together in one pipeline.

    Args:
        extract_alerts_args: the arguments for the extract_alerts step.
        inference_materialize_args: arguments for materializing inference layers.
        vis_materialize_args: arguments for materializing visualization layers.
        select_least_cloudy_images_args: the arguments for the select_least_cloudy_images step.
        make_tiles_args: the arguments for the make_tiles step. Note that the ds_root
            here will be overwritten.
    """

    extract_alerts_args: ExtractAlertsArgs = field(default_factory=ExtractAlertsArgs)
    inference_materialize_args: InferenceLayerMaterializeArgs = field(
        default_factory=InferenceLayerMaterializeArgs
    )
    vis_materialize_args: VisLayerMaterializeArgs = field(
        default_factory=VisLayerMaterializeArgs
    )
    select_least_cloudy_images_args: SelectLeastCloudyImagesArgs = field(
        default_factory=SelectLeastCloudyImagesArgs
    )
    make_tiles_args: MakeTilesArgs | None = None


def _get_most_recent_friday() -> datetime:
    """Get the most recent Friday."""
    now = datetime.now()
    friday = now - timedelta(days=(now.weekday() - 4) % 7)
    return friday


def integrated_pipeline(integrated_config: IntegratedConfig) -> None:
    """Integrated pipeline that runs all stages for forest loss driver model.

    1. Extract dataset on Weka.
    2. Apply model.
    3. Index windows.
    4. Upload relevant parts of the dataset from Weka to GCS.
    5. Make tiles (in public GCS bucket).

    This pipeline is Ai2-specific due to the use of Weka and GCS, so other uses should
    call the individual steps directly.

    Args:
        integrated_config: the integrated configuration for all inference pipeline
            steps.
    """
    # Determine dataset path to use based on the current date.
    friday = _get_most_recent_friday()
    dated_dataset_name = f"dataset_{friday.strftime('%Y%m%d')}"
    # Dataset root for first three steps.
    weka_ds_root = f"/dfive-default/rslearn-eai/datasets/forest_loss_driver/prediction/{dated_dataset_name}"
    # Dataset root that web app will read the images and other data from.
    gcs_ds_root = (
        f"gs://rslearn-eai/datasets/forest_loss_driver/prediction/{dated_dataset_name}"
    )

    # Override ds_root in MakeTilesArgs.
    ds_root = UPath(weka_ds_root)
    make_tiles_args = integrated_config.make_tiles_args
    if make_tiles_args is None:
        make_tiles_args = MakeTilesArgs(ds_root=ds_root)
    else:
        # Override the ds_root to match what we used for dataset extraction.
        make_tiles_args.ds_root = ds_root

    extract_dataset(
        ds_path=ds_root,
        extract_alerts_args=integrated_config.extract_alerts_args,
        select_least_cloudy_images_args=integrated_config.select_least_cloudy_images_args,
        inference_materialize_args=integrated_config.inference_materialize_args,
        vis_materialize_args=integrated_config.vis_materialize_args,
    )
    predict_pipeline(ds_root)
    index_windows(ds_root)
    copy_dataset(UPath(weka_ds_root), UPath(gcs_ds_root))
    make_tiles(make_tiles_args)


workflows = {
    "extract_alerts": extract_alerts,
    "select_least_cloudy_images": select_least_cloudy_images_pipeline,
    # extract_dataset combines extract_alerts, dataset materialization, and
    # select_least_cloudy_images.
    "extract_dataset": extract_dataset,
    "predict": predict_pipeline,
    "index_windows": index_windows,
    "make_tiles": make_tiles,
    "integrated_pipeline": integrated_pipeline,
}
