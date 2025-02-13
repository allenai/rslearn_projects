"""Forest loss driver classification project."""

from datetime import datetime, timedelta

from .copy import copy_dataset
from .predict_pipeline import (
    PredictPipelineConfig,
    extract_dataset_main,
    run_model_predict_main,
)
from .webapp.index_windows import index_windows
from .webapp.make_tiles import MakeTilesArgs, make_tiles


def _get_most_recent_friday() -> datetime:
    """Get the most recent Friday."""
    now = datetime.now()
    friday = now - timedelta(days=(now.weekday() - 4) % 7)
    return friday


def integrated_pipeline(
    pred_pipeline_config: PredictPipelineConfig,
    make_tiles_args: MakeTilesArgs | None = None,
) -> None:
    """Integrated pipeline that runs all stages for forest loss driver model.

    1. Extract dataset on Weka.
    2. Apply model.
    3. Index windows.
    4. Upload relevant parts of the dataset from Weka to GCS.
    5. Make tiles (in public GCS bucket).

    This pipeline is Ai2-specific due to the use of Weka and GCS, so other uses should
    call the individual steps directly.

    Args:
        pred_pipeline_config: the PredictPipelineConfig used by the dataset extraction
            and model prediction steps. The ds_root will be overwritten.
        make_tiles_args: arguments for the make tiles step. The ds_root will be
            overwritten.
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

    pred_pipeline_config.ds_root = weka_ds_root
    if make_tiles_args is None:
        make_tiles_args = MakeTilesArgs(ds_root=pred_pipeline_config.ds_root)
    else:
        # Override the ds_root to match what we used for dataset extraction.
        make_tiles_args.ds_root = pred_pipeline_config.ds_root

    extract_dataset_main(pred_pipeline_config)
    run_model_predict_main(pred_pipeline_config)
    index_windows(pred_pipeline_config.ds_root)
    copy_dataset(weka_ds_root, gcs_ds_root)
    make_tiles(make_tiles_args)


workflows = {
    "extract_dataset": extract_dataset_main,
    "predict": run_model_predict_main,
    "index_windows": index_windows,
    "make_tiles": make_tiles,
    "integrated_pipeline": integrated_pipeline,
}
