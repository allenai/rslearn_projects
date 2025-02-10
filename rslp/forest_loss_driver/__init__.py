"""Forest loss driver classification project."""

from .predict_pipeline import (
    PredictPipelineConfig,
    extract_dataset_main,
    run_model_predict_main,
)
from .webapp.index_windows import index_windows
from .webapp.make_tiles import MakeTilesArgs, make_tiles


def integrated_pipeline(
    pred_pipeline_config: PredictPipelineConfig,
    make_tiles_args: MakeTilesArgs | None = None,
) -> None:
    """Integrated pipeline that runs all stages for forest loss driver model.

    1. Extract dataset.
    2. Apply model.
    3. Index windows.
    4. Make tiles.

    Args:
        pred_pipeline_config: the PredictPipelineConfig used by the dataset extraction
            and model prediction steps.
        make_tiles_args: arguments for the make tiles step. The ds_root will be
            overwritten by the one selected by pred_pipeline_config.
    """
    extract_dataset_main(pred_pipeline_config)
    run_model_predict_main(pred_pipeline_config)
    index_windows(pred_pipeline_config.ds_root)

    if make_tiles_args is None:
        make_tiles_args = MakeTilesArgs(ds_root=pred_pipeline_config.ds_root)
    else:
        # Override the ds_root to match what we used for dataset extraction.
        make_tiles_args.ds_root = pred_pipeline_config.ds_root
    make_tiles(make_tiles_args)


workflows = {
    "extract_dataset": extract_dataset_main,
    "predict": run_model_predict_main,
    "index_windows": index_windows,
    "make_tiles": make_tiles,
    "integrated_pipeline": integrated_pipeline,
}
