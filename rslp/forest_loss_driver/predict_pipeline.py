"""Forest loss driver prediction pipeline."""

from rslp.forest_loss_driver.inference import (
    extract_alerts_pipeline,
    forest_loss_driver_model_predict,
    materialize_forest_loss_driver_dataset,
    select_best_images_pipeline,
)
from rslp.forest_loss_driver.inference.config import PredictPipelineConfig
from rslp.log_utils import get_logger

logger = get_logger(__name__)

GCS_FILENAMES = [
    "070W_10S_060W_00N.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "070W_20S_060W_10S.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "080W_10S_070W_00N.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "080W_20S_070W_10S.tif",  # What are these files I presume the s2 tiffs of the quarter?
]

# ENVIRONMENT VARIABLES we need here
WINDOW_SIZE = 128
# os.environ["INFERENCE_DATASET_CONFIG"] = str(
#     Path(__file__).resolve().parents[3] / "data" / "forest_loss_driver" / "config.json"
# )
# os.environ["INDEX_CACHE_DIR"] = (
#     "/Users/henryh/Desktop/eai-repos/rslearn_projects/data/henryh/rslearn_cache/"
# )
# os.environ["TILE_STORE_ROOT_DIR"] = (
#     "/Users/henryh/Desktop/eai-repos/rslearn_projects/data/henryh/tile_store "
# )
# os.environ["RSLP_PREFIX"] = "gs://rslearn-eai"


# TODO: All important configuration should be transparently passed in via the PredictPipelineConfig
# TODO: We need to add an environment variable validation step here for the entire pipeline
# This is the main function that should be called to run the prediction pipeline. the alerts stuff likely should be in a different module
def predict_pipeline(
    pred_config: PredictPipelineConfig,
    model_cfg_fname: str,
    gcs_tiff_filenames: list[str],
) -> None:
    """Run the prediction pipeline.

    Currently this is just for populating the initial rslearn dataset based on GLAD
    forest loss events in Peru over the last year.

    So need to prepare/ingest/materialize the dataset afterward, and run the
    select_best_images_pipeline. Then apply the model.

    Args:
        pred_config: the pipeline configuration
        model_cfg_fname: the model configuration file name
        gcs_tiff_filenames: the list of GCS TIFF filenames

    Outputs:
        None
        a folder with outputs and the prediciton.json file.
    """
    # Move thsi for loop to the pipeline?
    for filename in gcs_tiff_filenames:
        extract_alerts_pipeline(pred_config, filename)

    materialize_forest_loss_driver_dataset(
        pred_config.path,
        disabled_layers=pred_config.disabled_layers,
        group=pred_config.group,
        workers=pred_config.workers,
    )

    select_best_images_pipeline(
        pred_config.path,
        workers=pred_config.workers,
    )

    forest_loss_driver_model_predict(model_cfg_fname, pred_config.path)
