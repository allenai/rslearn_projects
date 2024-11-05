"""Forest loss driver prediction pipeline."""

from pathlib import Path

from rslp.forest_loss_driver.inference.best_image_selector import (
    select_best_images_pipeline,
)
from rslp.forest_loss_driver.inference.config import PredictPipelineConfig
from rslp.forest_loss_driver.inference.extract_alerts import extract_alerts_pipeline
from rslp.forest_loss_driver.inference.materialize_dataset import (
    materialize_forest_loss_driver_dataset,
)
from rslp.forest_loss_driver.inference.model_predict import (
    forest_loss_driver_model_predict,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)

GCS_FILENAMES = [
    "070W_10S_060W_00N.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "070W_20S_060W_10S.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "080W_10S_070W_00N.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "080W_20S_070W_10S.tif",  # What are these files I presume the s2 tiffs of the quarter?
]

WINDOW_SIZE = 128

# PIPELINE CONFIG USED FOR INFERENCE
PREDICT_PIPELINE_CONFIG_PATH = str(
    Path(__file__).parent
    / "inference"
    / "config"
    / "forest_loss_driver_predict_pipeline_config.yaml"
)


def load_predict_pipeline_config() -> PredictPipelineConfig:
    """Load the predict pipeline configuration."""
    return PredictPipelineConfig.from_yaml(PREDICT_PIPELINE_CONFIG_PATH)


# TODO: We need to add an environment variable validation step here for the entire pipeline
def forest_loss_driver_prediction_pipeline(
    pred_config: PredictPipelineConfig,
) -> None:
    """Run the prediction pipeline.

    Currently this is just for populating the initial rslearn dataset based on GLAD
    forest loss events in Peru over the last year.

    So need to prepare/ingest/materialize the dataset afterward, and run the
    select_best_images_pipeline. Then apply the model.

    Args:
        pred_config: the pipeline configuration

    Outputs:
        None
        a folder with outputs and the prediciton.json file.
    """
    # Move thsi for loop to the pipeline?
    for filename in pred_config.gcs_tiff_filenames:
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

    forest_loss_driver_model_predict(pred_config.model_cfg_fname, pred_config.path)


def predict_pipeline_main() -> None:
    """Run the predict pipeline."""
    pred_config = load_predict_pipeline_config()
    forest_loss_driver_prediction_pipeline(pred_config)
