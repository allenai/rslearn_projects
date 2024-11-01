"""Forest loss driver prediction pipeline."""

from rslp.forest_loss_driver.inference.config import PredictPipelineConfig
from rslp.log_utils import get_logger

logger = get_logger(__name__)

GCS_FILENAMES = [
    "070W_10S_060W_00N.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "070W_20S_060W_10S.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "080W_10S_070W_00N.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "080W_20S_070W_10S.tif",  # What are these files I presume the s2 tiffs of the quarter?
]

# How big the rslearn windows should be.
WINDOW_SIZE = 128


# TODO: We need to add an environment variable validation step here for the entire pipeline
# This is the main function that should be called to run the prediction pipeline. the alerts stuff likely should be in a different module
def predict_pipeline(pred_config: PredictPipelineConfig) -> None:
    """Run the prediction pipeline.

    Currently this is just for populating the initial rslearn dataset based on GLAD
    forest loss events in Peru over the last year.

    So need to prepare/ingest/materialize the dataset afterward, and run the
    select_best_images_pipeline. Then apply the model.

    Args:
        pred_config: the pipeline configuration
    """
    pass
