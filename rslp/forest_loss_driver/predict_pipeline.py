"""Forest loss driver prediction pipeline."""

from pathlib import Path

from rslp.log_utils import get_logger
from rslp.utils.rslearn import materialize_dataset

from .inference import (
    PredictPipelineConfig,
    forest_loss_driver_model_predict,
    select_least_cloudy_images_pipeline,
)

logger = get_logger(__name__)

GCS_FILENAMES = [
    "070W_10S_060W_00N.tif",
    "070W_20S_060W_10S.tif",
    "080W_10S_070W_00N.tif",
    "080W_20S_070W_10S.tif",
]

WINDOW_SIZE = 128

# PIPELINE CONFIG USED FOR INFERENCE
DEFAULT_PREDICT_PIPELINE_CONFIG_PATH = str(
    Path(__file__).parent
    / "inference"
    / "config"
    / "forest_loss_driver_predict_pipeline_config.yaml"
)


# TODO: Add Data vlaidation steps after each step to check to ensure the directory structure is correct
class ForestLossDriverPredictionPipeline:
    """Forest loss driver prediction pipeline."""

    def __init__(self, pred_pipeline_config: PredictPipelineConfig) -> None:
        """Initialize the pipeline.

        Args:
            pred_pipeline_config: the prediction pipeline config,

        """
        self.pred_config = pred_pipeline_config
        logger.info(f"Initialized pipeline with config: {self.pred_config}")

    def extract_dataset(self) -> None:
        """Extract the dataset."""
        # extract_alerts_pipeline(
        #    self.pred_config.path,
        #    self.pred_config.extract_alerts_args,
        # )

        materialize_dataset(
            self.pred_config.path,
            self.pred_config.materialize_pipeline_args,
        )

        select_least_cloudy_images_pipeline(
            self.pred_config.path,
            self.pred_config.select_least_cloudy_images_args,
        )

    def run_model_predict(self) -> None:
        """Run the model predict."""
        # TODO: Add some validation that the extract dataset step is done by checking the dataset bucket
        logger.info(f"running model predict with config: {self.pred_config}")
        forest_loss_driver_model_predict(
            self.pred_config.path,
            self.pred_config.model_predict_args,
        )


def extract_dataset_main(pred_pipeline_config: PredictPipelineConfig) -> None:
    """Extract the dataset."""
    pipeline = ForestLossDriverPredictionPipeline(
        pred_pipeline_config=pred_pipeline_config
    )
    pipeline.extract_dataset()


def run_model_predict_main(pred_pipeline_config: PredictPipelineConfig) -> None:
    """Run the model predict."""
    pipeline = ForestLossDriverPredictionPipeline(
        pred_pipeline_config=pred_pipeline_config
    )
    pipeline.run_model_predict()
