"""Forest loss driver prediction pipeline."""

import os
from pathlib import Path

from rslp.log_utils import get_logger

from .inference import (
    PredictPipelineConfig,
    extract_alerts_pipeline,
    forest_loss_driver_model_predict,
    materialize_forest_loss_driver_dataset,
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

    def _validate_required_env_vars(
        self, required_env_vars: list[str], optional_env_vars: list[str]
    ) -> None:
        """Validate the required environment variables."""
        missing_vars = [var for var in required_env_vars if var not in os.environ]
        if missing_vars:
            missing_vars_str = ", ".join(missing_vars)
            raise OSError(
                f"The following required environment variables are missing: {missing_vars_str}"
            )
        missing_optional_vars = [
            var for var in optional_env_vars if var not in os.environ
        ]
        if missing_optional_vars:
            missing_optional_vars_str = ", ".join(missing_optional_vars)
            logger.warning(
                f"The following optional environment variables are missing: {missing_optional_vars_str}"
            )
        if "INDEX_CACHE_DIR" in os.environ:
            cache_dir = os.environ["INDEX_CACHE_DIR"]
            if not any(
                cache_dir.startswith(prefix) for prefix in ["gs://", "s3://", "file://"]
            ):
                logger.warning(
                    f"INDEX_CACHE_DIR '{cache_dir}' does not specify filesystem - "
                    "will be treated as relative path"
                )

    def extract_dataset(self) -> None:
        """Extract the dataset."""
        REQUIRED_ENV_VARS: list[str] = []
        OPTIONAL_ENV_VARS: list[str] = [
            "INDEX_CACHE_DIR",
            "TILE_STORE_ROOT_DIR",
            "PL_API_KEY",
        ]
        # TODO: make sure the env variables are parsed in the config json
        self._validate_required_env_vars(REQUIRED_ENV_VARS, OPTIONAL_ENV_VARS)
        extract_alerts_pipeline(
            self.pred_config.path,
            self.pred_config.extract_alerts_args,
        )

        materialize_forest_loss_driver_dataset(
            self.pred_config.path,
            self.pred_config.materialize_pipeline_args,
        )

        select_least_cloudy_images_pipeline(
            self.pred_config.path,
            self.pred_config.select_least_cloudy_images_args,
        )

    def run_model_predict(self) -> None:
        """Run the model predict."""
        REQUIRED_ENV_VARS: list[str] = ["RSLP_PREFIX"]
        OPTIONAL_ENV_VARS: list[str] = []
        self._validate_required_env_vars(REQUIRED_ENV_VARS, OPTIONAL_ENV_VARS)
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
