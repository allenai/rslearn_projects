"""Forest loss driver prediction pipeline."""

import os
from pathlib import Path

from rslp.log_utils import get_logger

from .inference import (
    PredictPipelineConfig,
    extract_alerts_pipeline,
    forest_loss_driver_model_predict,
    materialize_forest_loss_driver_dataset,
    select_best_images_pipeline,
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
PREDICT_PIPELINE_CONFIG_PATH = str(
    Path(__file__).parent
    / "inference"
    / "config"
    / "forest_loss_driver_predict_pipeline_config.yaml"
)


# TODO: Add Data vlaidation steps after each step to check to ensure the directory structure is correct
class ForestLossDriverPredictionPipeline:
    """Forest loss driver prediction pipeline."""

    PREDICT_PIPELINE_CONFIG_PATH = str(
        Path(__file__).parent
        / "inference"
        / "config"
        / "forest_loss_driver_predict_pipeline_config.yaml"
    )

    def __init__(
        self,
        pred_config_path: str | None = None,
        pred_pipeline_config: PredictPipelineConfig | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            pred_config_path: the path to the prediction pipeline config,
            pred_pipeline_config: the prediction pipeline config,

        We always default to the config at the path specified in the class
        """
        if pred_config_path is None:
            pred_config_path = self.PREDICT_PIPELINE_CONFIG_PATH
        if pred_pipeline_config is None:
            pred_pipeline_config = PredictPipelineConfig.from_yaml(pred_config_path)
        self.pred_config = pred_pipeline_config

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
        # check that we have PL_API_KEY if we are looking for planet images

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
        for filename in self.pred_config.gcs_tiff_filenames:
            extract_alerts_pipeline(self.pred_config, filename)

        materialize_forest_loss_driver_dataset(
            self.pred_config.path,
            ignore_errors=self.pred_config.ignore_errors,
            disabled_layers=self.pred_config.disabled_layers,
            group=self.pred_config.group,
            workers=self.pred_config.workers,
        )

        select_best_images_pipeline(
            self.pred_config.path,
            workers=self.pred_config.workers,
        )

    def run_model_predict(self) -> None:
        """Run the model predict."""
        REQUIRED_ENV_VARS: list[str] = ["RSLP_PREFIX"]
        OPTIONAL_ENV_VARS: list[str] = []
        self._validate_required_env_vars(REQUIRED_ENV_VARS, OPTIONAL_ENV_VARS)
        # TODO: Add some validation that the extract dataset step is done by checking the dataset bucket
        # TODO: This may have unneeded levels of wrapping and abstraction
        logger.info(f"running model predict with config: {self.pred_config}")
        forest_loss_driver_model_predict(
            self.pred_config.model_cfg_fname,
            self.pred_config.path,
            self.pred_config.model_data_load_workers,
        )


def extract_dataset_main() -> None:
    """Extract the dataset."""
    pipeline = ForestLossDriverPredictionPipeline()
    pipeline.extract_dataset()


def run_model_predict_main() -> None:
    """Run the model predict."""
    pipeline = ForestLossDriverPredictionPipeline()
    pipeline.run_model_predict()
