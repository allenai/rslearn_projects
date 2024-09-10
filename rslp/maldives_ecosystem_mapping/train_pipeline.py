"""Model training pipeline for Maldives ecosystem mapping project."""

from rslp.config import BaseTrainPipelineConfig
from rslp.launch_beaker import launch_job


def maxar_train_pipeline(config: BaseTrainPipelineConfig):
    """Run the training pipeline.

    Args:
        config: the model training config.
    """
    launch_job("data/maldives_ecosystem_mapping/config.yaml")


def sentinel2_train_pipeline(config: BaseTrainPipelineConfig):
    """Run the training pipeline.

    Args:
        config: the model training config.
    """
    launch_job("data/maldives_ecosystem_mapping/config_sentinel2.yaml")
