"""Config objects."""


class BaseDataPipelineConfig:
    """Base configuration for a data pipeline."""

    def __init__(self, ds_root: str, workers: int = 1):
        """Create a new BaseDataPipelineConfig.

        Args:
            ds_root: the dataset root directory.
            workers: the number of workers.
        """
        self.ds_root = ds_root
        self.workers = workers


class BaseTrainPipelineConfig:
    """Base configuration for a model training pipeline."""

    def __init__(self):
        """Create a new BaseTrainPipelineConfig."""
        pass
