"""Run EsPredictRunner inference pipeline."""

from pathlib import Path

from esrun.runner.local.predict_runner import EsPredictRunner
from esrun.shared.models.task_results import InferenceResultsDataType

from rslp.log_utils import get_logger

logger = get_logger(__name__)


def esrun(
    config_path: Path, scratch_path: Path, data_type: InferenceResultsDataType
) -> None:
    """Run EsPredictRunner inference pipeline.

    Args:
        config_path: directory containing the model.yaml, partition_strategies.yaml,
            and postprocessing_strategies.yaml configuration files.
        scratch_path: directory to use for scratch space.
        data_type: the output data type of the model.
    """
    runner = EsPredictRunner(
        project_path=config_path,
        scratch_path=scratch_path,
        inference_results_data_type=data_type,
    )
    partitions = runner.partition()
    logger.info(f"Got {len(partitions)} partitions")

    for partition_id in partitions:
        logger.info(f"Building dataset for partition {partition_id}")
        runner.build_dataset(partition_id)
        logger.info(f"Running inference for partition {partition_id}")
        runner.run_inference(partition_id)
        logger.info(f"Postprocessing for partition {partition_id}")
        runner.postprocess(partition_id)

    logger.info("Combining across partitions")
    runner.combine(partitions)
