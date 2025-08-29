"""Run EsPredictRunner inference pipeline."""

from enum import StrEnum
from pathlib import Path

from esrun.runner.local.predict_runner import EsPredictRunner
from rslearn.utils.fsspec import get_upath_local
from upath import UPath

from rslp.log_utils import get_logger

logger = get_logger(__name__)


def esrun(config_path: Path, scratch_path: Path, checkpoint_path: str) -> None:
    """Run EsPredictRunner inference pipeline.

    Args:
        config_path: directory containing the model.yaml, partition_strategies.yaml,
            and postprocessing_strategies.yaml configuration files.
        scratch_path: directory to use for scratch space.
        checkpoint_path: path to the model checkpoint.
    """
    # Copy checkpoint to local filesystem if it is not local.
    with get_upath_local(UPath(checkpoint_path)) as local_checkpoint_path:
        runner = EsPredictRunner(
            # ESRun does not work with relative path, so make sure to convert to absolute here.
            project_path=config_path.absolute(),
            scratch_path=scratch_path,
            checkpoint_path=local_checkpoint_path,
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


class EsrunStage(StrEnum):
    """The stage of esrun pipeline to run.

    We always run the partition stage so that is not an option here.
    """

    BUILD_DATASET = "build_dataset"
    RUN_INFERENCE = "run_inference"
    POSTPROCESS = "postprocess"
    COMBINE = "combine"


def one_stage(
    config_path: Path,
    scratch_path: Path,
    checkpoint_path: Path,
    stage: EsrunStage,
    partition_id: str | None = None,
) -> None:
    """Run EsPredictRunner inference pipeline.

    Args:
        config_path: see esrun.
        scratch_path: see esrun.
        checkpoint_path: see esrun.
        stage: which stage to run.
        partition_id: the partition to run the stage for. If not set, we run the stage
            for all partitions, except COMBINE, which happens across partitions.
    """
    if stage == EsrunStage.COMBINE and partition_id is not None:
        raise ValueError("partition_id cannot be set for COMBINE stage")

    # Copy checkpoint to local filesystem if it is not local.
    with get_upath_local(checkpoint_path) as local_checkpoint_path:
        runner = EsPredictRunner(
            # ESRun does not work with relative path, so make sure to convert to absolute here.
            project_path=config_path,
            scratch_path=scratch_path,
            checkpoint_path=local_checkpoint_path,
        )
        partitions = runner.partition()

        if stage in [
            EsrunStage.BUILD_DATASET,
            EsrunStage.RUN_INFERENCE,
            EsrunStage.POSTPROCESS,
        ]:
            fn = None
            if stage == EsrunStage.BUILD_DATASET:
                fn = runner.build_dataset
            elif stage == EsrunStage.RUN_INFERENCE:
                fn = runner.run_inference
            elif stage == EsrunStage.POSTPROCESS:
                fn = runner.postprocess
            else:
                assert False

            if partition_id is not None:
                if partition_id not in partitions:
                    raise ValueError(f"partition {partition_id} does not exist")
                fn(partition_id)
            else:
                for partition_id in partitions:
                    fn(partition_id)

        elif stage == EsrunStage.COMBINE:
            runner.combine(partitions)
