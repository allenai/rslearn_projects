"""Run EsPredictRunner inference pipeline."""

import hashlib
import logging
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path

import fsspec
from esrun.runner.local.fine_tune_runner import EsFineTuneRunner
from esrun.runner.local.predict_runner import EsPredictRunner
from esrun.shared.tools.logger import configure_logging
from upath import UPath

from rslp.log_utils import get_logger

logger = get_logger(__name__)


def get_local_checkpoint(checkpoint_path: UPath) -> Path:
    """Get a local path to the specified checkpoint file, caching it locally if needed.

    Args:
        checkpoint_path: a UPath to the checkpoint file.

    Returns:
        a local UPath, which is the same as checkpoint_path if it is already local, or
            points to a cached version in the system temporary directory.
    """
    # Cache the checkpoint if it isn't already local.
    if isinstance(checkpoint_path.fs, fsspec.implementations.local.LocalFileSystem):
        logger.info("using local checkpoint at %s", checkpoint_path)
        return Path(checkpoint_path)

    cache_id = hashlib.sha256(str(checkpoint_path).encode()).hexdigest()
    local_upath = (
        UPath(tempfile.gettempdir())
        / "rslearn_cache"
        / "esrun_checkpoints"
        / f"{cache_id}.ckpt"
    )

    if not local_upath.exists():
        logger.info("caching checkpoint from %s to %s", checkpoint_path, local_upath)
        local_upath.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("rb") as src, local_upath.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    logger.info("using cached checkpoint at %s", local_upath)
    return Path(local_upath)


def prepare_labeled_windows(project_path: Path, scratch_path: Path) -> None:
    """Run EsFineTuneRunner's prepare_windows pipeline."""
    logger.info("Loading EsFineTuneRunner")
    runner = EsFineTuneRunner(
        project_path=project_path,
        scratch_path=scratch_path,
    )
    logger.info("Running prepare_labeled_windows")
    runner.prepare_labeled_windows()


def esrun(config_path: Path, scratch_path: Path, checkpoint_path: str) -> None:
    """Run EsPredictRunner inference pipeline.

    Args:
        config_path: directory containing the model.yaml, partition_strategies.yaml,
            and postprocessing_strategies.yaml configuration files.
        scratch_path: directory to use for scratch space.
        checkpoint_path: path to the model checkpoint.
    """
    # Configure esrun logging before creating the runner
    configure_logging(log_level=logging.INFO)

    runner = EsPredictRunner(
        # ESRun does not work with relative path, so make sure to convert to absolute here.
        project_path=config_path.absolute(),
        scratch_path=scratch_path,
        checkpoint_path=get_local_checkpoint(UPath(checkpoint_path)),
    )
    logger.info("Partitioning...")
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
    checkpoint_path: str,
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

    # Configure esrun logging before creating the runner
    configure_logging(log_level=logging.INFO)

    runner = EsPredictRunner(
        # ESRun does not work with relative path, so make sure to convert to absolute here.
        project_path=config_path,
        scratch_path=scratch_path,
        checkpoint_path=get_local_checkpoint(UPath(checkpoint_path)),
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
