"""Run one task's share of a render job list, for array jobs such as Google Batch.

A driver writes the whole job list once to a JSON file; every task of the array job
reads it and runs every `task_count`-th job starting at its own index. Each job is an
idempotent render step that skips work its marker records, so a retried or preempted
task only redoes what it had not finished.

A list can instead be given as ordered stages, each finished before the next starts, for
work where a stage reads what the one before wrote (a pyramid's coarser levels). There is
no barrier across tasks, so a staged list runs as a single task.
"""

import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from upath import UPath

from rslp.large_scale_embeddings.render_pca import render_pca_targets_pipeline
from rslp.large_scale_embeddings.render_web_pca import (
    compose_web_shard_pipeline,
    render_web_pca_pipeline,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)

FUNCTIONS: dict[str, Callable[..., None]] = {
    "render_utm_targets": render_pca_targets_pipeline,
    "render_web": render_web_pca_pipeline,
    "compose_web": compose_web_shard_pipeline,
}


def write_batch_jobs(
    jobs_path: str,
    function: str,
    jobs: list[dict[str, Any]] | None = None,
    stages: list[list[dict[str, Any]]] | None = None,
) -> None:
    """Write a job list for `run_batch_slice`.

    Args:
        jobs_path: where to write the JSON file.
        function: a key of FUNCTIONS.
        jobs: keyword arguments for each call.
        stages: instead of jobs, ordered groups of calls run one group at a time.

    Raises:
        ValueError: for an unknown function name.
    """
    if function not in FUNCTIONS:
        raise ValueError(
            f"unknown function {function}; expected one of {list(FUNCTIONS)}"
        )
    if (jobs is None) == (stages is None):
        raise ValueError("give exactly one of jobs and stages")
    spec: dict[str, Any] = {"function": function}
    if stages is not None:
        spec["stages"] = stages
    else:
        spec["jobs"] = jobs
    with UPath(jobs_path).open("w") as f:
        json.dump(spec, f)
    n = sum(len(s) for s in stages) if stages is not None else len(jobs or [])
    logger.info("wrote %d %s job(s) to %s", n, function, jobs_path)


def run_batch_slice(
    jobs_path: str,
    task_index: int | None = None,
    task_count: int | None = None,
    workers: int = 4,
) -> None:
    """Run this task's share of a job list.

    Args:
        jobs_path: the JSON file written by `write_batch_jobs`.
        task_index: this task's index; defaults to $BATCH_TASK_INDEX.
        task_count: the number of tasks; defaults to $BATCH_TASK_COUNT.
        workers: jobs to run at once within the task.

    Raises:
        RuntimeError: if any job failed, after every other job has been tried, so the
            task is retried and the retry skips what finished.
    """
    if task_index is None:
        task_index = int(os.environ.get("BATCH_TASK_INDEX", "0"))
    if task_count is None:
        task_count = int(os.environ.get("BATCH_TASK_COUNT", "1"))
    with UPath(jobs_path).open() as f:
        spec = json.load(f)
    fn = FUNCTIONS[spec["function"]]
    if "stages" in spec:
        if task_count != 1:
            raise ValueError("a staged job list must run as a single task")
        stages = spec["stages"]
    else:
        stages = [spec["jobs"][task_index::task_count]]
    total = sum(len(s) for s in stages)
    logger.info(
        "task %d of %d: %d %s job(s) in %d stage(s)",
        task_index,
        task_count,
        total,
        spec["function"],
        len(stages),
    )

    failed = 0
    for number, mine in enumerate(stages, 1):
        stage_failed = _run_stage(fn, mine, task_index, workers)
        failed += stage_failed
        if stage_failed and number < len(stages):
            # A later stage reads this one's output, so stop rather than build on gaps.
            raise RuntimeError(
                f"{stage_failed} job(s) failed in stage {number} of {len(stages)}"
            )
    if failed:
        raise RuntimeError(f"{failed} of {total} job(s) failed in task {task_index}")


def _run_stage(
    fn: Callable[..., None], mine: list[dict[str, Any]], task_index: int, workers: int
) -> int:
    """Run one group of jobs concurrently; returns how many failed."""
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, **kwargs): kwargs for kwargs in mine}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception:
                failed += 1
                logger.exception("job failed: %s", futures[future])
            if i % 10 == 0 or i == len(mine):
                logger.info(
                    "task %d: %d/%d done, %d failed", task_index, i, len(mine), failed
                )
    return failed
