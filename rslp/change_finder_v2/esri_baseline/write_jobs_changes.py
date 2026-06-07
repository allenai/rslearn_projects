"""Enumerate tiles with stacked GeoTIFFs and write Beaker jobs to compute changes.

Lists the stack directory, filters to tiles that don't yet have transition
outputs, batches them, and writes jobs to a Beaker queue for processing by
``compute_changes_batch``.

Usage::

    python -m rslp.main change_finder_v2 esri_write_jobs_changes \
        --stack_path /weka/.../esri_lc_stacks/ \
        --transitions_path /weka/.../esri_lc_transitions/ \
        --change_count_path /weka/.../esri_lc_change_count/ \
        --queue_name yawenzzzz/esri-changes-queue
"""

from __future__ import annotations

import json
import logging
import os
import random
import re

from upath import UPath

import rslp.common.worker

logger = logging.getLogger(__name__)

_TILE_RE = re.compile(r"^(.+)_lc_stack\.tif$")


def write_jobs_changes(
    stack_path: str,
    transitions_path: str,
    change_count_path: str,
    queue_name: str,
    batch_size: int = 10,
    count: int | None = None,
) -> None:
    """Enumerate tiles and write change-computation jobs to a Beaker queue.

    Args:
        stack_path: directory containing ``{tile_id}_lc_stack.tif`` files.
        transitions_path: directory where transition rasters are written.
        change_count_path: directory where change count rasters are written.
        queue_name: the Beaker queue to write job entries to.
        batch_size: number of tiles per worker job.
        count: if set, randomly sample this many tiles (for testing).
    """
    stack_dir = UPath(stack_path)
    all_tile_ids = sorted(
        m.group(1) for f in os.listdir(str(stack_dir)) if (m := _TILE_RE.match(f))
    )
    logger.info("found %d tiles in %s", len(all_tile_ids), stack_path)

    trans_dir = UPath(transitions_path)
    count_dir = UPath(change_count_path)
    trans_existing = (
        {p.name for p in trans_dir.iterdir()} if trans_dir.exists() else set()
    )
    count_existing = (
        {p.name for p in count_dir.iterdir()} if count_dir.exists() else set()
    )

    pending = [
        tid
        for tid in all_tile_ids
        if f"{tid}_transitions.tif" not in trans_existing
        or f"{tid}_change_count.tif" not in count_existing
    ]
    logger.info(
        "%d tiles pending (%d already completed)",
        len(pending),
        len(all_tile_ids) - len(pending),
    )

    if count is not None and len(pending) > count:
        pending = random.sample(pending, count)
        logger.info("randomly sampled %d tiles", len(pending))

    jobs: list[list[str]] = []
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        args = [
            "--stack_path",
            stack_path,
            "--transitions_path",
            transitions_path,
            "--change_count_path",
            change_count_path,
            "--tile_ids",
            json.dumps(batch),
        ]
        jobs.append(args)

    random.shuffle(jobs)
    rslp.common.worker.write_jobs(
        queue_name, "change_finder_v2", "esri_compute_changes_batch", jobs
    )
    logger.info(
        "wrote %d jobs (%d tiles) to queue %s", len(jobs), len(pending), queue_name
    )
