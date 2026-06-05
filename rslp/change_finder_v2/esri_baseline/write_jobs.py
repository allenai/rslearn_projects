"""Enumerate ESRI land cover tiles and write Beaker jobs to build stacked GeoTIFFs.

Queries the Planetary Computer STAC API for all tiles in the
``io-lulc-annual-v02`` collection (one year is sufficient to discover the full
tile set), filters to tiles that don't yet have outputs, batches them, and writes
jobs to a Beaker queue for processing by ``build_stacks``.

Usage::

    python -m rslp.main change_finder_v2 esri_write_jobs \
        --out_path /weka/dfive-default/rslearn-eai/datasets/esri_lc_stacks/ \
        --queue_name favyen/esri-baseline-queue
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone

from rslearn.data_sources.planetary_computer import PlanetaryComputerStacClient
from upath import UPath

import rslp.common.worker

from .build_stacks import COLLECTION, STAC_ENDPOINT, _get_output_path

logger = logging.getLogger(__name__)

# Query a single recent year to discover all tile IDs.
DISCOVERY_YEAR = 2023


def _discover_tile_ids(
    client: PlanetaryComputerStacClient,
) -> list[str]:
    """Get all tile IDs from the STAC collection by querying one year.

    Uses datetime filtering which is standard STAC (no query/CQL2 needed).

    Returns a sorted list of unique tile ID strings.
    """
    logger.info(
        "discovering tile IDs from collection %s (year %d)", COLLECTION, DISCOVERY_YEAR
    )
    items = client.search(
        collections=[COLLECTION],
        date_time=(
            datetime(DISCOVERY_YEAR, 1, 1, tzinfo=timezone.utc),
            datetime(DISCOVERY_YEAR, 12, 31, tzinfo=timezone.utc),
        ),
    )
    logger.info("STAC returned %d items for year %d", len(items), DISCOVERY_YEAR)

    tile_ids: set[str] = set()
    for item in items:
        parts = item.id.rsplit("-", 1)
        if len(parts) == 2:
            tile_ids.add(parts[0])
    return sorted(tile_ids)


def write_jobs(
    out_path: str,
    queue_name: str,
    batch_size: int = 10,
    count: int | None = None,
) -> None:
    """Enumerate ESRI land cover tiles and write jobs to a Beaker queue.

    Args:
        out_path: directory where stacked GeoTIFFs are written by workers.
            Also used to skip tiles whose output already exists.
        queue_name: the Beaker queue to write job entries to.
        batch_size: number of tiles per worker job.
        count: if set, randomly sample this many tiles (for testing).
    """
    client = PlanetaryComputerStacClient(STAC_ENDPOINT)
    all_tile_ids = _discover_tile_ids(client)
    logger.info("discovered %d tile IDs", len(all_tile_ids))

    # Filter out tiles that already have outputs.
    out_upath = UPath(out_path)
    if out_upath.exists():
        existing = {p.name for p in out_upath.iterdir()}
    else:
        existing = set()

    pending = [
        tid
        for tid in all_tile_ids
        if _get_output_path(out_path, tid).name not in existing
    ]
    logger.info(
        "%d tiles pending (%d already completed)",
        len(pending),
        len(all_tile_ids) - len(pending),
    )

    if count is not None and len(pending) > count:
        pending = random.sample(pending, count)
        logger.info("randomly sampled %d tiles", len(pending))

    # Batch into jobs.
    jobs: list[list[str]] = []
    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        args = [
            "--out_path",
            out_path,
            "--tile_ids",
            json.dumps(batch),
        ]
        jobs.append(args)

    random.shuffle(jobs)
    rslp.common.worker.write_jobs(
        queue_name, "change_finder_v2", "esri_build_stacks", jobs
    )
    logger.info(
        "wrote %d jobs (%d tiles) to queue %s", len(jobs), len(pending), queue_name
    )
