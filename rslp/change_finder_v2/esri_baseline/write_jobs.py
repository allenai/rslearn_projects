"""Enumerate ESRI land cover tiles and write Beaker jobs to build stacked GeoTIFFs.

Discovers all available S2 tile IDs by listing the public S3 bucket for a
single year, filters to tiles that don't yet have outputs, batches them, and
writes jobs to a Beaker queue for processing by ``build_stacks``.

Usage::

    python -m rslp.main change_finder_v2 esri_write_jobs \
        --out_path /weka/dfive-default/rslearn-eai/datasets/esri_lc_stacks/ \
        --queue_name favyen/esri-baseline-queue
"""

from __future__ import annotations

import json
import logging
import random
import re

import requests
from upath import UPath

import rslp.common.worker

from .build_stacks import _get_output_path

logger = logging.getLogger(__name__)

S3_BUCKET = "io-10m-annual-lulc"
S3_LIST_URL = f"https://{S3_BUCKET}.s3.us-west-2.amazonaws.com"

# Use a single year to discover all tile IDs.
DISCOVERY_YEAR = 2023

# Matches filenames like "10T_2023.tif" and extracts the tile ID.
_TILE_RE = re.compile(rf"^(.+)_{DISCOVERY_YEAR}\.tif$")


def _discover_tile_ids() -> list[str]:
    """List the S3 bucket to discover all tile IDs for one year.

    Uses the S3 XML list API (no auth needed for public buckets).

    Returns a sorted list of unique tile ID strings.
    """
    logger.info(
        "discovering tile IDs from s3://%s (year %d)", S3_BUCKET, DISCOVERY_YEAR
    )

    tile_ids: set[str] = set()
    marker: str | None = None

    while True:
        params: dict[str, str] = {"max-keys": "1000"}
        if marker is not None:
            params["marker"] = marker

        resp = requests.get(S3_LIST_URL, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.text

        # Parse keys from the XML response (avoid adding lxml dependency).
        keys = re.findall(r"<Key>([^<]+)</Key>", body)
        if not keys:
            break

        for key in keys:
            m = _TILE_RE.match(key)
            if m:
                tile_ids.add(m.group(1))

        # S3 truncation indicator
        if "<IsTruncated>true</IsTruncated>" in body:
            marker = keys[-1]
        else:
            break

    logger.info("discovered %d tile IDs", len(tile_ids))
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
    all_tile_ids = _discover_tile_ids()

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
