"""Worker that predicts tiles from a job directory (see job_dir.py).

The worker claims one tile at a time. While a tile is predicted on the GPU, the next
tile is claimed and prepared (windows created and imagery materialized) in a
background process, so only the worker's first tile waits for materialization. The
worker exits once no tile is claimable; tiles left behind by workers that died
become claimable after the stale-claim timeout and are picked up by workers launched
later.
"""

import os
import shutil
import signal
import socket
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from types import FrameType
from typing import Any

from beaker import BeakerJobPriority

import rslp.common.worker
from rslp.log_utils import get_logger
from rslp.utils.beaker import DEFAULT_MIN_RUNTIME, WekaMount

from .fast_inference import prepare_fast_inference
from .job_dir import (
    DEFAULT_MAX_FAILURES,
    DEFAULT_REFRESH_SECONDS,
    DEFAULT_STALE_SECONDS,
    Claim,
    ClaimRefresher,
    JobConfig,
    check_local_path,
    claim_next,
    finish,
    read_job_config,
    release,
)
from .predict_pipeline import (
    BackgroundPrepare,
    PreparedTile,
    check_tile,
    predict_tile,
    prepare_tile,
)

logger = get_logger(__name__)


@dataclass
class _InFlightTile:
    """A claimed tile that is being prepared or predicted."""

    claim: Claim
    tile_dir: str
    prepare_kwargs: dict[str, Any]
    background: BackgroundPrepare | None = None
    # Whether the claim was finished or released.
    settled: bool = field(default=False)

    def prepared(self) -> PreparedTile:
        """Wait for (or run) the tile's preparation."""
        if self.background is not None:
            return self.background.result()
        return prepare_tile(**self.prepare_kwargs)


def _raise_system_exit(signo: int, frame: FrameType | None) -> None:
    raise SystemExit(f"received signal {signo}")


def worker(
    job_dir: str,
    scratch_path: str = "/tmp/scratch",
    stale_seconds: int = DEFAULT_STALE_SECONDS,
    refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
    max_failures: int = DEFAULT_MAX_FAILURES,
    max_consecutive_errors: int = 3,
    background_prepare: bool = True,
) -> None:
    """Predict tiles from a job directory until none is claimable.

    Args:
        job_dir: the job directory (written by write_jobs), a local POSIX path.
        scratch_path: local directory for the tiles' scratch datasets. Up to two
            tiles of imagery are stored at once.
        stale_seconds: claims not refreshed for this long can be taken over.
        refresh_seconds: seconds between refreshes of this worker's claims.
        max_failures: skip tiles that failed this many times.
        max_consecutive_errors: raise after this many tiles fail in a row.
        background_prepare: prepare the next tile in a background process while
            predicting; if false, tiles are prepared and predicted sequentially in
            this process.
    """
    check_local_path(job_dir)
    config = read_job_config(job_dir)
    worker_id = f"{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    logger.info("worker %s starting on %s", worker_id, job_dir)

    # Fast inference is prepared in this process, which runs the model; the
    # background processes only materialize.
    model_extra_args = prepare_fast_inference() if config.fast_inference else []
    os.makedirs(scratch_path, exist_ok=True)
    refresher = ClaimRefresher(refresh_seconds)
    refresher.start()
    # Release claims on preemption by unwinding through the finally block below.
    previous_handler = signal.signal(signal.SIGTERM, _raise_system_exit)

    # Tiles that failed in this worker, which it leaves to other workers.
    failed_tiles: set[str] = set()

    def claim_and_start() -> _InFlightTile | None:
        claim = claim_next(
            job_dir,
            config.out_path,
            worker_id,
            stale_seconds=stale_seconds,
            max_failures=max_failures,
            exclude=failed_tiles,
        )
        if claim is None:
            return None
        refresher.add(claim)
        tile_dir = tempfile.mkdtemp(dir=scratch_path)
        tile = _InFlightTile(
            claim=claim,
            tile_dir=tile_dir,
            prepare_kwargs=dict(
                projection=claim.task.projection,
                bounds=claim.task.bounds,
                time_range=claim.task.time_range,
                scratch_path=os.path.join(tile_dir, "scratch"),
            ),
        )
        if background_prepare:
            try:
                tile.background = BackgroundPrepare(tile.prepare_kwargs)
            except BaseException:
                release(job_dir, claim, failed=False)
                cleanup(tile)
                raise
        return tile

    def cleanup(tile: _InFlightTile) -> None:
        refresher.remove(tile.claim)
        shutil.rmtree(tile.tile_dir, ignore_errors=True)

    current: _InFlightTile | None = None
    next_tile: _InFlightTile | None = None
    consecutive_errors = 0
    num_done = 0
    try:
        current = claim_and_start()
        while current is not None:
            next_tile = None
            start_time = time.monotonic()
            try:
                check_tile(
                    current.claim.task.projection,
                    current.claim.task.bounds,
                    config.zarr_pyramid_path,
                )
                prepared = current.prepared()
                timings = dict(prepared.timings)
                timings["materialize_wait"] = time.monotonic() - start_time
                if background_prepare:
                    next_tile = claim_and_start()
                _predict(
                    current, prepared, config, start_time, timings, model_extra_args
                )
                finish(job_dir, current.claim)
                current.settled = True
                num_done += 1
                consecutive_errors = 0
            except Exception:
                logger.exception("tile %s failed", current.claim.tile)
                # The tile may have failed before its preparation finished (e.g. in
                # check_tile), so stop the preparation before its scratch directory
                # is removed.
                if current.background is not None:
                    current.background.terminate()
                release(
                    job_dir, current.claim, failed=True, message=traceback.format_exc()
                )
                current.settled = True
                failed_tiles.add(current.claim.tile)
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    raise
            finally:
                cleanup(current)
            current = next_tile if next_tile is not None else claim_and_start()
        logger.info(
            "worker %s finished %d tiles; none left to claim", worker_id, num_done
        )
    finally:
        # On errors and preemption, give up the tiles this worker still holds.
        for tile in [current, next_tile]:
            if tile is None or tile.settled:
                continue
            if tile.background is not None:
                tile.background.terminate()
            release(job_dir, tile.claim, failed=False)
            tile.settled = True
            cleanup(tile)
        refresher.stop()
        signal.signal(signal.SIGTERM, previous_handler)


def _predict(
    tile: _InFlightTile,
    prepared: PreparedTile,
    config: JobConfig,
    start_time: float,
    timings: dict[str, float],
    model_extra_args: list[str],
) -> None:
    """Predict a prepared tile and write its outputs."""
    logger.info("predicting tile %s", tile.claim.tile)
    predict_tile(
        prepared,
        out_path=config.out_path,
        zarr_store_path=config.zarr_store_path,
        start_time=start_time,
        timings=timings,
        zarr_pyramid_path=config.zarr_pyramid_path,
        model_extra_args=model_extra_args,
    )


def launch_workers(
    job_dir: str,
    image_name: str,
    num_workers: int,
    cluster: list[str],
    gpus: int = 1,
    shared_memory: str | None = None,
    priority: BeakerJobPriority = BeakerJobPriority.low,
    weka_mounts: list[WekaMount] = [],
    extra_env_vars: dict[str, str] | None = None,
    extra_env_secrets: dict[str, str] | None = None,
    min_runtime: timedelta = DEFAULT_MIN_RUNTIME,
) -> None:
    """Launch Beaker jobs that run the worker on a job directory.

    Args:
        job_dir: the job directory, as mounted on the workers.
        image_name: the Beaker image name to use for the jobs.
        num_workers: number of workers to launch.
        cluster: clusters to target.
        gpus: number of GPUs to request per worker.
        shared_memory: shared memory string like "256GiB".
        priority: priority to assign the Beaker jobs.
        weka_mounts: list of weka mounts for Beaker job.
        extra_env_vars: additional environment variables mapping name to value.
        extra_env_secrets: additional environment variables mapping name to the
            Beaker secret to read the value from.
        min_runtime: how long to protect each (preemptible) worker job from
            preemption.
    """
    rslp.common.worker.launch_workers(
        image_name=image_name,
        num_workers=num_workers,
        cluster=cluster,
        gpus=gpus,
        shared_memory=shared_memory,
        priority=priority,
        weka_mounts=weka_mounts,
        extra_env_vars=extra_env_vars,
        extra_env_secrets=extra_env_secrets,
        min_runtime=min_runtime,
        worker_args=["olmoearth_lcc", "worker", "--job_dir", job_dir],
    )
