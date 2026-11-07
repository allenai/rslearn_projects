"""Job directory on a shared POSIX filesystem (WEKA) for LCC prediction tasks.

Each prediction task (one tile) is a file in the job directory, and workers claim
tiles by exclusively creating claim files, so no external queue is needed::

    <job_dir>/
      config.json                    # shared worker settings (JobConfig)
      tasks/<tile>.json              # one serialized PredictTaskArgs per tile
      claims/<tile>.claim.<gen>      # versioned claims, refreshed by the holder
      failures/<tile>.<uuid>         # one marker per failed attempt

``<tile>`` is the stem of the tile's ``_meta.json`` in the output directory, which
stays the source of truth for completion. Files are written to a temporary name and
renamed into place, so readers never see partial files.

A worker claims a tile by creating ``claims/<tile>.claim.<g+1>`` with O_EXCL, where
``g`` is the newest existing generation (0 if none). Only one creator can succeed,
even across nodes (a test with 144 processes on 8 nodes racing on the same files on
WEKA always had exactly one winner). A tile can be claimed if it has no claims or its newest claim is
stale (mtime older than stale_seconds); the holder keeps its claims fresh with
ClaimRefresher. Taking over a stale claim always creates a new generation, so a
fresh claim is never replaced by mistake. Claim file contents are informational
only: another worker can briefly see a new claim file still empty.

When a tile is done, its task file and then its claims are deleted. A failed attempt
writes a failure marker and deletes its claim; tiles with max_failures markers are
skipped. Tile outputs are idempotent, so a rare double claim only wastes compute.

Paths must be local POSIX paths (e.g. the WEKA mount), since exclusive creation and
atomic rename are not available through object storage APIs.
"""

import json
import os
import random
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from rslearn.utils.geometry import PixelBounds, Projection
from upath import UPath

from rslp.log_utils import get_logger

from .predict_pipeline import get_meta_fname

logger = get_logger(__name__)

CONFIG_FNAME = "config.json"
TASKS_DIR = "tasks"
CLAIMS_DIR = "claims"
FAILURES_DIR = "failures"
TASK_SUFFIX = ".json"
CLAIM_INFIX = ".claim."
META_SUFFIX = "_meta.json"

DEFAULT_STALE_SECONDS = 1800
DEFAULT_REFRESH_SECONDS = 120
DEFAULT_MAX_FAILURES = 3


@dataclass(frozen=True)
class PredictTaskArgs:
    """One prediction task: a tile and its reference timestamp."""

    projection: Projection
    bounds: PixelBounds
    # The reference timestamp as (T, T).
    time_range: tuple[datetime, datetime]

    def serialize(self) -> dict[str, Any]:
        """Serialize the task to a JSON-encodable dictionary."""
        return dict(
            projection_json=self.projection.serialize(),
            bounds=list(self.bounds),
            time_range=[t.isoformat() for t in self.time_range],
        )

    @staticmethod
    def deserialize(d: dict[str, Any]) -> "PredictTaskArgs":
        """Deserialize a task from its serialize() dictionary."""
        b = d["bounds"]
        return PredictTaskArgs(
            projection=Projection.deserialize(d["projection_json"]),
            bounds=(b[0], b[1], b[2], b[3]),
            time_range=(
                datetime.fromisoformat(d["time_range"][0]),
                datetime.fromisoformat(d["time_range"][1]),
            ),
        )


@dataclass(frozen=True)
class JobConfig:
    """Settings shared by all workers of a job directory."""

    out_path: str
    zarr_store_path: str
    zarr_pyramid_path: str | None = None
    fast_inference: bool = True


@dataclass(frozen=True)
class Claim:
    """A tile claimed by this worker."""

    tile: str
    path: str
    task: PredictTaskArgs


def check_local_path(path: str) -> None:
    """Raise unless path is a local POSIX path (not a URL)."""
    if "://" in path:
        raise ValueError(f"job directory must be a local POSIX path, not {path}")


def tile_name(projection: Projection, bounds: PixelBounds) -> str:
    """Get the name of a tile in the job directory (its _meta.json stem)."""
    return get_meta_fname(".", projection, bounds).name.removesuffix(META_SUFFIX)


def _atomic_write_json(path: str, data: Any) -> None:
    """Write JSON to a temporary file and rename it into place."""
    tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.rename(tmp_path, path)


def _task_path(job_dir: str, tile: str) -> str:
    return os.path.join(job_dir, TASKS_DIR, tile + TASK_SUFFIX)


def _claim_path(job_dir: str, tile: str, generation: int) -> str:
    return os.path.join(job_dir, CLAIMS_DIR, f"{tile}{CLAIM_INFIX}{generation}")


def read_job_config(job_dir: str) -> JobConfig:
    """Read the config of a job directory."""
    with open(os.path.join(job_dir, CONFIG_FNAME)) as f:
        return JobConfig(**json.load(f))


def write_job_dir(
    job_dir: str,
    config: JobConfig,
    tasks: list[PredictTaskArgs],
    count: int | None = None,
) -> int:
    """Create or extend a job directory with tasks.

    Args:
        job_dir: the job directory.
        config: the worker settings. An existing config.json must match it.
        tasks: the tasks to add. Tiles that already have a task file or a
            _meta.json in config.out_path are skipped.
        count: write at most this many of the remaining tasks (randomly sampled).

    Returns:
        the number of task files written.
    """
    check_local_path(job_dir)
    for subdir in [TASKS_DIR, CLAIMS_DIR, FAILURES_DIR]:
        os.makedirs(os.path.join(job_dir, subdir), exist_ok=True)

    config_path = os.path.join(job_dir, CONFIG_FNAME)
    if os.path.exists(config_path):
        existing = read_job_config(job_dir)
        if existing != config:
            raise ValueError(
                f"job directory {job_dir} has config {existing}, not {config}"
            )
    else:
        _atomic_write_json(config_path, asdict(config))

    existing_tasks = set(os.listdir(os.path.join(job_dir, TASKS_DIR)))
    out_upath = UPath(config.out_path)
    done = (
        {fname.name for fname in out_upath.iterdir()} if out_upath.exists() else set()
    )
    pending: dict[str, PredictTaskArgs] = {}
    for task in tasks:
        tile = tile_name(task.projection, task.bounds)
        if tile + TASK_SUFFIX in existing_tasks or tile + META_SUFFIX in done:
            continue
        pending[tile] = task
    tiles = list(pending)
    if count is not None and len(tiles) > count:
        tiles = random.sample(tiles, count)
    for tile in tiles:
        _atomic_write_json(_task_path(job_dir, tile), pending[tile].serialize())
    logger.info("wrote %d of %d tasks to %s", len(tiles), len(tasks), job_dir)
    return len(tiles)


@dataclass
class _DirState:
    """One listing of a job directory."""

    tiles: list[str]
    # Newest claim generation per tile.
    generations: dict[str, int]
    # Number of failure markers per tile.
    failures: dict[str, int]


def _list_state(job_dir: str) -> _DirState:
    tiles = [
        name.removesuffix(TASK_SUFFIX)
        for name in os.listdir(os.path.join(job_dir, TASKS_DIR))
        if name.endswith(TASK_SUFFIX)
    ]
    generations: dict[str, int] = {}
    for name in os.listdir(os.path.join(job_dir, CLAIMS_DIR)):
        tile, sep, generation = name.rpartition(CLAIM_INFIX)
        if not sep or not generation.isdigit():
            continue
        generations[tile] = max(generations.get(tile, 0), int(generation))
    failures: dict[str, int] = {}
    for name in os.listdir(os.path.join(job_dir, FAILURES_DIR)):
        tile = name.rpartition(".")[0]
        failures[tile] = failures.get(tile, 0) + 1
    return _DirState(tiles=tiles, generations=generations, failures=failures)


def _claim_age(path: str) -> float | None:
    """Get the seconds since a claim was last refreshed, or None if it is gone."""
    try:
        return time.time() - os.stat(path).st_mtime
    except FileNotFoundError:
        return None


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _remove_claims(job_dir: str, tile: str) -> None:
    prefix = tile + CLAIM_INFIX
    for name in os.listdir(os.path.join(job_dir, CLAIMS_DIR)):
        if name.startswith(prefix):
            _remove(os.path.join(job_dir, CLAIMS_DIR, name))


def claim_next(
    job_dir: str,
    out_path: str,
    worker_id: str,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    max_failures: int = DEFAULT_MAX_FAILURES,
    exclude: set[str] | None = None,
    rng: random.Random | None = None,
) -> Claim | None:
    """Claim a random claimable tile.

    Args:
        job_dir: the job directory.
        out_path: the output directory with the tiles' _meta.json.
        worker_id: identifies this worker in the claim file (informational).
        stale_seconds: claims not refreshed for this long can be taken over.
        max_failures: skip tiles with at least this many failure markers.
        exclude: tiles not to claim, e.g. ones that already failed in this worker.
        rng: random number generator for the order in which tiles are tried.

    Returns:
        the claim, or None if no tile is claimable.
    """
    state = _list_state(job_dir)
    candidates = [tile for tile in state.tiles if tile not in (exclude or set())]
    (rng or random).shuffle(candidates)
    for tile in candidates:
        if state.failures.get(tile, 0) >= max_failures:
            continue
        generation = state.generations.get(tile, 0)
        if generation > 0:
            age = _claim_age(_claim_path(job_dir, tile, generation))
            if age is not None and age < stale_seconds:
                continue

        path = _claim_path(job_dir, tile, generation + 1)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            f.write(worker_id)

        # The tile may have been finished since the listing.
        try:
            with open(_task_path(job_dir, tile)) as f:
                task = PredictTaskArgs.deserialize(json.load(f))
        except FileNotFoundError:
            _remove(path)
            continue
        if get_meta_fname(out_path, task.projection, task.bounds).exists():
            logger.info("tile %s is already done, cleaning up", tile)
            _remove(_task_path(job_dir, tile))
            _remove_claims(job_dir, tile)
            continue

        logger.info("claimed tile %s (generation %d)", tile, generation + 1)
        return Claim(tile=tile, path=path, task=task)
    return None


def finish(job_dir: str, claim: Claim) -> None:
    """Remove a finished tile's task file and then its claims."""
    _remove(_task_path(job_dir, claim.tile))
    _remove_claims(job_dir, claim.tile)


def release(job_dir: str, claim: Claim, failed: bool, message: str = "") -> None:
    """Give up a claim so that another worker can take the tile.

    Args:
        job_dir: the job directory.
        claim: the claim to release.
        failed: whether the attempt failed, in which case a failure marker (with
            message as its content) is written first.
        message: the failure details.
    """
    if failed:
        marker = os.path.join(job_dir, FAILURES_DIR, f"{claim.tile}.{uuid.uuid4().hex}")
        with open(marker, "w") as f:
            f.write(message)
    _remove(claim.path)


class ClaimRefresher:
    """Background thread that keeps this worker's claims fresh."""

    def __init__(self, interval: float = DEFAULT_REFRESH_SECONDS):
        """Create a new ClaimRefresher.

        Args:
            interval: seconds between refreshes.
        """
        self.interval = interval
        self._paths: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="claim_refresher", daemon=True
        )

    def start(self) -> None:
        """Start refreshing."""
        self._thread.start()

    def stop(self) -> None:
        """Stop refreshing."""
        self._stop.set()
        self._thread.join()

    def add(self, claim: Claim) -> None:
        """Start refreshing a claim."""
        with self._lock:
            self._paths.add(claim.path)

    def remove(self, claim: Claim) -> None:
        """Stop refreshing a claim."""
        with self._lock:
            self._paths.discard(claim.path)

    def refresh(self) -> None:
        """Refresh the mtime of every held claim."""
        with self._lock:
            paths = list(self._paths)
        for path in paths:
            try:
                os.utime(path)
            except FileNotFoundError:
                logger.warning("claim %s disappeared", path)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.refresh()


def job_status(
    job_dir: str,
    stale_seconds: float = DEFAULT_STALE_SECONDS,
    max_failures: int = DEFAULT_MAX_FAILURES,
) -> dict[str, int]:
    """Count the tiles of a job directory by state and log the counts.

    Args:
        job_dir: the job directory.
        stale_seconds: claims not refreshed for this long count as stale.
        max_failures: tiles with at least this many failure markers count as failed.

    Returns:
        the counts: done (tiles with a _meta.json in out_path), remaining (task
        files), and of the remaining ones claimed (fresh claim), stale (stale
        claim), failed (skipped for failures) and unclaimed.
    """
    check_local_path(job_dir)
    config = read_job_config(job_dir)
    state = _list_state(job_dir)
    out_upath = UPath(config.out_path)
    counts = dict(
        done=(
            sum(1 for p in out_upath.iterdir() if p.name.endswith(META_SUFFIX))
            if out_upath.exists()
            else 0
        ),
        remaining=len(state.tiles),
        claimed=0,
        stale=0,
        failed=0,
        unclaimed=0,
    )
    for tile in state.tiles:
        if state.failures.get(tile, 0) >= max_failures:
            counts["failed"] += 1
            continue
        generation = state.generations.get(tile, 0)
        age = (
            _claim_age(_claim_path(job_dir, tile, generation))
            if generation > 0
            else None
        )
        if age is None:
            counts["unclaimed"] += 1
        elif age < stale_seconds:
            counts["claimed"] += 1
        else:
            counts["stale"] += 1
    logger.info("job directory %s: %s", job_dir, counts)
    return counts
