"""Keep an embedding run making progress on preemptible workers.

Workers are preemptible and the GPU clusters are routinely saturated, so a long run
loses workers constantly. Two existing properties make that survivable:

1. Every job is idempotent -- ``predict_pipeline`` returns immediately when the tile's
   completion marker already exists -- so re-enqueuing work is cheap and safe.
2. ``get_jobs`` derives the remaining work from those markers, so "what is left" never
   has to be tracked separately.

Two design points:

**Keep the queue shallow.** A Beaker queue entry claimed by a worker that then dies is
not released back to the queue, and the queue API has no call to release one. Entries
do age out, but only after ``write_jobs``' ``expires_in_sec``, which defaults to a
week, so within a run that work is lost. Enqueuing a whole run up front therefore
bleeds work steadily. This enqueues only a small buffer and refills it from the markers,
bounding the loss to about one entry per worker death.

**Run each cycle in a child process.** The Beaker client has no RPC timeout and a hung
call cannot be interrupted in-process, so every cycle runs in a spawned child the parent
terminates if it overruns its budget.
"""

import json
import multiprocessing
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from multiprocessing.sharedctypes import Synchronized
from typing import Any, TypeVar

from beaker import (
    Beaker,
    BeakerConstraints,
    BeakerExperimentSpec,
    BeakerJobPriority,
    BeakerTaskResources,
    BeakerWorkloadType,
)
from upath import UPath

import rslp.common.worker
from rslp.large_scale_embeddings.predict_pipeline import EmbeddingInputs
from rslp.large_scale_embeddings.render_pca import get_render_jobs
from rslp.large_scale_embeddings.render_web_pca import get_web_jobs
from rslp.large_scale_embeddings.write_jobs import get_jobs
from rslp.large_scale_embeddings.zarr_store import (
    DEFAULT_PCA_MAX_LEVEL,
    get_store_years,
)
from rslp.log_utils import get_logger
from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_WORKSPACE,
    WekaMount,
    create_gcp_credentials_mount,
    get_base_env_vars,
)

logger = get_logger(__name__)

# The stages this supervisor can drive. Both are idempotent and marker-driven, so the
# same shallow-queue and worker-top-up loop works for either; only how remaining work is
# enumerated and which workflow the entries name differ.
STAGE_PREDICT = "predict"
STAGE_RENDER_UTM_PCA = "render_utm_pca"
# One zoom level at a time. A coarse shard is built from the four below it, so its
# inputs must already exist; running every zoom as one flat stage would race. The zoom
# is a supervise argument rather than a stage name so the resumability, worker
# management and marker handling are shared with every other stage.
STAGE_RENDER_WEB_PCA = "render_web_pca"
STAGES = (STAGE_PREDICT, STAGE_RENDER_UTM_PCA, STAGE_RENDER_WEB_PCA)

# Pending entries to keep per worker: enough that no worker idles waiting for work,
# few enough that entries orphaned by dying workers stay a rounding error.
PENDING_PER_WORKER = 3

# A cycle that outruns this is assumed wedged (almost always a hung Beaker RPC) and
# gets killed. Cycles normally take well under a minute.
DEFAULT_CYCLE_BUDGET_SECONDS = int(timedelta(minutes=10).total_seconds())

# Where a supervisor caches its enumerated block list between cycles.
DEFAULT_ENUMERATION_CACHE_DIR = "/tmp/rslp_enumeration_cache"

# Minimum runtime to request for the supervisor. It is a cheap CPU job that should stay
# up for the whole run, but a shorter request is placed sooner, and auto_resume brings it
# back if it is preempted, so there is no reason to ask for the eight-hour maximum.
DEFAULT_SUPERVISOR_MIN_RUNTIME = timedelta(hours=1)

# Cap on the workload listing the worker count is drawn from. Scoped to this user's
# unfinalized experiments, so it only has to cover one person's concurrent runs.
WORKER_LIST_LIMIT = 500

# Sentinel for "the child did not report a result" (timed out, crashed, or killed).
_NO_RESULT = -1

# Consecutive failed cycles tolerated before giving up. A killed cycle is transient
# and worth retrying, but a bad path or a revoked credential fails identically every
# time, and retrying forever turns a startup mistake into a job that looks alive.
MAX_CONSECUTIVE_CYCLE_FAILURES = 3

# Deployment defaults, collected here rather than buried in the signatures below so a
# different environment only has to change one place (or override them per call).
# The checkpoint lives on WEKA, so workers need it mounted.
DEFAULT_WEKA_BUCKET = "dfive-default"
DEFAULT_WEKA_MOUNT_PATH = "/weka/dfive-default"
# The OlmoEarth Datasets data source needs an endpoint plus a bearer token, the latter
# read from a Beaker secret of this name in the target workspace.
DEFAULT_DATASETS_API_URL = "https://datasets.olmoearth.allenai.org"
# This is the name of a Beaker secret, not a credential. (bandit flags the assignment
# because the name contains "token"/"secret".)
# Named to pair with OEDATASETS_API_URL.
DEFAULT_DATASETS_TOKEN_SECRET = "OEDATASETS_API_TOKEN"  # nosec
# Beaker secrets holding AWS credentials, mirroring what olmoearth_run's deployed runner
# injects. Data sources request assets with requester_pays=True, so an S3 asset needs
# signed requests or GDAL raises InvalidCredentials. S3 is the fallback backend.
DEFAULT_AWS_KEY_ID_SECRET = "AWS_ACCESS_KEY_ID"  # nosec
DEFAULT_AWS_SECRET_KEY_SECRET = "AWS_SECRET_ACCESS_KEY"  # nosec

# How long a claim is trusted before the job is offered again.
#
# Claims are never released, so a worker that dies mid-job leaves its claim behind
# forever and skipping every claimed job would deadlock the run on the first death.
# Trusting a claim only while it is young keeps that recovery without the duplicate work
# a blind top-up generates. Size it well above one job's runtime.
DEFAULT_CLAIM_STALE_SECONDS = int(timedelta(minutes=90).total_seconds())

# Workload states in which a worker is already running, and so has registered with the
# queue and is represented by its heartbeat. Counting these as "starting" as well
# double-counts a whole launch batch for the length of the startup grace.
RUNNING_WORKLOAD_STATUSES = frozenset({4, 5, 6})  # running, stopping, uploading_results

# Workload states in which a worker has been created but cannot have registered yet.
#
# These count no matter how old they are. A saturated GPU cluster leaves a worker queued
# for hours -- 118 minutes median, 316 at the tail, measured while scaling to 256 -- and
# aging it out of the count makes the supervisor believe it is short, launch more, and
# queue those too. That storm put 240 phantom requests on the cluster before anyone
# noticed, because queued workers hold no GPU and throughput looked fine.
PENDING_WORKLOAD_STATUSES = frozenset({1, 2, 3, 10})  # submitted, queued, initializing,
# ready_to_start

# Workload state of a worker that is up and claiming jobs, and so can be asked to
# retire. Narrower than RUNNING_WORKLOAD_STATUSES on purpose: a worker that is already
# stopping or uploading is on its way out and naming it in the drain list would shed
# capacity that is about to come back anyway.
DRAINABLE_WORKLOAD_STATUSES = frozenset({4})  # running

# How long a queue worker's heartbeat is trusted before it is presumed dead.
#
# A worker registers with the queue at start and the Beaker SDK refreshes the
# registration from a background thread, so the heartbeat keeps ticking through a job
# that runs for tens of minutes. It stops when the process dies or its channel thread
# does, which is how a worker ends up alive to Beaker but never claiming again. Counting
# those toward the pool strands the run: launches are capped by outstanding work, so once
# the dead count reaches the outstanding count the supervisor launches nothing and the
# last jobs are never claimed. Generous next to the heartbeat interval (seconds).
WORKER_HEARTBEAT_STALE_SECONDS = int(timedelta(minutes=5).total_seconds())

# Most a capacity-sized pool may grow in one cycle.
#
# Cluster availability swings by hundreds of slots as other teams' jobs land and finish,
# and a controller that chased the spot value would answer a transient dip in occupancy
# with a launch of that size. That is how 240 phantom requests reached the cluster once
# already. Growth is capped and shrink is not: giving capacity back is always safe, and
# a worker exits on its own once the queue runs dry.
CAPACITY_MAX_STEP = 32

# How many scheduled jobs to scan when totalling an allocation's usage. A busy cluster
# runs several hundred; this is sized well above that so the total is not silently
# truncated into an underestimate, which would read as free allocation and over-launch.
ALLOCATION_JOB_LIMIT = 5000

# Longest min_runtime a job may ask for and still count as unallocated.
#
# Beaker treats a job that wants more than this as a claim on an allocation. At or under
# it the job is unallocated: it schedules only onto slots no allocation is holding, and
# yields to allocated work the moment that work appears. That is exactly the bargain a
# backfill worker wants, so this is a threshold to stay under, not a duration to tune.
BACKFILL_MIN_RUNTIME = timedelta(minutes=5)

# GDAL environment every worker needs, merged in below so it cannot be forgotten.
#
# GS_USER_PROJECT is required for the Landsat reads: rasterio_session_for_path honours
# requester_pays only for S3, so the requester-pays USGS mirror needs GDAL handed a
# billing project this way. Omitting it fails with an HTTP 400 that names no variable.
# This lives here rather than in full_run because supervise is what launches workers.
DEFAULT_WORKER_ENV_VARS = {
    "GS_USER_PROJECT": "earthsystem-dev-c3po",
}


class _Metrics:
    """Pushes one point per cycle to Weights & Biases, or nowhere.

    Every number here was previously only available by parsing container logs, which
    is how a seven-hour scheduling wait got read as a throughput collapse and a
    24-of-128 worker sample got read as an exact parked count. A time series makes
    those mistakes visible instead of plausible.

    Logging must never take the run down: a bad key, an outage or a missing package
    all end up as one warning and a no-op sink. The supervisor is the thing keeping a
    month-long run alive.
    """

    def __init__(self, project: str | None, name: str, config: dict) -> None:
        self._run = None
        if not project:
            return
        try:
            import wandb

            self._run = wandb.init(
                project=project, name=name, config=config, reinit=False
            )
            logger.info(
                "logging cycle metrics to wandb project %s as %s", project, name
            )
        except Exception:
            # Deliberately broad: any failure here must degrade to not logging.
            logger.warning(
                "wandb init failed; continuing without metrics", exc_info=True
            )
            self._run = None

    def log(self, step: int, values: dict) -> None:
        """Log one cycle's metrics, dropping any that were not measured."""
        if self._run is None:
            return
        try:
            self._run.log({k: v for k, v in values.items() if v is not None}, step=step)
        except Exception:
            logger.warning("wandb log failed; continuing", exc_info=True)
            self._run = None

    def finish(self) -> None:
        """Close the run, ignoring failures: the work is already done by here."""
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception:
            logger.debug("wandb finish failed; ignoring", exc_info=True)


@dataclass
class ModelConfig:
    """What the encoder is and how it is run.

    These all change the embeddings, so a change here needs its own store.
    """

    checkpoint_path: str
    patch_size: int = 1
    window_size: int = 16
    overlap_size: int = 4
    compile_model: bool = True
    # Crops per batch, or None to keep the model config's value. The GPU-memory knob:
    # batching groups independent crops, so it changes footprint and speed, not output.
    batch_size: int | None = None


@dataclass
class WorkerConfig:
    """The Beaker worker pool: what to run, where, and with which credentials."""

    image_name: str
    cluster: list[str]
    num_workers: int = 8
    gpus: int = 1
    # A preempted worker loses its whole job, since there is no intra-job checkpointing.
    priority: str = "urgent"
    shared_memory: str = "256GiB"
    # How long a worker waits for new work before exiting. Must exceed the cycle
    # interval, or the pool empties between refills. None leaves the worker's own
    # ten-second default, which suits a queue filled once up front but not one a
    # supervisor refills.
    idle_seconds: int | None = 900
    env_vars: dict[str, str] | None = None
    weka_bucket: str = DEFAULT_WEKA_BUCKET
    weka_mount_path: str = DEFAULT_WEKA_MOUNT_PATH
    datasets_api_url: str = DEFAULT_DATASETS_API_URL
    datasets_token_secret: str = DEFAULT_DATASETS_TOKEN_SECRET
    aws_key_id_secret: str = DEFAULT_AWS_KEY_ID_SECRET
    aws_secret_key_secret: str = DEFAULT_AWS_SECRET_KEY_SECRET
    # Fraction of the cluster's schedulable GPU slots to hold, or None to keep the pool
    # fixed at num_workers. When set, num_workers becomes the ceiling rather than the
    # target and the pool tracks what the cluster actually has free.
    #
    # Only honoured for a GPU stage. Beaker reports occupancy in GPU slots and nothing
    # else -- there is no cpu or memory accounting at any level -- so for a stage that
    # requests no GPU the number measures a resource the stage does not consume, and
    # sizing against it would be sizing against noise.
    capacity_fraction: float | None = None
    # The organisation's GPU slot allocation on `cluster`. Required alongside
    # capacity_fraction. Passed in rather than hardcoded because it is an agreement
    # between teams, not a property of the code, and it changes without warning.
    capacity_slots: int | None = None
    # Whose jobs count against that allocation. Defaults to the run's own workspace.
    capacity_workspace: str | None = None
    # Floor for a capacity-sized pool, so a momentarily full allocation cannot park the
    # run at zero workers and leave nothing to make progress when slots free up again.
    capacity_min_workers: int = 8
    # Fraction of the cluster's idle GPU slots to take on top of the allocation, or None
    # to stay inside the allocation.
    #
    # These workers run at `backfill_priority` with a min_runtime short enough to count
    # as unallocated, so they occupy only slots no allocation is holding and are
    # preempted as soon as allocated work wants them back. That makes them nearly free
    # to take and unreliable to keep: a preempted worker loses its block, the queue
    # re-offers it, and another worker picks it up. Worth it when the alternative is an
    # idle cluster, which is why this is sized from what is actually idle.
    #
    # Below 1.0 so a pool sizing itself from the same reading as everyone else's does
    # not converge on the same slots and thrash.
    backfill_fraction: float | None = None
    # Sanity bound on the backfill pool, not a policy limit: the point of backfill is to
    # take whatever is going, so this is set well above any real cluster's slot count and
    # exists only so a garbage occupancy reading cannot turn into a launch of that size.
    # Growth is paced separately by CAPACITY_MAX_STEP, which is what keeps a genuine
    # spike in free slots from arriving all at once.
    backfill_max_workers: int = 512
    # Priority for backfill workers. Must outrank nothing in particular: it only has to
    # be a priority the run may use without spending allocation.
    backfill_priority: str = "high"

    def __post_init__(self) -> None:
        """Merge the GDAL defaults so a caller cannot drop them by passing env_vars."""
        self.env_vars = {**DEFAULT_WORKER_ENV_VARS, **(self.env_vars or {})}


@dataclass
class CycleConfig:
    """Pacing of the supervision loop."""

    seconds: int = 180
    budget_seconds: int = DEFAULT_CYCLE_BUDGET_SECONDS
    # Where to cache the enumerated block list, or None to re-enumerate every cycle.
    #
    # Each cycle runs in a fresh process, so nothing is cached in memory between them
    # and the enumeration is paid again every time. That is seconds for a bounded run
    # and minutes for a global one, against a cycle interval measured in minutes. A
    # local path is right: the cache is derived data, rebuilt in one cycle if lost, and
    # keyed on the coverage mask so it cannot outlive the thing it describes.
    enumeration_cache_dir: str | None = DEFAULT_ENUMERATION_CACHE_DIR
    claim_stale_seconds: int = DEFAULT_CLAIM_STALE_SECONDS
    pending_per_worker: int = PENDING_PER_WORKER
    max_cycles: int | None = None
    # Weights & Biases project for per-cycle metrics, or None to log nowhere. The API
    # key is already mounted on every job by get_base_env_vars.
    wandb_project: str | None = None


@dataclass
class AoiConfig:
    """Which ground the run covers, and how it is cut into jobs."""

    # A job is the unit of work lost to a preemption: the completion marker is written
    # once, after every window in the block, so a job killed near the end redoes all of
    # it. Must be a multiple of PATCH_SIZE and divide TILE_SIZE. Changing it mid-run
    # orphans existing markers, since a marker is keyed on its block's bounds.
    job_size: int = 4096
    geojson_fname: str | None = None
    epsg_code: int | None = None
    wgs84_bounds: tuple[float, float, float, float] | None = None
    zone_numbers: list[int] | None = None


@dataclass
class PcaConfig:
    """Paths and levels for the render stages. Unused by the predict stage."""

    artifact_path: str | None = None
    store_path: str | None = None
    completed_path: str | None = None
    store_url: str | None = None
    max_level: int = DEFAULT_PCA_MAX_LEVEL
    web_store_path: str | None = None
    web_completed_path: str | None = None
    web_zoom: int | None = None
    web_base_zoom: int = 14


@dataclass
class SuperviseConfig:
    """Everything one supervision cycle reads.

    Assembled by `supervise` and handed to each child process whole, so a cycle cannot
    silently read a value the parent never set.
    """

    inputs: EmbeddingInputs
    years: list[int]
    store_path: str
    completed_path_template: str
    queue_name: str
    stage: str
    model: ModelConfig
    worker: WorkerConfig
    cycle: CycleConfig
    aoi: AoiConfig
    pca: PcaConfig


_T = TypeVar("_T")


def require_config(value: _T | None, field: str, context: str) -> _T:
    """Read a config value the caller cannot run without.

    PcaConfig and AoiConfig hold their fields as optional because the predict stage
    never sets them. The render stages need them, and passing the optional straight
    through would surface as an obscure failure deep in the job enumerator, so read
    them through this instead and name the missing field.

    Args:
        value: the configured value, possibly None.
        field: the field's name, for the error message.
        context: the stage or step requiring it, for the error message.

    Returns:
        the value, narrowed to non-None.

    Raises:
        ValueError: if the value was never set.
    """
    if value is None:
        raise ValueError(f"{context} requires {field} to be set")
    return value


def _state_name(entry: Any) -> str:
    """Get the state enum name for a queue entry (PENDING/CLAIMED/COMPLETED)."""
    status = entry.status
    try:
        return (
            status.DESCRIPTOR.fields_by_name["state"]
            .enum_type.values_by_number[int(status.state)]
            .name.split("_")[-1]
        )
    except (KeyError, ValueError):
        return "UNKNOWN"


def _entry_job_key(entry: Any) -> tuple[str, ...] | None:
    """The job an entry runs, as its argument list.

    Args:
        entry: a Beaker queue entry.

    Returns:
        the entry's args as a tuple, or None if the payload does not carry them.
    """
    try:
        values = entry.input.fields["args"].list_value.values
    except Exception:  # noqa: BLE001 - a malformed entry must not stop a cycle
        return None
    keys = tuple(v.string_value for v in values)
    return keys or None


def _in_flight_job_keys(
    entries: Any,
    now: float,
    claim_stale_seconds: int = DEFAULT_CLAIM_STALE_SECONDS,
) -> set[tuple[str, ...]]:
    """Jobs already queued or being worked on, which need no second entry.

    A pending entry is waiting to be picked up. A claimed entry counts only while its
    claim is younger than `claim_stale_seconds`; past that the worker holding it is
    presumed dead and the job is offered again.

    Args:
        entries: the queue's entries.
        now: current unix time, passed in so this stays testable.
        claim_stale_seconds: age past which a claim is ignored.

    Returns:
        the set of job keys that should not be enqueued again this cycle.
    """
    in_flight: set[tuple[str, ...]] = set()
    for entry in entries:
        state = _state_name(entry)
        if state not in ("PENDING", "CLAIMED"):
            continue
        key = _entry_job_key(entry)
        if key is None:
            continue
        if state == "PENDING":
            in_flight.add(key)
            continue
        claimed = getattr(getattr(entry, "status", None), "claimed", None)
        seconds = getattr(claimed, "seconds", 0) or 0
        # No timestamp means nothing can be concluded about the claim's age, so treat it
        # as live: re-offering a job that is genuinely being worked costs a duplicate,
        # while wrongly skipping one costs the whole run.
        if seconds == 0 or now - seconds < claim_stale_seconds:
            in_flight.add(key)
    return in_flight


def _stage_marker_paths(config: SuperviseConfig) -> list[str]:
    """The completion-marker directories this stage writes into.

    Args:
        config: the run configuration.

    Returns:
        one path per marker directory the stage is responsible for.
    """
    if config.stage == STAGE_RENDER_UTM_PCA:
        return [
            require_config(
                config.pca.completed_path, "pca.completed_path", config.stage
            )
        ]
    return [config.completed_path_template.format(year=year) for year in config.years]


def _any_completion_markers(config: SuperviseConfig) -> bool:
    """Whether this stage has already written at least one completion marker.

    A remaining count of zero has two very different causes: the stage is genuinely
    finished, or the AOI and zone filters excluded everything so nothing was ever
    enumerated. Markers on disk are what separates them, so the first-cycle guard
    consults this instead of inferring from the count alone.

    Args:
        config: the run configuration.

    Returns:
        True if any marker exists for this stage.
    """
    for path in _stage_marker_paths(config):
        upath = UPath(path)
        if upath.exists() and any(True for _ in upath.iterdir()):
            return True
    return False


def worker_name_prefix(queue_name: str) -> str:
    """Experiment-name prefix identifying the workers of one run.

    Args:
        queue_name: the Beaker queue name, e.g. "user/my-queue".

    Returns:
        the prefix to name this run's worker experiments with.
    """
    return "worker_" + queue_name.replace("/", "-")


def _allocated_slots_in_use(
    beaker: Any,
    worker: "WorkerConfig",
    workspace_id: str,
    live: int,
) -> int:
    """GPU slots the workspace holds on `worker.cluster`, excluding this run's own pool.

    The allocation is shared with everyone else in the workspace, so what this run may
    take is the allocation minus what colleagues are already holding. Our own workers
    are subtracted back out: counting them would make scaling up shrink our own
    headroom, and the pool would ratchet itself to zero.

    Args:
        beaker: an open Beaker client.
        worker: the pool configuration.
        workspace_id: the workspace whose jobs count against the allocation.
        live: this run's workers, starting or alive.

    Returns:
        slots held or queued for by everyone else in the workspace, never below zero.
    """
    total = 0
    for name in worker.cluster:
        # Eligible, not scheduled: a colleague's queued job has not been placed on a
        # node yet, but it is a claim on the allocation and will take slots the moment
        # any free up. Counting only what is running lets this pool take capacity
        # someone is already waiting for. Eligibility is the looser predicate -- a job
        # listing several clusters counts against each -- so this can overcount when
        # such a job lands elsewhere. That is the safe direction for a ceiling you are
        # trying not to exceed.
        for job in beaker.job.list(
            elegible_for_cluster=beaker.cluster.get(name),
            finalized=False,
            limit=ALLOCATION_JOB_LIMIT,
        ):
            if job.workspace_id == workspace_id:
                total += job.container_spec.resource_request.gpu_count
    # `live` includes workers that have not been scheduled yet, so they are not in the
    # sum above; clamping keeps that from reading as negative usage by other people.
    return max(0, total - live * worker.gpus)


def _capacity_target(
    beaker: Any,
    worker: "WorkerConfig",
    live: int,
) -> int:
    """How many workers to aim for, sized from the organisation's own allocation.

    Returns `worker.num_workers` unchanged unless `capacity_fraction` is set and the
    stage requests GPUs. Beaker reports usage only in GPU slots, so for a CPU-only stage
    the reading describes a resource the stage never occupies and holding the configured
    number is the honest behaviour.

    Two bounds apply, and the tighter wins. `capacity_fraction` of the allocation is the
    share this run takes when nothing else is running, leaving the rest for colleagues
    who have not launched yet. The allocation minus what colleagues hold right now is
    what is actually left. On a quiet workspace the fraction binds; on a busy one the
    remainder does.

    Cluster-wide occupancy is deliberately not consulted. The allocation is what the
    organisation may use, and other teams' jobs neither grant nor remove that
    entitlement; letting a saturated cluster shrink the pool would forfeit capacity the
    run is owed, and priority exists to resolve the contention.

    Args:
        beaker: an open Beaker client.
        worker: the pool configuration.
        live: workers currently starting or demonstrably alive.

    Returns:
        the number of workers to aim for this cycle.

    Raises:
        ValueError: if capacity sizing is on but no allocation was given.
    """
    if worker.capacity_fraction is None or worker.gpus <= 0:
        return worker.num_workers
    if worker.capacity_slots is None:
        raise ValueError(
            "worker.capacity_slots is required when worker.capacity_fraction is set: "
            "sizing against an allocation needs to know how large it is"
        )

    workspace_id = worker.capacity_workspace or DEFAULT_WORKSPACE
    try:
        workspace_id = beaker.workspace.get(workspace_id).id
        others = _allocated_slots_in_use(beaker, worker, workspace_id, live)
    except Exception:
        # Sizing blind is worse than not resizing: the ceiling may be far above what is
        # left, so guessing could dump a launch onto an allocation colleagues are using.
        # Hold the pool where it is and try again next cycle.
        logger.exception(
            "could not read allocation usage; holding the pool at %d", live
        )
        return max(live, worker.capacity_min_workers)

    share = worker.capacity_fraction * worker.capacity_slots
    remaining = worker.capacity_slots - others
    target_slots = max(0.0, min(share, float(remaining)))
    # Slots to workers: they coincide only while a worker holds one GPU.
    slots_per_worker = max(1, worker.gpus)
    target = int(target_slots / slots_per_worker)
    target = max(worker.capacity_min_workers, min(target, worker.num_workers))
    # Shrinking is immediate, growing is capped. See CAPACITY_MAX_STEP.
    capped = min(target, live + CAPACITY_MAX_STEP)
    logger.info(
        "capacity target %d worker(s): allocation %d slot(s), %d held by others, "
        "share %.2f -> %d, remaining -> %d, %d gpu(s)/worker, %d live, ceiling %d",
        capped,
        worker.capacity_slots,
        others,
        worker.capacity_fraction,
        int(share),
        remaining,
        slots_per_worker,
        live,
        worker.num_workers,
    )
    return capped


def _cluster_free_slots(beaker: Any, worker: "WorkerConfig") -> int:
    """GPU slots on `worker.cluster` that no job currently holds.

    This is the whole cluster, not the workspace's allocation: the point of backfill is
    to use what the organisation is not entitled to but nobody else is using either.
    Cordoned slots are already excluded from the reading.

    Args:
        beaker: an open Beaker client.
        worker: the pool configuration.

    Returns:
        idle slots summed over every cluster in the pool's list.
    """
    total = 0
    for name in worker.cluster:
        occupancy = beaker.cluster.get(name, include_cluster_occupancy=True)
        total += occupancy.cluster_occupancy.slot_counts.available
    return max(0, total)


def _backfill_target(
    beaker: Any,
    worker: "WorkerConfig",
    live: int,
    allocated: int,
) -> int:
    """How many unallocated workers to add on top of the allocated pool.

    Sized from what the cluster has idle right now, because that is the only capacity an
    unallocated job can actually get: it is admitted onto free slots and evicted when an
    allocation wants them. Reading the allocation instead would size the pool from an
    entitlement these workers deliberately do not use.

    The pool's own backfill workers are added back before the fraction is applied. They
    are sitting on the very slots being counted, so without this the reading falls by
    exactly what was launched, the target collapses to zero, the surplus drain retires
    them, the slots free up and the whole thing repeats -- a launch-and-kill loop that
    costs a block every time round. Adding them back makes the steady state a fixed
    point: once the idle slots are taken, the target equals the pool already holding
    them and nothing moves.

    Holds the pool where it is when the reading fails, rather than reporting zero.
    Zero reads as "give it all back" to the surplus drain, so a momentary Beaker hiccup
    would retire every backfill worker mid-block.

    Args:
        beaker: an open Beaker client.
        worker: the pool configuration.
        live: workers currently starting or demonstrably alive, both kinds.
        allocated: this cycle's allocated-pool target.

    Returns:
        workers to run at backfill priority, possibly zero.
    """
    if worker.backfill_fraction is None or worker.gpus <= 0:
        return 0
    slots_per_worker = max(1, worker.gpus)
    # Anything above the allocated target is already running as backfill.
    held = max(0, live - allocated)
    try:
        free = _cluster_free_slots(beaker, worker)
    except Exception:
        logger.exception(
            "could not read cluster occupancy; holding backfill at %d", held
        )
        return min(held, worker.backfill_max_workers)

    target = int((free + held * slots_per_worker) * worker.backfill_fraction)
    target //= slots_per_worker
    target = min(target, worker.backfill_max_workers)
    logger.info(
        "backfill target %d worker(s): %d idle slot(s) on %s plus %d held by this pool, "
        "fraction %.2f, %d gpu(s)/worker, ceiling %d",
        target,
        free,
        ",".join(worker.cluster),
        held,
        worker.backfill_fraction,
        slots_per_worker,
        worker.backfill_max_workers,
    )
    return target


def _drain_path(store_path: str, queue_name: str) -> str:
    """Where the drain list for one run lives.

    Beside the store, never inside it. Everything under a `.zarr` prefix belongs to
    that store's key space: a stray file there shows up in store listings and rides
    along in any copy of the store. The drain list is run bookkeeping, so it sits next
    to the store alongside the `completed_*` marker directories.

    Keyed by queue rather than by store: several runs share one store, and each has to
    retire its own workers.

    Args:
        store_path: the run's GeoZarr store, e.g. ".../run/embeddings.zarr".
        queue_name: the Beaker queue name, e.g. "user/my-queue".

    Returns:
        the path to publish the drain list at.
    """
    slug = queue_name.replace("/", "-")
    return str(UPath(store_path.rstrip("/")).parent / "worker_drain" / f"{slug}.json")


def _publish_drain_list(drain_path: str, worker_names: list[str]) -> None:
    """Publish the workers that should retire once their current job is done.

    Written every cycle, including empty, because the list is what workers read to
    decide whether to stop. Leaving a stale non-empty list in place would keep retiring
    workers after the surplus was gone.

    Args:
        drain_path: where to publish.
        worker_names: experiment names of the workers to retire.
    """
    upath = UPath(drain_path)
    upath.parent.mkdir(parents=True, exist_ok=True)
    with upath.open("w") as f:
        json.dump({"written": time.time(), "workers": worker_names}, f)


def _release_surplus_workers(
    beaker: Any,
    workspace: Any,
    name_prefix: str,
    surplus: int,
    drain_path: str | None = None,
) -> int:
    """Give capacity back when the pool is over its target.

    Two stages, because workers differ in what stopping them costs. A worker that has
    not started holds nothing, so cancelling it is free and instant; that is the first
    stage, and it targets exactly the case that matters most, a burst of queued
    launches that would otherwise land later and overshoot all at once.

    A running worker owns a claimed job and there is no intra-job checkpointing, so
    cancelling one throws away up to a whole job. Instead the rest of the surplus is
    published to a drain list, and each named worker stops itself after its current job
    and before claiming the next. That costs no work and hands the GPU back within one
    job, which is what keeps the pool responsive to demand from colleagues rather than
    only when our own queue empties.

    The idle timeout is not a fallback for this. It only fires when no work is
    available, and the queue is kept topped up to `target_pending` all run, so a
    surplus worker mid-run never sees an empty queue and would hold its slot
    indefinitely.

    Args:
        beaker: an open Beaker client.
        workspace: the workspace to search.
        name_prefix: the prefix from `worker_name_prefix`.
        surplus: how many workers over target the pool is; may be zero, which still
            republishes an empty drain list.
        drain_path: where to publish the drain list, or None to only cancel.

    Returns:
        the number of workers cancelled, which is the capacity freed immediately.
        Drained workers are not counted: they are still working.
    """
    surplus = max(0, surplus)
    workers = [
        workload
        for workload in beaker.workload.list(
            workspace=workspace,
            author=beaker.user.get(),
            finalized=False,
            workload_type=BeakerWorkloadType.experiment,
            limit=WORKER_LIST_LIMIT,
        )
        if getattr(getattr(workload, "experiment", None), "name", "").startswith(
            name_prefix
        )
    ]

    # Newest first. For the cancel stage that is the least likely to be moments away
    # from starting and the most likely to be the tail of a launch burst that has not
    # landed yet. For the drain stage the order is close to arbitrary, since when a
    # workload was created says nothing about how far into its current job a worker is:
    # it has run many jobs by then. Newest first is kept so that the most recently
    # added capacity is the first given back.
    def _created(workload: Any) -> int:
        experiment = getattr(workload, "experiment", None)
        created = getattr(experiment, "created", None)
        return getattr(created, "seconds", 0) or 0

    workers.sort(key=_created, reverse=True)
    waiting = [
        w for w in workers if getattr(w, "status", None) in PENDING_WORKLOAD_STATUSES
    ]
    running = [
        w for w in workers if getattr(w, "status", None) in DRAINABLE_WORKLOAD_STATUSES
    ]

    cancelled = 0
    doomed = waiting[:surplus]
    if doomed:
        try:
            beaker.workload.cancel(*doomed)
            cancelled = len(doomed)
            logger.info(
                "cancelled %d queued worker(s) to stay within the capacity target "
                "(%d over, %d were still waiting to start)",
                cancelled,
                surplus,
                len(waiting),
            )
        except Exception:
            # Not fatal: the pool is over target, not broken, and the next cycle tries
            # again. Launching is already capped by the target, so nothing compounds.
            logger.exception("could not cancel %d surplus worker(s)", len(doomed))

    if drain_path is None:
        return cancelled

    draining = [w.experiment.name for w in running[: surplus - cancelled]]
    try:
        _publish_drain_list(drain_path, draining)
    except Exception:
        # The pool stays over target for a cycle; nothing is lost and the next cycle
        # republishes.
        logger.exception("could not publish the drain list to %s", drain_path)
        return cancelled
    if draining:
        logger.info(
            "asked %d running worker(s) to retire after their current job "
            "(%d over target, %d cancelled while queued)",
            len(draining),
            surplus,
            cancelled,
        )
    return cancelled


def _count_workers(
    beaker: Any,
    workspace: Any,
    name_prefix: str,
    queue: Any = None,
    now: float | None = None,
) -> int:
    """Count this run's workers that can still claim work.

    Two things have to be counted, and neither alone is enough. A worker still pulling
    its 17 GB image has not registered with the queue yet, so counting registrations
    alone undercounts for the whole of container start and the pool overshoots by
    however many cycles that takes. A worker whose process or channel thread has died
    stays unfinalized to Beaker indefinitely, so counting workloads alone overcounts
    and, because launches are capped by outstanding work, eventually strands the run:
    once the dead count reaches the outstanding count nothing is launched and the last
    jobs are never claimed.

    So: registrations whose heartbeat is fresh, plus workloads that have not started
    running yet and therefore cannot have registered. Registration happens within
    seconds of the container starting, so a running worker is already counted by its
    heartbeat and must not be counted again -- doing so inflated the count by a whole
    launch batch, 128 on one measured scale-up, and the pool then refuses to backfill
    until those age out of the window. Both inputs come from listings, so this costs
    two calls rather than one per worker.

    Args:
        beaker: an open Beaker client.
        workspace: the workspace to search.
        name_prefix: the prefix from `worker_name_prefix`.
        queue: the queue whose worker registrations to read. Without it this falls back
            to counting unfinalized workloads, which is the old behaviour.
        now: current unix time, passed in so this stays testable.

    Returns:
        the number of workers that are starting or demonstrably alive.
    """
    workloads = [
        workload
        for workload in beaker.workload.list(
            workspace=workspace,
            author=beaker.user.get(),
            finalized=False,
            workload_type=BeakerWorkloadType.experiment,
            limit=WORKER_LIST_LIMIT,
        )
        if getattr(getattr(workload, "experiment", None), "name", "").startswith(
            name_prefix
        )
    ]
    if queue is None:
        return len(workloads)

    now = time.time() if now is None else now
    starting = 0
    for workload in workloads:
        status = getattr(workload, "status", None)
        if status in RUNNING_WORKLOAD_STATUSES:
            # Already running, so it has registered and its heartbeat speaks for it.
            continue
        if status in PENDING_WORKLOAD_STATUSES or status is None:
            # Created but not running: waiting on the cluster, or on a 17 GB image
            # pull. Either way nothing else is counting it, and it will run eventually.
            starting += 1

    fresh = 0
    for worker in beaker.queue.list_workers(queue):
        heartbeat = getattr(worker, "heartbeat", None)
        if heartbeat is None or not heartbeat.seconds:
            continue
        if now - heartbeat.seconds < WORKER_HEARTBEAT_STALE_SECONDS:
            fresh += 1
    return starting + fresh


def _run_cycle(
    config: SuperviseConfig,
    result: Any,
    launched: Any = None,
    stats: Any = None,
) -> None:
    """Run one supervision cycle, reporting the remaining job count via `result`.

    This runs in a child process so the parent can kill it if a Beaker RPC hangs. It
    sets `result.value` to the number of jobs still lacking a completion marker, or
    leaves it at `_NO_RESULT` if it does not get that far.

    Args:
        config: the run configuration.
        result: shared int the remaining-job count is written to.
        stats: shared int array (pending, claimed, completed, rejected, workers,
            worker_target) the
            cycle's queue and pool counts are written to. The parent logs them, since
            a cycle killed for overrunning its budget still has numbers worth keeping.
        launched: shared int the number of workers launched is written to, so the
            parent can carry it into the next cycle's liveness count.
    """
    queue_name = config.queue_name
    # Only meaningful when the pool is sized to the cluster: a static pool has no
    # surplus to retire, and without a publisher the list would never be cleared.
    drain_path = (
        _drain_path(config.store_path, queue_name)
        if config.worker.capacity_fraction is not None
        else None
    )

    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        queue = beaker.queue.get(queue_name)
        entries = list(beaker.queue.list_entries(queue))
        counts: dict[str, int] = {}
        for entry in entries:
            name = _state_name(entry)
            counts[name] = counts.get(name, 0) + 1
        now = time.time()
        in_flight = _in_flight_job_keys(entries, now, config.cycle.claim_stale_seconds)
        workspace = beaker.workspace.get(DEFAULT_WORKSPACE)
        live = _count_workers(
            beaker,
            workspace,
            worker_name_prefix(queue_name),
            queue=queue,
            now=now,
        )
        # Resolved here rather than from config because capacity sizing needs both the
        # live count and a Beaker client. Static runs get config.worker.num_workers.
        allocated_target = _capacity_target(beaker, config.worker, live)
        # Backfill rides on top of the allocation rather than replacing part of it, so
        # the run keeps every slot it is entitled to and adds whatever the cluster is
        # wasting. Re-capped because each half is capped separately and the sum is not.
        num_workers = min(
            allocated_target
            + _backfill_target(beaker, config.worker, live, allocated_target),
            live + CAPACITY_MAX_STEP,
        )
        # Staying inside an allocation means giving capacity back, not just declining
        # to take more. Only done when capacity sizing is on: a static pool sits a few
        # over target routinely, because a worker that has exited stays unfinalized for
        # a while, and cancelling on that would fight its own bookkeeping.
        #
        # Runs every cycle rather than only when over target, because the drain list
        # has to be cleared once the surplus is gone, not just written when it appears.
        if config.worker.capacity_fraction is not None:
            live -= _release_surplus_workers(
                beaker,
                workspace,
                worker_name_prefix(queue_name),
                live - num_workers,
                drain_path=drain_path,
            )
    # How deep to keep the queue. The default assumes long jobs. A stage of short jobs
    # needs much more: a worker drains its few entries and then idles until the next
    # cycle, making the cycle interval the throughput ceiling.
    target_pending = num_workers * config.cycle.pending_per_worker
    pending = counts.get("PENDING", 0)
    if stats is not None:
        stats[0] = pending
        stats[1] = counts.get("CLAIMED", 0)
        stats[2] = counts.get("COMPLETED", 0)
        stats[3] = counts.get("REJECTED", 0)
        stats[4] = live
        stats[5] = num_workers
    logger.info("queue=%s workers=%d", counts, live)

    # Recompute what is left directly from the completion markers. This doubles as the
    # completion check, so it runs every cycle rather than only when the queue drains.
    years = config.years
    stage = config.stage
    remaining: list[list[str]] = []
    if stage == STAGE_RENDER_WEB_PCA:
        # Enumerated from the UTM PCA store's own object keys: the destination grid is
        # global and almost entirely empty, so listing what exists beats probing it.
        remaining.extend(
            get_web_jobs(
                source_store_path=require_config(
                    config.pca.store_path, "pca.store_path", stage
                ),
                web_store_path=require_config(
                    config.pca.web_store_path, "pca.web_store_path", stage
                ),
                completed_path=require_config(
                    config.pca.web_completed_path, "pca.web_completed_path", stage
                ),
                zoom=require_config(config.pca.web_zoom, "pca.web_zoom", stage),
                years=years,
                zone_numbers=require_config(
                    config.aoi.zone_numbers, "aoi.zone_numbers", stage
                ),
                base_zoom=config.pca.web_base_zoom,
                source_url=config.pca.store_url,
            )
        )
    elif stage == STAGE_RENDER_UTM_PCA:
        # Enumerated from the predict markers, so this needs no model settings and no
        # land or wedge filtering: the markers already name what exists.
        remaining.extend(
            get_render_jobs(
                store_path=config.store_path,
                pca_store_path=require_config(
                    config.pca.store_path, "pca.store_path", stage
                ),
                artifact_path=require_config(
                    config.pca.artifact_path, "pca.artifact_path", stage
                ),
                source_completed_paths=[
                    config.completed_path_template.format(year=year) for year in years
                ],
                completed_path=require_config(
                    config.pca.completed_path, "pca.completed_path", stage
                ),
                patch_size=config.model.patch_size,
                max_level=config.pca.max_level,
            )
        )
    else:
        # The slot a year occupies is a property of the store, not of this run's --years.
        # Deriving it from --years puts a single-year run into slot 0 whatever year it
        # names, so a 2025 run against a store built for 2017-2025 would silently land
        # in 2017's slot with the markers still reading completed_2025.
        store_years = get_store_years(config.store_path)
        missing = [year for year in years if year not in store_years]
        if missing:
            raise ValueError(
                f"store {config.store_path} has years {store_years}, which do not "
                f"include {missing}; init_store fixes the time axis at creation, so "
                "the store must be created with every year the run will write"
            )
        for year in years:
            remaining.extend(
                get_jobs(
                    inputs=config.inputs,
                    timestamp=datetime(year, 1, 1, tzinfo=UTC),
                    store_path=config.store_path,
                    completed_path=config.completed_path_template.format(year=year),
                    checkpoint_path=config.model.checkpoint_path,
                    time_index=store_years.index(year),
                    patch_size=config.model.patch_size,
                    window_size=config.model.window_size,
                    overlap_size=config.model.overlap_size,
                    compile_model=config.model.compile_model,
                    batch_size=config.model.batch_size,
                    epsg_code=config.aoi.epsg_code,
                    wgs84_bounds=config.aoi.wgs84_bounds,
                    geojson_fname=config.aoi.geojson_fname,
                    job_size=config.aoi.job_size,
                    enumeration_cache_dir=config.cycle.enumeration_cache_dir,
                )
            )
    result.value = len(remaining)
    logger.info("%d job(s) still without a completion marker", len(remaining))
    if not remaining:
        return

    # Top the queue up only when it is shallow, and only with work that is not already
    # queued or actively claimed. The shallow-queue guard bounds duplication; this bounds
    # it much harder, because a job can be re-offered many times over a long run.
    fresh = [job for job in remaining if tuple(job) not in in_flight]
    if pending < target_pending:
        random.shuffle(fresh)
        batch = fresh[: target_pending - pending]
        if batch:
            rslp.common.worker.write_jobs(
                queue_name, "large_scale_embeddings", stage, batch
            )
        logger.info(
            "enqueued %d job(s) (pending was %d; %d of %d already in flight)",
            len(batch),
            pending,
            len(remaining) - len(fresh),
            len(remaining),
        )

    # A worker holds one job at a time, so launching more than there are outstanding
    # jobs is pure churn: each surplus worker starts, finds nothing to claim and exits.
    worker_target = min(num_workers, len(remaining))
    if live < worker_target:

        def launch(count: int, priority: str, min_runtime: timedelta) -> None:
            """Launch `count` workers differing only in what may evict them."""
            rslp.common.worker.launch_workers(
                image_name=config.worker.image_name,
                queue_name=queue_name,
                num_workers=count,
                cluster=config.worker.cluster,
                gpus=config.worker.gpus,
                shared_memory=config.worker.shared_memory,
                priority=BeakerJobPriority[priority],
                min_runtime=min_runtime,
                weka_mounts=[
                    WekaMount(
                        bucket_name=config.worker.weka_bucket,
                        mount_path=config.worker.weka_mount_path,
                    )
                ],
                idle_timeout=config.worker.idle_seconds,
                drain_path=drain_path,
                name_prefix=worker_name_prefix(queue_name),
                extra_env_vars={
                    "OEDATASETS_API_URL": config.worker.datasets_api_url,
                    **(config.worker.env_vars or {}),
                },
                extra_env_secrets={
                    "DATASETS_API_TOKEN": config.worker.datasets_token_secret,
                    "AWS_ACCESS_KEY_ID": config.worker.aws_key_id_secret,
                    "AWS_SECRET_ACCESS_KEY": config.worker.aws_secret_key_secret,
                },
            )

        # Spend the allocation before taking anything on loan. An allocated worker keeps
        # its slot until the block is done; a backfill worker can lose it at any moment,
        # so a slot the run is entitled to is worth more held allocated than unallocated.
        just_launched = worker_target - live
        allocated = max(0, min(allocated_target, worker_target) - live)
        allocated = min(allocated, just_launched)
        if allocated:
            launch(
                allocated,
                config.worker.priority,
                rslp.common.worker.DEFAULT_WORKER_MIN_RUNTIME,
            )
        if just_launched - allocated:
            launch(
                just_launched - allocated,
                config.worker.backfill_priority,
                BACKFILL_MIN_RUNTIME,
            )
        if launched is not None:
            launched.value = just_launched
        logger.info(
            "launched %d worker(s), %d allocated and %d backfill (target %d, "
            "%d existing, %d outstanding job(s))",
            just_launched,
            allocated,
            just_launched - allocated,
            worker_target,
            live,
            len(remaining),
        )


def launch_supervisor(
    image_name: str,
    cluster: list[str],
    supervise_args: list[str],
    priority: str = "urgent",
    task_name: str = "geozarr-supervisor",
    cpu_count: float = 2,
    memory: str = "8GiB",
    gpu_count: int = 0,
    min_runtime: timedelta = DEFAULT_SUPERVISOR_MIN_RUNTIME,
    auto_resume: bool = True,
) -> str:
    """Launch `supervise` as a CPU-only Beaker job so a run outlives any one session.

    The supervisor must not depend on a workstation: it needs to keep refilling the
    queue and the worker pool for the whole run. It needs no GPU. The base env vars
    already supply BEAKER_TOKEN (to manage the queue and launch workers) and
    GOOGLE_APPLICATION_CREDENTIALS (to read completion markers), so no extra wiring
    is required.

    Args:
        image_name: the Beaker image to run (must contain this workflow).
        cluster: clusters to schedule on; a CPU cluster is appropriate.
        supervise_args: arguments forwarded verbatim to the `supervise` workflow, e.g.
            ["--inputs", "S2", "--years", "[2024, 2025]", ...]. Passed through rather
            than re-declared so this launcher never drifts from supervise()'s options.
        priority: Beaker priority. With no supervisor nothing refills the queue and no
            preempted worker's job is re-offered, so every later preemption becomes
            permanent loss rather than a retry. Defaults to urgent for that reason.
        task_name: name for the Beaker experiment.
        cpu_count: CPUs to request.
        memory: memory to request.
        gpu_count: GPUs to request. The supervisor needs none, but on saturated
            GPU clusters a 0-GPU task may never be scheduled (slots are counted
            in GPUs), so requesting 1 is sometimes the only way to place it
            alongside the workers. Wasteful; prefer 0 where it schedules.
        min_runtime: how long the scheduler should let the supervisor run before it may
            be preempted. Above five minutes the job counts as allocated; at or below it
            it is unallocated and yields to any allocated work.
        auto_resume: whether Beaker replaces the job when it is preempted. Leave this on:
            without it a preempted supervisor is gone for good, and with it gone nothing
            refills the queue or replaces a worker. Restarting is safe, since the
            supervisor re-reads completion markers and holds no state.

    Returns:
        the created Beaker experiment's ID.
    """
    spec = BeakerExperimentSpec.new(
        budget=DEFAULT_BUDGET,
        description="large_scale_embeddings supervisor",
        beaker_image=image_name,
        priority=BeakerJobPriority[priority],
        command=["python", "-m", "rslp.main"],
        arguments=["large_scale_embeddings", "supervise", *supervise_args],
        constraints=BeakerConstraints(cluster=cluster),
        min_runtime=min_runtime,
        auto_resume=auto_resume,
        datasets=[create_gcp_credentials_mount()],
        env_vars=get_base_env_vars(),
        resources=BeakerTaskResources(
            cpu_count=cpu_count, memory=memory, gpu_count=gpu_count
        ),
    )
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        workload = beaker.experiment.create(name=task_name, spec=spec)
    experiment_id = getattr(workload, "id", None) or str(workload)
    logger.info("launched supervisor experiment %s on %s", experiment_id, cluster)
    return experiment_id


def supervise(
    inputs: EmbeddingInputs,
    years: list[int],
    store_path: str,
    completed_path_template: str,
    queue_name: str,
    model: ModelConfig,
    worker: WorkerConfig,
    stage: str = STAGE_PREDICT,
    cycle: CycleConfig | None = None,
    aoi: AoiConfig | None = None,
    pca: PcaConfig | None = None,
) -> None:
    """Refill the queue and worker pool each cycle until every tile has a marker.

    Args:
        inputs: which input variant to embed.
        years: the annual reference years to cover.
        store_path: the GeoZarr store to write into.
        completed_path_template: marker directory containing ``{year}``.
        queue_name: the Beaker queue to enqueue work on.
        model: the encoder and how it is run. See `ModelConfig`.
        worker: the Beaker worker pool. See `WorkerConfig`.
        stage: which stage to drive; one of `STAGES`.
        cycle: loop pacing. See `CycleConfig`.
        aoi: the ground to cover and how to cut it up. See `AoiConfig`.
        pca: paths for the render stages. See `PcaConfig`. Required by
            `STAGE_RENDER_UTM_PCA` and `STAGE_RENDER_WEB_PCA`.

    Raises:
        ValueError: if the stage is unknown, if a render stage is missing a path it
            needs, or if the first cycle enumerates no work at all.
    """
    cycle = cycle or CycleConfig()
    config = SuperviseConfig(
        inputs=inputs,
        years=years,
        store_path=store_path,
        completed_path_template=completed_path_template,
        queue_name=queue_name,
        stage=stage,
        model=model,
        worker=worker,
        cycle=cycle,
        aoi=aoi or AoiConfig(),
        pca=pca or PcaConfig(),
    )

    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if stage == STAGE_RENDER_UTM_PCA:
        missing = [
            name
            for name, value in (
                ("pca.artifact_path", config.pca.artifact_path),
                ("pca.store_path", config.pca.store_path),
                ("pca.completed_path", config.pca.completed_path),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"stage {STAGE_RENDER_UTM_PCA} requires {', '.join(missing)}; fit the "
                "basis with the fit_pca workflow first"
            )

    # "spawn" rather than the default fork: the child creates gRPC channels, and
    # forking a process that may already hold them is a known source of hangs.
    ctx = multiprocessing.get_context("spawn")
    seen_work = False
    consecutive_failures = 0
    cycle_number = 0
    total = None
    metrics = _Metrics(
        cycle.wandb_project,
        f"{config.stage}-{config.queue_name.split('/')[-1]}",
        {
            "stage": config.stage,
            "queue": config.queue_name,
            "years": config.years,
            "store_path": config.store_path,
            "num_workers": config.worker.num_workers,
            "job_size": config.aoi.job_size,
            "cluster": config.worker.cluster,
            "image": config.worker.image_name,
        },
    )

    while cycle.max_cycles is None or cycle_number < cycle.max_cycles:
        cycle_number += 1
        # Typeshed types Value() as SynchronizedBase, which has no .value; the "i"
        # type code makes it a Synchronized[int].
        result: Synchronized[int] = ctx.Value("i", _NO_RESULT)  # type: ignore[assignment]
        launched: Synchronized[int] = ctx.Value("i", 0)  # type: ignore[assignment]
        stats = ctx.Array("i", 6)
        proc = ctx.Process(target=_run_cycle, args=(config, result, launched, stats))
        started = time.time()
        proc.start()
        proc.join(cycle.budget_seconds)
        if proc.is_alive():
            logger.warning(
                "cycle %d exceeded its %ds budget; killing it (likely a hung Beaker "
                "RPC) and continuing",
                cycle_number,
                cycle.budget_seconds,
            )
            proc.terminate()
            proc.join(30)
            if proc.is_alive():
                proc.kill()
                proc.join(30)
        elapsed = int(time.time() - started)
        remaining = result.value
        pending, claimed, completed, rejected, live, worker_target = (
            stats[0],
            stats[1],
            stats[2],
            stats[3],
            stats[4],
            stats[5],
        )
        # The denominator is the first cycle's count: it is the only cycle that has
        # seen the whole job list, so later cycles can be expressed as a fraction.
        if remaining != _NO_RESULT and total is None:
            total = remaining + completed
        metrics.log(
            cycle_number,
            {
                "cycle/seconds": elapsed,
                "cycle/killed": int(remaining == _NO_RESULT),
                "jobs/remaining": None if remaining == _NO_RESULT else remaining,
                "jobs/done": None
                if total is None or remaining == _NO_RESULT
                else total - remaining,
                "jobs/fraction_done": None
                if not total or remaining == _NO_RESULT
                else (total - remaining) / total,
                "queue/pending": pending,
                "queue/claimed": claimed,
                "queue/completed": completed,
                "queue/rejected": rejected,
                "pool/workers": live,
                "pool/launched": launched.value,
                # Against the target the cycle actually resolved, which is not
                # config.worker.num_workers once capacity sizing is on: there that
                # value is a ceiling, and differencing against it would report a
                # shortfall for a pool that is exactly the size it meant to be.
                "pool/shortfall": worker_target - live,
                "pool/target": worker_target,
            },
        )

        if remaining == _NO_RESULT:
            # Killed, crashed, or otherwise did not report. Nothing to conclude about
            # the run's state from one of these, so retry -- but not forever, since a
            # deterministic failure looks exactly the same and would otherwise spin.
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_CYCLE_FAILURES:
                raise RuntimeError(
                    f"{consecutive_failures} consecutive cycles failed to report a "
                    f"result (last exit code {proc.exitcode}); treating this as a "
                    "permanent error rather than retrying. Check the traceback above: "
                    "a missing aoi.geojson_fname, an unreadable store_path or an "
                    "expired "
                    "credential all fail this way on every cycle."
                )
            logger.warning(
                "cycle %d did not report a result after %ds (exit code %s); retrying "
                "(%d/%d consecutive failures)",
                cycle_number,
                elapsed,
                proc.exitcode,
                consecutive_failures,
                MAX_CONSECUTIVE_CYCLE_FAILURES,
            )
        elif remaining == 0:
            consecutive_failures = 0
            if not seen_work and not _any_completion_markers(config):
                # Nothing on the first cycle usually means the AOI filters, bounds
                # or zone selection exclude everything, not that the run is done. A
                # resumed run with existing markers is the legitimate exception.
                raise ValueError(
                    "enumerated no jobs at all on the first cycle; check "
                    "aoi.geojson_fname/aoi.wgs84_bounds/aoi.epsg_code and that "
                    "store_path and completed_path_template are correct"
                )
            logger.info("all tiles have completion markers; run complete")
            metrics.finish()
            return
        else:
            consecutive_failures = 0
            seen_work = True
            logger.info(
                "cycle %d done in %ds; %d job(s) remaining",
                cycle_number,
                elapsed,
                remaining,
            )

        time.sleep(cycle.seconds)

    metrics.finish()
    logger.info("reached max_cycles=%d; exiting", cycle_number)
