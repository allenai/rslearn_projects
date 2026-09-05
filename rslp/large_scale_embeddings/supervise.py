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

import multiprocessing
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from multiprocessing.sharedctypes import Synchronized
from typing import Any

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

# GDAL environment every worker needs, merged in below so it cannot be forgotten.
#
# GS_USER_PROJECT is required for the Landsat reads: rasterio_session_for_path honours
# requester_pays only for S3, so the requester-pays USGS mirror needs GDAL handed a
# billing project this way. Omitting it fails with an HTTP 400 that names no variable.
# This lives here rather than in full_run because supervise is what launches workers.
DEFAULT_WORKER_ENV_VARS = {
    "GS_USER_PROJECT": "earthsystem-dev-c3po",
}


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

    def __post_init__(self) -> None:
        """Merge the GDAL defaults so a caller cannot drop them by passing env_vars."""
        self.env_vars = {**DEFAULT_WORKER_ENV_VARS, **(self.env_vars or {})}


@dataclass
class CycleConfig:
    """Pacing of the supervision loop."""

    seconds: int = 180
    budget_seconds: int = DEFAULT_CYCLE_BUDGET_SECONDS
    claim_stale_seconds: int = DEFAULT_CLAIM_STALE_SECONDS
    pending_per_worker: int = PENDING_PER_WORKER
    max_cycles: int | None = None


@dataclass
class AoiConfig:
    """Which ground the run covers, and how it is cut into jobs."""

    job_size: int = 8192
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
        return [config.pca.completed_path]
    return [
        config.completed_path_template.format(year=year) for year in config.years
    ]


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


def _count_workers(beaker: Any, workspace: Any, name_prefix: str) -> int:
    """Count this run's workers that exist and have not finalized.

    Beaker knows a worker exists the moment its experiment is created, so this counts
    one that is still pulling its image just as it counts one mid-job. Deriving the
    count from queue heartbeats instead undercounts for the whole of container start,
    which is minutes for a large image, and the pool overshoots by however many cycles
    that takes.

    Args:
        beaker: an open Beaker client.
        workspace: the workspace to search.
        name_prefix: the prefix from `worker_name_prefix`.

    Returns:
        the number of live or starting workers belonging to this run.
    """
    return sum(
        1
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
    )


def _run_cycle(config: SuperviseConfig, result: Any, launched: Any = None) -> None:
    """Run one supervision cycle, reporting the remaining job count via `result`.

    This runs in a child process so the parent can kill it if a Beaker RPC hangs. It
    sets `result.value` to the number of jobs still lacking a completion marker, or
    leaves it at `_NO_RESULT` if it does not get that far.

    Args:
        config: the run configuration.
        result: shared int the remaining-job count is written to.
        launched: shared int the number of workers launched is written to, so the
            parent can carry it into the next cycle's liveness count.
    """
    queue_name = config.queue_name
    num_workers = config.worker.num_workers
    # How deep to keep the queue. The default assumes long jobs. A stage of short jobs
    # needs much more: a worker drains its few entries and then idles until the next
    # cycle, making the cycle interval the throughput ceiling.
    target_pending = num_workers * config.cycle.pending_per_worker

    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        queue = beaker.queue.get(queue_name)
        entries = list(beaker.queue.list_entries(queue))
        counts: dict[str, int] = {}
        for entry in entries:
            name = _state_name(entry)
            counts[name] = counts.get(name, 0) + 1
        now = time.time()
        in_flight = _in_flight_job_keys(entries, now, config.cycle.claim_stale_seconds)
        live = _count_workers(
            beaker,
            beaker.workspace.get(DEFAULT_WORKSPACE),
            worker_name_prefix(queue_name),
        )
    pending = counts.get("PENDING", 0)
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
                source_store_path=config.pca.store_path,
                web_store_path=config.pca.web_store_path,
                completed_path=config.pca.web_completed_path,
                zoom=config.pca.web_zoom,
                years=years,
                zone_numbers=config.aoi.zone_numbers,
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
                pca_store_path=config.pca.store_path,
                artifact_path=config.pca.artifact_path,
                source_completed_paths=[
                    config.completed_path_template.format(year=year) for year in years
                ],
                completed_path=config.pca.completed_path,
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
        rslp.common.worker.launch_workers(
            image_name=config.worker.image_name,
            queue_name=queue_name,
            num_workers=worker_target - live,
            cluster=config.worker.cluster,
            gpus=config.worker.gpus,
            shared_memory=config.worker.shared_memory,
            priority=BeakerJobPriority[config.worker.priority],
            weka_mounts=[
                WekaMount(
                    bucket_name=config.worker.weka_bucket,
                    mount_path=config.worker.weka_mount_path,
                )
            ],
            idle_timeout=config.worker.idle_seconds,
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
        just_launched = worker_target - live
        if launched is not None:
            launched.value = just_launched
        logger.info(
            "launched %d worker(s) (target %d, %d existing, %d outstanding job(s))",
            just_launched,
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
    preemptible: bool = True,
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
        preemptible: whether the supervisor itself may be preempted. Defaults to True,
            which reads backwards but is the safe setting: Beaker replaces a preempted
            preemptible task and abandons a non-preemptible one, so False makes
            preemption terminal. Non-preemptible buys only a minimum-runtime floor.
            Restarting is safe, since the supervisor holds no state a restart loses.

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
        preemptible=preemptible,
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

    while cycle.max_cycles is None or cycle_number < cycle.max_cycles:
        cycle_number += 1
        # Typeshed types Value() as SynchronizedBase, which has no .value; the "i"
        # type code makes it a Synchronized[int].
        result: Synchronized[int] = ctx.Value("i", _NO_RESULT)  # type: ignore[assignment]
        launched: Synchronized[int] = ctx.Value("i", 0)  # type: ignore[assignment]
        proc = ctx.Process(target=_run_cycle, args=(config, result, launched))
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

    logger.info("reached max_cycles=%d; exiting", cycle_number)
