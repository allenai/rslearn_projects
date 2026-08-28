"""Keep an embedding run making progress on preemptible workers.

Workers are preemptible and the GPU clusters are routinely saturated, so a long run
loses workers constantly. Two existing properties make that survivable:

1. Every job is idempotent -- ``predict_pipeline`` returns immediately when the tile's
   completion marker already exists -- so re-enqueuing work is cheap and safe.
2. ``get_jobs`` derives the remaining work from those markers, so "what is left" never
   has to be tracked separately.

Two design points, both learned the hard way:

**Keep the queue shallow.** A Beaker queue entry claimed by a worker that then dies is
not released back to the queue. Entries were still CLAIMED 5 hours after being claimed,
with no worker alive for the last 1.4 of those, and the queue API has no call to release
one. They are not immortal -- ``status.expiry`` is set from ``expires_in_sec`` (7 days
by default), so they eventually age out -- but a week is far longer than any job, so
within a run that work is simply lost. Note also ``max_claimed_entries=1``: a dead
worker's claim permanently occupies that entry's only claim slot. (``wait_timeout`` on
the queue is unrelated; it bounds how long a worker waits for work to appear.)

Enqueuing a whole run up front therefore bleeds work steadily (one run accumulated 327
orphaned entries). This enqueues only a small buffer and refills it from the markers,
which bounds the loss to about one entry per worker death.

**Run each cycle in a child process.** The Beaker client has no RPC timeout, and a
hung gRPC call cannot be interrupted by ``signal.alarm`` because the C core does not
yield to the interpreter between bytecodes. An in-process watchdog therefore does not
work -- two earlier attempts silently stalled for ~10 hours each, which stopped both
queue refills and worker top-up. Only a process-level kill bounds it, so every cycle
runs in a spawned child that the parent will terminate if it overruns its budget.
"""

import multiprocessing
import random
import time
from datetime import UTC, datetime
from multiprocessing.sharedctypes import Synchronized
from typing import Any

from rslp.log_utils import get_logger

from .predict_pipeline import EmbeddingInputs
from .zarr_store import DEFAULT_PCA_MAX_LEVEL

logger = get_logger(__name__)

# The stages this supervisor can drive. Both are idempotent and marker-driven, so the
# same shallow-queue and worker-top-up loop works for either; only how remaining work is
# enumerated and which workflow the entries name differ.
STAGE_PREDICT = "predict"
STAGE_RENDER_PCA = "render_pca"
STAGES = (STAGE_PREDICT, STAGE_RENDER_PCA)

# Pending entries to keep per worker: enough that no worker idles waiting for work,
# few enough that entries orphaned by dying workers stay a rounding error.
PENDING_PER_WORKER = 3

# A cycle that outruns this is assumed wedged (almost always a hung Beaker RPC) and
# gets killed. Cycles normally take well under a minute.
DEFAULT_CYCLE_BUDGET_SECONDS = 600

# Sentinel for "the child did not report a result" (timed out, crashed, or killed).
_NO_RESULT = -1

# Consecutive failed cycles tolerated before giving up.
#
# A cycle reports nothing when it was killed for exceeding its budget, or when it
# crashed. The first is transient and worth retrying; the second may not be. A missing
# geojson, a bad store path or a revoked credential fails identically on every cycle, and
# retrying forever turns a startup mistake into a job that looks alive indefinitely while
# doing nothing. Three strikes distinguishes the two without tripping on a single hang.
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
# Named to pair with OEDATASETS_API_URL. The predecessor, LCC_DATASETS_API_TOKEN,
# carried a project prefix (land cover change) despite being the general datasets
# token; it still holds the same value, so an older invocation passing it by name
# keeps working.
DEFAULT_DATASETS_TOKEN_SECRET = "OEDATASETS_API_TOKEN"  # nosec
# Beaker secrets holding AWS credentials, mirroring what olmoearth_run's deployed runner
# injects from Secret Manager (see runner_secret_vars_google_batch_mapping). The data
# sources request every asset with requester_pays=True, so an S3 asset needs signed
# requests: without credentials GDAL raises InvalidCredentials. GCS is the preferred
# backend and S3 the fallback, so this is rarely exercised but fails hard when it is.
DEFAULT_AWS_KEY_ID_SECRET = "AWS_ACCESS_KEY_ID"  # nosec
DEFAULT_AWS_SECRET_KEY_SECRET = "AWS_SECRET_ACCESS_KEY"  # nosec


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


# How long a claim is trusted before the job is offered again.
#
# Claims in a Beaker queue are never released, so a worker that dies mid-job leaves its
# claim behind forever. Skipping every claimed job would therefore deadlock the run on the
# first worker death -- which is why the queue used to be topped up blindly. Trusting a
# claim only while it is young keeps that recovery while cutting the duplicate work that
# blind top-up generates: at Kenya scale the queue reached 56 claims against 33 live
# workers, and throughput decayed as the wasted share grew.
#
# Sized well above a job: a predict job at job_size 8192 runs ~25 min, so 90 minutes only
# re-offers work whose worker is almost certainly gone.
DEFAULT_CLAIM_STALE_SECONDS = 5400


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


def _stage_marker_paths(kwargs: dict[str, Any]) -> list[str]:
    """The completion-marker directories this stage writes into.

    Args:
        kwargs: the supervise() arguments.

    Returns:
        one path per marker directory the stage is responsible for.
    """
    if kwargs["stage"] == STAGE_RENDER_PCA:
        return [kwargs["pca_completed_path"]]
    return [
        kwargs["completed_path_template"].format(year=year) for year in kwargs["years"]
    ]


def _any_completion_markers(kwargs: dict[str, Any]) -> bool:
    """Whether this stage has already written at least one completion marker.

    A remaining count of zero has two very different causes: the stage is genuinely
    finished, or the AOI and zone filters excluded everything so nothing was ever
    enumerated. Markers on disk are what separates them, so the first-cycle guard
    consults this instead of inferring from the count alone.

    Args:
        kwargs: the supervise() arguments.

    Returns:
        True if any marker exists for this stage.
    """
    from upath import UPath

    for path in _stage_marker_paths(kwargs):
        upath = UPath(path)
        if upath.exists() and any(True for _ in upath.iterdir()):
            return True
    return False


def _run_cycle(kwargs: dict[str, Any], result: Any) -> None:
    """Run one supervision cycle, reporting the remaining job count via `result`.

    This runs in a child process so the parent can kill it if a Beaker RPC hangs. It
    sets `result.value` to the number of jobs still lacking a completion marker, or
    leaves it at `_NO_RESULT` if it does not get that far.

    Args:
        kwargs: the supervise() arguments this cycle needs.
        result: shared int the remaining-job count is written to.
    """
    from beaker import Beaker, BeakerJobPriority

    import rslp.common.worker
    from rslp.utils.beaker import DEFAULT_WORKSPACE, WekaMount

    from .render_pca import get_render_jobs
    from .write_jobs import get_jobs

    queue_name = kwargs["queue_name"]
    num_workers = kwargs["num_workers"]
    target_pending = num_workers * PENDING_PER_WORKER

    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        queue = beaker.queue.get(queue_name)
        entries = list(beaker.queue.list_entries(queue))
        counts: dict[str, int] = {}
        for entry in entries:
            name = _state_name(entry)
            counts[name] = counts.get(name, 0) + 1
        now = time.time()
        in_flight = _in_flight_job_keys(
            entries, now, kwargs.get("claim_stale_seconds", DEFAULT_CLAIM_STALE_SECONDS)
        )
        live = sum(
            1
            for worker in beaker.queue.list_workers(queue)
            if now - int(getattr(getattr(worker, "heartbeat", None), "seconds", 0) or 0)
            < kwargs["stale_seconds"]
        )
    pending = counts.get("PENDING", 0)
    logger.info("queue=%s live_workers=%d", counts, live)

    # Recompute what is left directly from the completion markers. This doubles as the
    # completion check, so it runs every cycle rather than only when the queue drains.
    years = kwargs["years"]
    stage = kwargs["stage"]
    remaining: list[list[str]] = []
    if stage == STAGE_RENDER_PCA:
        # Step 3 enumerates from step 1's markers, so it needs no model settings and no
        # land or wedge filtering: the source markers already name what exists.
        remaining.extend(
            get_render_jobs(
                store_path=kwargs["store_path"],
                pca_store_path=kwargs["pca_store_path"],
                artifact_path=kwargs["artifact_path"],
                source_completed_paths=[
                    kwargs["completed_path_template"].format(year=year)
                    for year in years
                ],
                completed_path=kwargs["pca_completed_path"],
                patch_size=kwargs["patch_size"],
                max_level=kwargs["max_level"],
            )
        )
    else:
        for year in years:
            remaining.extend(
                get_jobs(
                    inputs=kwargs["inputs"],
                    timestamp=datetime(year, 1, 1, tzinfo=UTC),
                    store_path=kwargs["store_path"],
                    completed_path=kwargs["completed_path_template"].format(year=year),
                    checkpoint_path=kwargs["checkpoint_path"],
                    time_index=years.index(year),
                    patch_size=kwargs["patch_size"],
                    window_size=kwargs["window_size"],
                    overlap_size=kwargs["overlap_size"],
                    compile_model=kwargs["compile_model"],
                    epsg_code=kwargs["epsg_code"],
                    wgs84_bounds=kwargs["wgs84_bounds"],
                    geojson_fname=kwargs["geojson_fname"],
                    job_size=kwargs["job_size"],
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

    # A worker holds one job at a time, so launching more workers than there are
    # outstanding jobs is pure churn: each surplus worker starts, finds nothing to claim,
    # and exits, and the next cycle's top-up launches it again. This is invisible during
    # the bulk of a run, where jobs outnumber workers, and dominates a tail resume: four
    # jobs left kept relaunching toward num_workers=32, each surplus worker living long
    # enough only to log "listening for messages" and stop.
    worker_target = min(num_workers, len(remaining))
    if live < worker_target:
        rslp.common.worker.launch_workers(
            image_name=kwargs["image_name"],
            queue_name=queue_name,
            num_workers=worker_target - live,
            cluster=kwargs["cluster"],
            gpus=kwargs["gpus"],
            shared_memory=kwargs["shared_memory"],
            priority=BeakerJobPriority[kwargs["priority"]],
            weka_mounts=[
                WekaMount(
                    bucket_name=kwargs["weka_bucket"],
                    mount_path=kwargs["weka_mount_path"],
                )
            ],
            extra_env_vars={
                "OEDATASETS_API_URL": kwargs["datasets_api_url"],
                **(kwargs.get("worker_env_vars") or {}),
            },
            extra_env_secrets={
                "DATASETS_API_TOKEN": kwargs["datasets_token_secret"],
                # .get so a caller passing a partial config still launches; these have
                # module defaults and are only overridden to point at other secrets.
                "AWS_ACCESS_KEY_ID": kwargs.get(
                    "aws_key_id_secret", DEFAULT_AWS_KEY_ID_SECRET
                ),
                "AWS_SECRET_ACCESS_KEY": kwargs.get(
                    "aws_secret_key_secret", DEFAULT_AWS_SECRET_KEY_SECRET
                ),
            },
        )
        logger.info(
            "launched %d worker(s) (target %d for %d outstanding job(s))",
            worker_target - live,
            worker_target,
            len(remaining),
        )


def launch_supervisor(
    image_name: str,
    cluster: list[str],
    supervise_args: list[str],
    priority: str = "high",
    task_name: str = "geozarr-supervisor",
    cpu_count: float = 2,
    memory: str = "8GiB",
    gpu_count: int = 0,
    preemptible: bool = False,
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
        priority: Beaker priority. Losing the supervisor stalls the whole run, so
            prefer a priority that will not be evicted.
        task_name: name for the Beaker experiment.
        cpu_count: CPUs to request.
        memory: memory to request.
        gpu_count: GPUs to request. The supervisor needs none, but on saturated
            GPU clusters a 0-GPU task may never be scheduled (slots are counted
            in GPUs), so requesting 1 is sometimes the only way to place it
            alongside the workers. Wasteful; prefer 0 where it schedules.
        preemptible: whether the supervisor itself may be preempted. Defaults to
            False; the supervisor is cheap and losing it stops all progress.

    Returns:
        the created Beaker experiment's ID.
    """
    from beaker import (
        Beaker,
        BeakerConstraints,
        BeakerExperimentSpec,
        BeakerJobPriority,
        BeakerTaskResources,
    )

    from rslp.utils.beaker import (
        DEFAULT_BUDGET,
        DEFAULT_WORKSPACE,
        create_gcp_credentials_mount,
        get_base_env_vars,
    )

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
    checkpoint_path: str,
    image_name: str,
    cluster: list[str],
    stage: str = STAGE_PREDICT,
    artifact_path: str | None = None,
    pca_store_path: str | None = None,
    pca_completed_path: str | None = None,
    max_level: int = DEFAULT_PCA_MAX_LEVEL,
    weka_bucket: str = DEFAULT_WEKA_BUCKET,
    weka_mount_path: str = DEFAULT_WEKA_MOUNT_PATH,
    datasets_api_url: str = DEFAULT_DATASETS_API_URL,
    datasets_token_secret: str = DEFAULT_DATASETS_TOKEN_SECRET,
    aws_key_id_secret: str = DEFAULT_AWS_KEY_ID_SECRET,
    aws_secret_key_secret: str = DEFAULT_AWS_SECRET_KEY_SECRET,
    worker_env_vars: dict[str, str] | None = None,
    num_workers: int = 8,
    gpus: int = 1,
    job_size: int = 8192,
    patch_size: int = 1,
    window_size: int = 16,
    overlap_size: int = 4,
    compile_model: bool = True,
    priority: str = "high",
    shared_memory: str = "256GiB",
    geojson_fname: str | None = None,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    cycle_seconds: int = 900,
    stale_seconds: int = 900,
    claim_stale_seconds: int = DEFAULT_CLAIM_STALE_SECONDS,
    cycle_budget_seconds: int = DEFAULT_CYCLE_BUDGET_SECONDS,
    max_cycles: int | None = None,
) -> None:
    """Refill the queue and worker pool each cycle until every tile has a marker.

    Args:
        inputs: which input variant to use.
        years: the reference years to produce; each becomes a (T, T) timestamp. The
            order must match the store's time axis.
        store_path: the GeoZarr store to write into (must already be initialized).
        completed_path_template: marker directory containing a ``{year}``
            placeholder, e.g. ``gs://bucket/prefix/s2_{year}_completed/``.
        queue_name: the Beaker queue to keep topped up.
        checkpoint_path: the OlmoEarth checkpoint to compute embeddings with.
        image_name: the Beaker image the workers run.
        cluster: Beaker clusters to schedule workers on.
        stage: which step to drive. "predict" writes embeddings and needs GPUs;
            "render_pca" writes the derived false-color layer from step 1's markers and
            needs none. Both are idempotent and marker-driven, so the same shallow-queue
            and worker-top-up loop gives both the same resilience.
        artifact_path: the fitted PCA artifact. Required for the render_pca stage.
        pca_store_path: the sibling store to write the pyramid into. Required for the
            render_pca stage; create it once with init_pca_store.
        pca_completed_path: marker directory for the render_pca stage's own output.
            Required for the render_pca stage.
        max_level: deepest pyramid level the render_pca stage writes.
        weka_bucket: WEKA bucket to mount (the checkpoint lives there).
        weka_mount_path: where to mount it.
        datasets_api_url: OlmoEarth Datasets API URL for the data source.
        datasets_token_secret: Beaker secret holding the datasets bearer token.
        aws_key_id_secret: Beaker secret holding an AWS access key id, for the S3
            fallback the data sources read with requester_pays=True.
        aws_secret_key_secret: Beaker secret holding the matching AWS secret key.
        worker_env_vars: extra plain env vars for the workers this launches, merged over
            the defaults. Needed for GDAL settings the data sources rely on, notably
            GS_USER_PROJECT: olmoearth_shared's rasterio_session_for_path honours
            requester_pays only for S3, so a GCS requester-pays bucket (Landsat) returns
            HTTP 400 unless GDAL is given a billing project this way.
        num_workers: how many workers to keep alive.
        gpus: GPUs to request per worker. Raising this reduces how many workers share
            a node, which is worth trying if workers are dying to memory pressure.
        job_size: pixel size of each job. Must be small enough that a job finishes
            inside the typical gap between preemptions (see the README).
        patch_size: the encoder patch size.
        window_size: the size of the crops the model operates on.
        overlap_size: overlap in pixels between adjacent crops.
        compile_model: whether to compile the encoder transformer blocks.
        priority: Beaker priority for the workers.
        shared_memory: shared memory to request per worker.
        geojson_fname: limit work to tiles intersecting this WGS84 GeoJSON file.
        epsg_code: limit work to the zone of this UTM EPSG code.
        wgs84_bounds: limit work to tiles intersecting these WGS84 bounds.
        cycle_seconds: how long to sleep between cycles.
        stale_seconds: a worker with no heartbeat for this long counts as dead.
        claim_stale_seconds: how long a queue entry's claim is trusted before the
            job is offered again. Must comfortably exceed one job's runtime or live
            work gets duplicated; claims are never released, so it cannot be
            infinite either, or a dead worker's job would never be retried.
        cycle_budget_seconds: kill a cycle that runs longer than this.
        max_cycles: stop after this many cycles; None runs until the work is done.
    """
    kwargs: dict[str, Any] = {
        "inputs": inputs,
        "years": years,
        "store_path": store_path,
        "completed_path_template": completed_path_template,
        "queue_name": queue_name,
        "checkpoint_path": checkpoint_path,
        "image_name": image_name,
        "cluster": cluster,
        "weka_bucket": weka_bucket,
        "weka_mount_path": weka_mount_path,
        "datasets_api_url": datasets_api_url,
        "datasets_token_secret": datasets_token_secret,
        "aws_key_id_secret": aws_key_id_secret,
        "aws_secret_key_secret": aws_secret_key_secret,
        "worker_env_vars": worker_env_vars,
        "num_workers": num_workers,
        "gpus": gpus,
        "job_size": job_size,
        "patch_size": patch_size,
        "window_size": window_size,
        "overlap_size": overlap_size,
        "compile_model": compile_model,
        "priority": priority,
        "shared_memory": shared_memory,
        "geojson_fname": geojson_fname,
        "epsg_code": epsg_code,
        "wgs84_bounds": wgs84_bounds,
        "stale_seconds": stale_seconds,
        "claim_stale_seconds": claim_stale_seconds,
        "stage": stage,
        "artifact_path": artifact_path,
        "pca_store_path": pca_store_path,
        "pca_completed_path": pca_completed_path,
        "max_level": max_level,
    }

    # "spawn" rather than the default fork: the child creates gRPC channels, and
    # forking a process that may already hold them is a known source of hangs.
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if stage == STAGE_RENDER_PCA:
        missing = [
            name
            for name, value in (
                ("artifact_path", artifact_path),
                ("pca_store_path", pca_store_path),
                ("pca_completed_path", pca_completed_path),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"stage {STAGE_RENDER_PCA} requires {', '.join(missing)}; fit the basis "
                "with the fit_pca workflow first"
            )

    ctx = multiprocessing.get_context("spawn")
    seen_work = False
    consecutive_failures = 0
    cycle = 0

    while max_cycles is None or cycle < max_cycles:
        cycle += 1
        # Typeshed types Value() as SynchronizedBase, which has no .value; the "i"
        # type code makes it a Synchronized[int].
        result: Synchronized[int] = ctx.Value("i", _NO_RESULT)  # type: ignore[assignment]
        proc = ctx.Process(target=_run_cycle, args=(kwargs, result))
        started = time.time()
        proc.start()
        proc.join(cycle_budget_seconds)
        if proc.is_alive():
            logger.warning(
                "cycle %d exceeded its %ds budget; killing it (likely a hung Beaker "
                "RPC) and continuing",
                cycle,
                cycle_budget_seconds,
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
                    "a missing geojson_fname, an unreadable store_path or an expired "
                    "credential all fail this way on every cycle."
                )
            logger.warning(
                "cycle %d did not report a result after %ds (exit code %s); retrying "
                "(%d/%d consecutive failures)",
                cycle,
                elapsed,
                proc.exitcode,
                consecutive_failures,
                MAX_CONSECUTIVE_CYCLE_FAILURES,
            )
        elif remaining == 0:
            consecutive_failures = 0
            if not seen_work and not _any_completion_markers(kwargs):
                # Enumerating nothing on the very first cycle almost never means "the
                # run is finished" -- far more often the AOI filters, bounds, or zone
                # selection exclude everything. Existing markers are the exception: a
                # resumed run whose stage is already complete legitimately sees zero
                # remaining on cycle one, and must skip the stage rather than fail.
                raise ValueError(
                    "enumerated no jobs at all on the first cycle; check "
                    "geojson_fname/wgs84_bounds/epsg_code and that store_path and "
                    "completed_path_template are correct"
                )
            logger.info("all tiles have completion markers; run complete")
            return
        else:
            consecutive_failures = 0
            seen_work = True
            logger.info(
                "cycle %d done in %ds; %d job(s) remaining", cycle, elapsed, remaining
            )

        time.sleep(cycle_seconds)

    logger.info("reached max_cycles=%d; exiting", cycle)
