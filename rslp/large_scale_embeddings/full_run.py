"""Drive the whole embedding flow as one Beaker job.

The flow is three ordered steps with real dependencies: ``fit_pca`` cannot run until
``predict`` has produced data to fit on, and ``render_pca`` cannot run without the
fitted basis. Running them by hand means four or five invocations with a human holding
the ordering in their head, and the intermediate states are where mistakes hide -- a
basis fitted on a partly-written archive produces colours that look fine and are wrong.

So this owns the ordering, and refuses to advance on partial input. Every step is
already idempotent and marker-driven, which is what makes chaining them safe: a failed
run can be re-invoked and picks up where it stopped.

Failures are not absorbed. A stage that cannot finish raises, which fails the Beaker
job, rather than letting the next stage run on incomplete data.
"""

from dataclasses import replace
from datetime import datetime, timedelta

from beaker import (
    Beaker,
    BeakerConstraints,
    BeakerExperimentSpec,
    BeakerJobPriority,
    BeakerTaskResources,
)
from upath import UPath

from rslp.large_scale_embeddings.pca import fit_pca
from rslp.large_scale_embeddings.predict_pipeline import EmbeddingInputs
from rslp.large_scale_embeddings.render_pca import annotate_pca_store, get_render_jobs
from rslp.large_scale_embeddings.render_web_pca import get_web_jobs, init_web_store
from rslp.large_scale_embeddings.supervise import (
    DEFAULT_SUPERVISOR_MIN_RUNTIME,
    STAGE_PREDICT,
    STAGE_RENDER_UTM_PCA,
    STAGE_RENDER_WEB_PCA,
    AoiConfig,
    CycleConfig,
    ModelConfig,
    PcaConfig,
    WorkerConfig,
    supervise,
)
from rslp.large_scale_embeddings.supervise import (
    DEFAULT_WORKER_ENV_VARS as SUPERVISE_WORKER_ENV_VARS,
)
from rslp.large_scale_embeddings.write_jobs import get_jobs, init_store
from rslp.large_scale_embeddings.zarr_store import (
    DEFAULT_MODEL_URL,
    init_pca_store,
    source_data_for,
)
from rslp.log_utils import get_logger
from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_WORKSPACE,
    create_gcp_credentials_mount,
    get_base_env_vars,
)

logger = get_logger(__name__)

# GS_USER_PROJECT is required for the Landsat reads and easy to forget: omitting it
# fails minutes in, after the image pull and dataset prepare, with an HTTP 400 that names
# no variable. olmoearth_shared's rasterio_session_for_path honours requester_pays only
# for S3, returning a bare GSSession() for GCS, so the requester-pays USGS mirror needs
# GDAL to be handed a billing project this way. The deployed olmoearth_run runner sets
# exactly this, from the same project id.
#
# AWS credentials are NOT here: they are secrets, mounted by supervise from Beaker
# secrets. Note that AWS_NO_SIGN_REQUEST would be actively wrong -- the data sources
# request assets with requester_pays=True, which cannot be served by an unsigned request.
# Queue depth per worker for the web stage. Its jobs take seconds, not hours, so the
# default of three would leave workers idle between supervisor cycles and make the
# cycle interval the throughput ceiling regardless of worker count.
WEB_PENDING_PER_WORKER = 64

# Supervisor cycle length for the web stage. The default is sized for predict, whose
# jobs run tens of minutes; a web shard takes seconds, so a full pool drains the queue
# in minutes and then idles, making the interval rather than the worker count set
# throughput. Queue depth alone cannot fix it, since enqueueing is rate-limited.
WEB_CYCLE_SECONDS = int(timedelta(minutes=2).total_seconds())

# How long a web worker waits for more work before exiting. Must exceed the cycle
# interval above, or the pool dies between refills. These workers hold no GPU, so
# waiting costs almost nothing next to paying container start again. Kept explicit
# because the value is load-bearing for this stage in a way it is not elsewhere.
WEB_WORKER_IDLE_SECONDS = int(timedelta(minutes=15).total_seconds())

# Re-exported from supervise, which now owns it: the merge has to happen where workers
# are launched, or it is skipped by every entry point that is not this one.
DEFAULT_WORKER_ENV_VARS = SUPERVISE_WORKER_ENV_VARS


def run_all(
    inputs: EmbeddingInputs,
    years: list[int],
    store_path: str,
    completed_path_template: str,
    queue_name: str,
    model: ModelConfig,
    worker: WorkerConfig,
    pca: PcaConfig,

    model_url: str = DEFAULT_MODEL_URL,
    source_data: list[str] | None = None,
    cycle: CycleConfig | None = None,
    aoi: AoiConfig | None = None,
    matryoshka_dims: list[int] | None = None,
    render_gpus: int = 0,
    skip_pca: bool = False,
    skip_web_pca: bool = False,
    web_min_zoom: int = 8,
    web_max_zoom: int = 14,
) -> None:
    """Run init_store, predict, fit_pca, render_pca and annotate to completion.

    Args:
        inputs: which input variant to use.
        years: reference years to produce; the order must match the store time axis.
        store_path: the embeddings GeoZarr store.
        completed_path_template: ``predict`` marker directory containing ``{year}``.
        queue_name: the Beaker queue to drive.
        model: the encoder and how it is run. See `ModelConfig`.
        worker: the Beaker worker pool. See `WorkerConfig`.
        pca: the derived-layer paths. `artifact_path`, `store_path` and
            `completed_path` are required unless `skip_pca` is set.
        model_url: URL reference to the encoder model, recorded in the store.
            Defaults to the released encoder these embeddings come from.
        source_data: URLs of the source datasets. Derived from `inputs` if unset.
        cycle: loop pacing for the predict and render stages. See `CycleConfig`.
        aoi: the ground to cover. See `AoiConfig`.
        matryoshka_dims: prefix widths the model supports, recorded in the store.
        render_gpus: GPUs for the render stages. They need none; a nonzero value is
            only for saturated clusters that count slots in GPUs.
        skip_pca: stop after predict. For a run whose only product is embeddings.
        skip_web_pca: stop after annotate, leaving the display pyramid unbuilt.
        web_min_zoom: shallowest zoom to build.
        web_max_zoom: deepest zoom, warped directly from the UTM store.

    Raises:
        RuntimeError: if a stage finishes with work outstanding. supervise returns
            normally when it hits max_cycles, so its return is not proof of
            completion; advancing on partial data is the failure this guards.
    """
    cycle = cycle or CycleConfig()
    aoi = aoi or AoiConfig()
    source_data = source_data or source_data_for(inputs.value)
    completed_paths = [completed_path_template.format(year=year) for year in years]

    logger.info("step 0/5: ensuring the store exists at %s", store_path)
    if UPath(store_path).exists():
        logger.info("store already exists; leaving it as is")
    else:
        init_store(
            store_path=store_path,
            years=years,
            model_url=model_url,
            source_data=source_data,
            zone_numbers=aoi.zone_numbers,
            matryoshka_dims=matryoshka_dims,
            patch_size=model.patch_size,
        )

    logger.info("step 1/5: predict")
    supervise(
        inputs=inputs,
        years=years,
        store_path=store_path,
        completed_path_template=completed_path_template,
        queue_name=queue_name,
        model=model,
        worker=worker,
        stage=STAGE_PREDICT,
        cycle=cycle,
        aoi=aoi,
    )
    _require_no_predict_jobs(
        inputs=inputs,
        years=years,
        store_path=store_path,
        completed_paths=completed_paths,
        model=model,
        aoi=aoi,
    )

    if skip_pca:
        logger.info("skip_pca set; stopping after predict")
        return

    logger.info("step 2/5: fit_pca -> %s", pca.artifact_path)
    fit_pca(
        store_path=store_path,
        completed_paths=completed_paths,
        artifact_path=pca.artifact_path,
    )

    logger.info("step 3/5: render_pca into %s", pca.store_path)
    if not UPath(pca.store_path).exists():
        init_pca_store(
            pca_store_path=pca.store_path,
            zone_numbers=aoi.zone_numbers or list(range(1, 61)),
            years=years,
            model_url=model_url,
            source_data=source_data,
            resolution=10,
            tile_size=32768,
            max_level=pca.max_level,
        )
    supervise(
        inputs=inputs,
        years=years,
        store_path=store_path,
        completed_path_template=completed_path_template,
        queue_name=queue_name,
        model=model,
        worker=replace(worker, gpus=render_gpus),
        stage=STAGE_RENDER_UTM_PCA,
        cycle=cycle,
        aoi=aoi,
        pca=pca,
    )
    remaining = get_render_jobs(
        store_path=store_path,
        pca_store_path=pca.store_path,
        artifact_path=pca.artifact_path,
        source_completed_paths=completed_paths,
        completed_path=pca.completed_path,
        patch_size=model.patch_size,
        max_level=pca.max_level,
    )
    if remaining:
        raise RuntimeError(
            f"render_pca finished with {len(remaining)} block(s) unrendered; "
            "not annotating a partly-rendered store"
        )

    logger.info("step 4/5: annotate_pca_store")
    annotate_pca_store(
        pca_store_path=pca.store_path,
        artifact_path=pca.artifact_path,
        zone_numbers=aoi.zone_numbers,
        max_level=pca.max_level,
    )
    if skip_web_pca:
        logger.info("skip_web_pca set; stopping after annotate")
        return

    # Siblings of the UTM pyramid by default, so a run needs no extra paths to gain a
    # display layer. The version lives in the name, as it does for pca_v1.zarr, so a
    # rebuild can be staged beside the old one.
    # Siblings of the UTM pyramid by default, so a run needs no extra paths to gain a
    # display layer. The version lives in the name, as it does for pca_v1.zarr, so a
    # rebuild can be staged beside the old one.
    web_store_path = pca.web_store_path or pca.store_path.replace(
        "pca_v1.zarr", "pca_web_v1.zarr"
    )
    web_completed_path = pca.web_completed_path or (
        pca.completed_path.rstrip("/") + "_web/"
    )

    # The display pyramid. One supervise stage per zoom, deepest first, because
    # a coarse shard is built from the four below it and those must already exist.
    # Zoom order is the dependency, so this is a sequence rather than one flat stage.
    logger.info("step 5/5: render_web_pca (zooms %d..%d)", web_min_zoom, web_max_zoom)
    if not UPath(web_store_path).exists():
        init_web_store(
            store_path=web_store_path,
            years=years,
            min_zoom=web_min_zoom,
            max_zoom=web_max_zoom,
            source_store_path=pca.store_path,
        )
    else:
        logger.info("web store already exists; leaving it as is")

    # A shard takes seconds rather than tens of minutes, so the long-job defaults are
    # all wrong here: the queue has to be deeper, the cycle shorter, and the worker has
    # to outlive the gap between refills.
    web_cycle = replace(
        cycle,
        seconds=WEB_CYCLE_SECONDS,
        pending_per_worker=WEB_PENDING_PER_WORKER,
    )
    web_worker = replace(
        worker, gpus=render_gpus, idle_seconds=WEB_WORKER_IDLE_SECONDS
    )

    for zoom in range(web_max_zoom, web_min_zoom - 1, -1):
        supervise(
            inputs=inputs,
            years=years,
            store_path=store_path,
            completed_path_template=completed_path_template,
            queue_name=queue_name,
            model=model,
            worker=web_worker,
            stage=STAGE_RENDER_WEB_PCA,
            cycle=web_cycle,
            aoi=aoi,
            pca=replace(
                pca,
                web_store_path=web_store_path,
                web_completed_path=web_completed_path,
                web_zoom=zoom,
                web_base_zoom=web_max_zoom,
            ),
        )
        outstanding = get_web_jobs(
            source_store_path=pca.store_path,
            web_store_path=web_store_path,
            completed_path=web_completed_path,
            zoom=zoom,
            years=years,
            zone_numbers=aoi.zone_numbers or [],
            base_zoom=web_max_zoom,
        )
        if outstanding:
            raise RuntimeError(
                f"render_web_pca z{zoom} finished with {len(outstanding)} shard(s) "
                "outstanding; a coarser level built on an incomplete one would be wrong"
            )

    logger.info("run complete: all five steps finished with no work outstanding")


def _require_no_predict_jobs(
    inputs: EmbeddingInputs,
    years: list[int],
    store_path: str,
    completed_paths: list[str],
    model: ModelConfig,
    aoi: AoiConfig,
) -> None:
    """Raise unless every predict block has a completion marker.

    Checked per year against the markers rather than trusting supervise's return: it
    exits normally when it runs out of cycles, and a basis fitted on a partly-written
    archive is wrong in a way nothing downstream detects.
    """
    outstanding = 0
    for time_index, (year, completed_path) in enumerate(zip(years, completed_paths)):
        jobs = get_jobs(
            inputs=inputs,
            timestamp=datetime.fromisoformat(f"{year}-01-01T00:00:00+00:00"),
            store_path=store_path,
            completed_path=completed_path,
            checkpoint_path=model.checkpoint_path,
            time_index=time_index,
            patch_size=model.patch_size,
            window_size=model.window_size,
            overlap_size=model.overlap_size,
            compile_model=model.compile_model,
            geojson_fname=aoi.geojson_fname,
            epsg_code=aoi.epsg_code,
            wgs84_bounds=aoi.wgs84_bounds,
            job_size=aoi.job_size,
        )
        if jobs:
            logger.error("year %d still has %d block(s) to predict", year, len(jobs))
            outstanding += len(jobs)
    if outstanding:
        raise RuntimeError(
            f"predict finished with {outstanding} block(s) incomplete; refusing to fit "
            "a PCA basis on a partly-written archive"
        )


def launch_run_all(
    image_name: str,
    cluster: list[str],
    run_all_args: list[str],
    priority: str = "urgent",
    task_name: str = "geozarr-run-all",
    cpu_count: float = 2,
    memory: str = "8GiB",
    gpu_count: int = 0,
    min_runtime: timedelta = DEFAULT_SUPERVISOR_MIN_RUNTIME,
    auto_resume: bool = True,
) -> str:
    """Launch :func:`run_all` as a CPU-only Beaker job.

    A full run spans hours to days across three stages, so the driver cannot live on a
    workstation. It needs no GPU of its own: it launches the workers that do.

    Args:
        image_name: the Beaker image to run (must contain this workflow).
        cluster: clusters to schedule on; a CPU cluster is appropriate.
        run_all_args: arguments forwarded verbatim to the ``run_all`` workflow. Passed
            through rather than re-declared so this launcher never drifts from
            run_all()'s options.
        priority: Beaker priority. Losing the driver stalls the whole run, so this
            defaults to urgent.
        task_name: name for the Beaker experiment.
        cpu_count: CPUs to request.
        memory: memory to request.
        gpu_count: GPUs to request; 0 unless the cluster only schedules by GPU slot.
        min_runtime: how long the scheduler should let the driver run before it may be
            preempted. Above five minutes the job counts as allocated.
        auto_resume: whether Beaker replaces the job when it is preempted. Leave this on:
            without it a preempted driver stops the whole run.

    Returns:
        the created Beaker experiment's ID.
    """
    spec = BeakerExperimentSpec.new(
        budget=DEFAULT_BUDGET,
        description="large_scale_embeddings full run",
        beaker_image=image_name,
        priority=BeakerJobPriority[priority],
        command=["python", "-m", "rslp.main"],
        arguments=["large_scale_embeddings", "run_all", *run_all_args],
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
    logger.info("launched run_all experiment %s on %s", experiment_id, cluster)
    return experiment_id
