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

from typing import Any

from upath import UPath

from rslp.log_utils import get_logger

from .pca import fit_pca
from .predict_pipeline import EmbeddingInputs
from .render_pca import annotate_pca_store, get_render_jobs
from .render_web_pca import get_web_jobs, init_web_store
from .supervise import (
    DEFAULT_WORKER_ENV_VARS as SUPERVISE_WORKER_ENV_VARS,
)
from .supervise import (
    STAGE_PREDICT,
    STAGE_RENDER_UTM_PCA,
    STAGE_RENDER_WEB_PCA,
    supervise,
)
from .write_jobs import get_jobs, init_store
from .zarr_store import DEFAULT_PCA_MAX_LEVEL, init_pca_store

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

# Supervisor cycle length for the web stage, overriding supervise's 900s default. That
# default is sized for predict, whose jobs run tens of minutes. A web shard takes a few
# seconds, so 32 workers drain a full 2,048-job queue in under three minutes and then
# idle out the rest of the cycle: measured at 900s the stage ran ~18% of the time and
# the interval, not the worker count, set throughput. Depth alone cannot fix it, since
# enqueueing is rate-limited to ~17 entries/s; the interval has to come down too.
WEB_CYCLE_SECONDS = 120

# How long a web worker waits for more work before exiting. It must exceed the cycle
# interval above, or the pool dies between refills: measured on the Kenya rebuild, the
# ten-second default had all 32 workers quit within seconds of draining the queue, and
# the supervisor then counted them as live for another fifteen minutes, so a three-minute
# burst of work was followed by a fifteen-minute gap. These workers hold no GPU, so
# waiting costs almost nothing next to paying container start again.
#
# This now equals supervise's own default, which was raised off None once the same defect
# was found to cost the predict stage 40% of its wall clock. Kept explicit anyway: the
# value is load-bearing for this stage in a way it is not elsewhere, since a web queue
# drains in under three minutes and a predict queue does not.
WEB_WORKER_IDLE_SECONDS = 900

# Re-exported from supervise, which now owns it: the merge has to happen where workers
# are launched, or it is skipped by every entry point that is not this one.
DEFAULT_WORKER_ENV_VARS = SUPERVISE_WORKER_ENV_VARS


def run_all(
    inputs: EmbeddingInputs,
    years: list[int],
    store_path: str,
    completed_path_template: str,
    queue_name: str,
    checkpoint_path: str,
    image_name: str,
    cluster: list[str],
    model_url: str,
    source_data: list[str],
    artifact_path: str,
    pca_store_path: str,
    pca_completed_path: str,
    zone_numbers: list[int] | None = None,
    matryoshka_dims: list[int] | None = None,
    max_level: int = DEFAULT_PCA_MAX_LEVEL,
    render_gpus: int = 0,
    skip_pca: bool = False,
    skip_web_pca: bool = False,
    web_store_path: str | None = None,
    web_completed_path: str | None = None,
    web_min_zoom: int = 8,
    web_max_zoom: int = 14,
    **supervise_kwargs: Any,
) -> None:
    """Run init_store, predict, fit_pca, render_pca and annotate to completion.

    Args:
        inputs: which input variant to use.
        years: reference years to produce; the order must match the store time axis.
        store_path: the embeddings GeoZarr store.
        completed_path_template: step 1 marker directory containing ``{year}``.
        queue_name: the Beaker queue to drive.
        checkpoint_path: the OlmoEarth checkpoint.
        image_name: the Beaker image for the workers.
        cluster: Beaker clusters to schedule workers on.
        model_url: URL reference to the encoder model, recorded in the store.
        source_data: URLs of the source datasets, recorded in the store.
        artifact_path: where step 2 writes the fitted basis.
        pca_store_path: the sibling store for the false-colour pyramid.
        pca_completed_path: step 3 marker directory.
        zone_numbers: UTM zones to create; defaults to all of 1-60.
        matryoshka_dims: prefix widths the model supports, recorded in the store.
        max_level: deepest pyramid level to write.
        render_gpus: GPUs for the render stage. It needs none; a nonzero value is only
            for saturated clusters that count slots in GPUs.
        skip_pca: stop after predict. For a run whose only product is embeddings.
        skip_web_pca: stop after annotate, leaving the display pyramid unbuilt.
        web_store_path: the web-mercator PCA store. Defaults beside the UTM one.
        web_completed_path: marker directory for step 5. Defaults beside the store.
        web_min_zoom: shallowest zoom to build.
        web_max_zoom: deepest zoom, warped directly from the UTM store.
        supervise_kwargs: forwarded verbatim to :func:`supervise`, so this never
            drifts from its options.

    Raises:
        RuntimeError: if a stage finishes with work outstanding. supervise returns
            normally when it hits max_cycles, so its return is not proof of
            completion; advancing on partial data is the failure this guards.
    """
    # Merge so an explicit value wins but the GDAL defaults are never simply forgotten.
    supervise_kwargs["worker_env_vars"] = {
        **DEFAULT_WORKER_ENV_VARS,
        **(supervise_kwargs.get("worker_env_vars") or {}),
    }
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
            zone_numbers=zone_numbers,
            matryoshka_dims=matryoshka_dims,
            patch_size=int(supervise_kwargs.get("patch_size", 1)),
        )

    logger.info("step 1/5: predict")
    supervise(
        inputs=inputs,
        years=years,
        store_path=store_path,
        completed_path_template=completed_path_template,
        queue_name=queue_name,
        checkpoint_path=checkpoint_path,
        image_name=image_name,
        cluster=cluster,
        stage=STAGE_PREDICT,
        **supervise_kwargs,
    )
    _require_no_predict_jobs(
        inputs=inputs,
        years=years,
        store_path=store_path,
        completed_paths=completed_paths,
        checkpoint_path=checkpoint_path,
        supervise_kwargs=supervise_kwargs,
    )

    if skip_pca:
        logger.info("skip_pca set; stopping after predict")
        return

    logger.info("step 2/5: fit_pca -> %s", artifact_path)
    fit_pca(
        store_path=store_path,
        completed_paths=completed_paths,
        artifact_path=artifact_path,
    )

    logger.info("step 3/5: render_pca into %s", pca_store_path)
    if not UPath(pca_store_path).exists():
        init_pca_store(
            pca_store_path=pca_store_path,
            zone_numbers=zone_numbers or list(range(1, 61)),
            years=years,
            model_url=model_url,
            source_data=source_data,
            resolution=10,
            tile_size=32768,
            max_level=max_level,
        )
    render_kwargs = dict(supervise_kwargs)
    render_kwargs["gpus"] = render_gpus
    supervise(
        inputs=inputs,
        years=years,
        store_path=store_path,
        completed_path_template=completed_path_template,
        queue_name=queue_name,
        checkpoint_path=checkpoint_path,
        image_name=image_name,
        cluster=cluster,
        stage=STAGE_RENDER_UTM_PCA,
        artifact_path=artifact_path,
        pca_store_path=pca_store_path,
        pca_completed_path=pca_completed_path,
        max_level=max_level,
        **render_kwargs,
    )
    remaining = get_render_jobs(
        store_path=store_path,
        pca_store_path=pca_store_path,
        artifact_path=artifact_path,
        source_completed_paths=completed_paths,
        completed_path=pca_completed_path,
        patch_size=int(supervise_kwargs.get("patch_size", 1)),
        max_level=max_level,
    )
    if remaining:
        raise RuntimeError(
            f"render_pca finished with {len(remaining)} block(s) unrendered; "
            "not annotating a partly-rendered store"
        )

    logger.info("step 4/5: annotate_pca_store")
    annotate_pca_store(
        pca_store_path=pca_store_path,
        artifact_path=artifact_path,
        zone_numbers=zone_numbers,
        max_level=max_level,
    )
    if skip_web_pca:
        logger.info("skip_web_pca set; stopping after annotate")
        return

    # Siblings of the UTM pyramid by default, so a run needs no extra paths to gain a
    # display layer. The version lives in the name, as it does for pca_v1.zarr, so a
    # rebuild can be staged beside the old one.
    web_store_path = web_store_path or pca_store_path.replace(
        "pca_v1.zarr", "pca_web_v1.zarr"
    )
    web_completed_path = web_completed_path or (
        pca_completed_path.rstrip("/") + "_web/"
    )

    # Step 5: the display pyramid. One supervise stage per zoom, deepest first, because
    # a coarse shard is built from the four below it and those must already exist.
    # Zoom order is the dependency, so this is a sequence rather than one flat stage.
    logger.info("step 5/5: render_web_pca (zooms %d..%d)", web_min_zoom, web_max_zoom)
    if not UPath(web_store_path).exists():
        init_web_store(
            store_path=web_store_path,
            years=years,
            min_zoom=web_min_zoom,
            max_zoom=web_max_zoom,
            source_store_path=pca_store_path,
        )
    else:
        logger.info("web store already exists; leaving it as is")

    # jsonargparse expands supervise's own signature into this function's CLI, because
    # of **supervise_kwargs. So a supervise-only option like --web_zoom arrives here
    # carrying its default, and passing it on while also naming it below raises
    # "got multiple values for keyword argument". Drop anything set explicitly.
    web_explicit = {
        "inputs",
        "years",
        "store_path",
        "completed_path_template",
        "queue_name",
        "checkpoint_path",
        "image_name",
        "cluster",
        "stage",
        "pca_store_path",
        "artifact_path",
        "web_store_path",
        "web_completed_path",
        "web_zoom",
        "web_base_zoom",
        "zone_numbers",
        # Stripped and then set below rather than left to setdefault. Because run_all
        # takes **supervise_kwargs, jsonargparse expands supervise's whole signature
        # into run_all's CLI, so these three arrive here already present, carrying
        # supervise's own defaults. setdefault therefore never fired and the web values
        # below were dead code: the runs that looked correct only did so because the
        # same numbers were also passed on the command line by hand.
        "cycle_seconds",
        "pending_per_worker",
        "worker_idle_seconds",
    }
    web_kwargs = {k: v for k, v in supervise_kwargs.items() if k not in web_explicit}
    web_kwargs["gpus"] = render_gpus
    # A shard takes seconds rather than tens of minutes, so supervise's long-job
    # defaults are all wrong here: the queue has to be deeper, the cycle shorter, and
    # the worker has to outlive the gap between refills. Assigned, not defaulted -- see
    # the note in web_explicit above for why a default cannot work through this CLI.
    web_kwargs["pending_per_worker"] = WEB_PENDING_PER_WORKER
    web_kwargs["cycle_seconds"] = WEB_CYCLE_SECONDS
    web_kwargs["worker_idle_seconds"] = WEB_WORKER_IDLE_SECONDS

    for zoom in range(web_max_zoom, web_min_zoom - 1, -1):
        supervise(
            inputs=inputs,
            years=years,
            store_path=store_path,
            completed_path_template=completed_path_template,
            queue_name=queue_name,
            checkpoint_path=checkpoint_path,
            image_name=image_name,
            cluster=cluster,
            stage=STAGE_RENDER_WEB_PCA,
            pca_store_path=pca_store_path,
            artifact_path=artifact_path,
            web_store_path=web_store_path,
            web_completed_path=web_completed_path,
            web_zoom=zoom,
            web_base_zoom=web_max_zoom,
            zone_numbers=zone_numbers,
            **web_kwargs,
        )
        outstanding = get_web_jobs(
            source_store_path=pca_store_path,
            web_store_path=web_store_path,
            completed_path=web_completed_path,
            zoom=zoom,
            years=years,
            zone_numbers=zone_numbers or [],
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
    checkpoint_path: str,
    supervise_kwargs: dict,
) -> None:
    """Raise unless every predict block has a completion marker.

    Checked per year against the markers rather than trusting supervise's return: it
    exits normally when it runs out of cycles, and a basis fitted on a partly-written
    archive is wrong in a way nothing downstream detects.
    """
    from datetime import datetime

    outstanding = 0
    for time_index, (year, completed_path) in enumerate(zip(years, completed_paths)):
        jobs = get_jobs(
            inputs=inputs,
            timestamp=datetime.fromisoformat(f"{year}-01-01T00:00:00+00:00"),
            store_path=store_path,
            completed_path=completed_path,
            checkpoint_path=checkpoint_path,
            time_index=time_index,
            patch_size=int(supervise_kwargs.get("patch_size", 1)),
            window_size=int(supervise_kwargs.get("window_size", 16)),
            overlap_size=int(supervise_kwargs.get("overlap_size", 4)),
            compile_model=bool(supervise_kwargs.get("compile_model", True)),
            geojson_fname=supervise_kwargs.get("geojson_fname"),
            epsg_code=supervise_kwargs.get("epsg_code"),
            wgs84_bounds=supervise_kwargs.get("wgs84_bounds"),
            job_size=int(supervise_kwargs.get("job_size", 8192)),
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
    preemptible: bool = False,
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
            defaults to urgent. "high" is not enough: the rc-9 supervisor ran at
            high with preemptible=False and was still evicted 12 hours in.
        task_name: name for the Beaker experiment.
        cpu_count: CPUs to request.
        memory: memory to request.
        gpu_count: GPUs to request; 0 unless the cluster only schedules by GPU slot.
        preemptible: whether the driver may be preempted. Defaults to False; it is
            cheap and losing it stops all progress.

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
        description="large_scale_embeddings full run",
        beaker_image=image_name,
        priority=BeakerJobPriority[priority],
        command=["python", "-m", "rslp.main"],
        arguments=["large_scale_embeddings", "run_all", *run_all_args],
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
    logger.info("launched run_all experiment %s on %s", experiment_id, cluster)
    return experiment_id
