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
from .supervise import STAGE_PREDICT, STAGE_RENDER_PCA, supervise
from .write_jobs import get_jobs, init_store
from .zarr_store import DEFAULT_PCA_MAX_LEVEL, init_pca_store

logger = get_logger(__name__)

# GDAL settings the Landsat and Sentinel-2 sources need, applied by default because
# omitting either fails the run minutes in, after the image pull and dataset prepare,
# with an error that names neither variable:
#
# GS_USER_PROJECT -- olmoearth_shared's rasterio_session_for_path honours
#   requester_pays only for S3, returning a bare GSSession() for GCS. The USGS Landsat
#   mirror is requester-pays, so without a billing project GDAL gets HTTP 400.
# AWS_NO_SIGN_REQUEST -- some assets resolve to public S3, where GDAL otherwise looks
#   for credentials that do not exist in a Beaker worker and raises InvalidCredentials.
#
# Both are overridable: a caller passing either key in worker_env_vars wins.
DEFAULT_WORKER_ENV_VARS = {
    "GS_USER_PROJECT": "earthsystem-dev-c3po",
    "AWS_NO_SIGN_REQUEST": "YES",
}


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

    logger.info("step 0/4: ensuring the store exists at %s", store_path)
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

    logger.info("step 1/4: predict")
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

    logger.info("step 2/4: fit_pca -> %s", artifact_path)
    fit_pca(
        store_path=store_path,
        completed_paths=completed_paths,
        artifact_path=artifact_path,
    )

    logger.info("step 3/4: render_pca into %s", pca_store_path)
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
        stage=STAGE_RENDER_PCA,
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

    logger.info("step 4/4: annotate_pca_store")
    annotate_pca_store(
        pca_store_path=pca_store_path,
        artifact_path=artifact_path,
        zone_numbers=zone_numbers,
        max_level=max_level,
    )
    logger.info("run complete: all four steps finished with no work outstanding")


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
    priority: str = "high",
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
        priority: Beaker priority. Losing the driver stalls the whole run, so prefer a
            priority that will not be evicted.
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
