"""Step 3 of the embedding flow: render the false-color pca_rgb layer.

The flow is three ordered steps, each depending on the previous one's output:

1. ``predict`` writes the int8 embeddings into the GeoZarr archive.
2. ``fit_pca`` samples that archive and fits the global basis, so the basis is fitted
   on exactly the data it will be applied to.
3. ``render_pca`` (this module) reads the embeddings back and writes ``pca_rgb``.

Step 3 is deliberately separate rather than folded into step 1, because the basis
cannot exist until step 1 has produced data to fit on. It is also the cheap step: no
model, no GPU, just a read, three dot products per pixel and a write. That means it
runs on ordinary CPU workers instead of competing for GPU capacity.

Work is enumerated from step 1's completion markers rather than by re-running the land
and wedge filters. Each marker lists exactly the windows that were written, so step 3
covers precisely step 1's output with no risk of the two disagreeing.
"""

import json
from datetime import datetime

import numpy as np
import zarr
from rslearn.utils.geometry import Projection
from upath import UPath

import rslp.common.worker
from rslp.log_utils import get_logger

from .pca import PcaArtifact, project_to_rgb
from .zarr_store import EMBEDDINGS_ARRAY, PCA_ARRAY, zone_group_name

logger = get_logger(__name__)


def get_pca_marker_fname(completed_path: str, source_marker_name: str) -> UPath:
    """Locate this step's completion marker for a given source marker.

    Args:
        completed_path: the directory holding step 3's markers.
        source_marker_name: the file name of the step 1 marker being rendered.

    Returns:
        the marker path for this unit of work.
    """
    return UPath(completed_path) / source_marker_name


def render_pca_pipeline(
    store_path: str,
    artifact_path: str,
    source_marker: str,
    completed_path: str,
    patch_size: int = 1,
    storage_options: dict | None = None,
) -> None:
    """Render one block's worth of windows into the store's pca_rgb array.

    Idempotent: returns immediately if this block's marker already exists, so jobs can
    be re-enqueued freely after a worker dies.

    Args:
        store_path: the GeoZarr store holding both arrays.
        artifact_path: the fitted global PCA artifact from step 2.
        source_marker: path to the step 1 completion marker naming the windows to
            render.
        completed_path: directory for this step's own completion markers.
        patch_size: the encoder patch size used when the embeddings were written.
        storage_options: fsspec storage options for remote stores.
    """
    source_fname = UPath(source_marker)
    marker_fname = get_pca_marker_fname(completed_path, source_fname.name)
    if marker_fname.exists():
        logger.info("marker file %s already exists", marker_fname)
        return

    with source_fname.open() as f:
        source = json.load(f)
    written: list[list[int]] = source.get("written") or []
    time_index = source["time_index"]

    projection = Projection.deserialize(source["projection"])
    epsg = projection.crs.to_epsg()
    if epsg is None:
        raise ValueError(f"projection in {source_fname.name} has no EPSG code")
    zone_number = epsg % 100

    artifact = PcaArtifact.load(artifact_path)
    group = zarr.open_group(
        store=store_path,
        path=zone_group_name(zone_number),
        mode="r+",
        storage_options=storage_options,
    )
    if PCA_ARRAY not in group:
        raise KeyError(
            f"{PCA_ARRAY} missing from {zone_group_name(zone_number)}; re-run init_store "
            "with create_pca_array=True"
        )
    transform = group.attrs["spatial:transform"]
    origin_x = round(transform[2] / transform[0])
    origin_y = round(transform[5] / transform[4])
    embeddings_array = group[EMBEDDINGS_ARRAY]
    pca_array = group[PCA_ARRAY]
    window_size = embeddings_array.shards[2]

    rendered: list[list[int]] = []
    empty: list[list[int]] = []
    for x, y in written:
        row = y // patch_size - origin_y
        col = x // patch_size - origin_x
        block = np.asarray(
            embeddings_array[
                time_index, :, row : row + window_size, col : col + window_size
            ]
        )
        rgb = project_to_rgb(block, artifact)
        if not rgb.any():
            # Nothing valid here; leave the shard unwritten so the array stays sparse.
            empty.append([x, y])
            continue
        pca_array[time_index, :, row : row + window_size, col : col + window_size] = rgb
        rendered.append([x, y])

    logger.info(
        "rendered %d window(s), skipped %d empty, for %s",
        len(rendered),
        len(empty),
        source_fname.name,
    )

    marker = {
        "source_marker": str(source_fname),
        "artifact_path": artifact_path,
        "time_index": time_index,
        "rendered": rendered,
        "skipped_empty": empty,
    }
    marker_fname.parent.mkdir(parents=True, exist_ok=True)
    with marker_fname.open("w") as f:
        json.dump(marker, f)
    logger.info("wrote marker file %s", marker_fname)


def get_render_jobs(
    store_path: str,
    artifact_path: str,
    source_completed_paths: list[str],
    completed_path: str,
    patch_size: int = 1,
) -> list[list[str]]:
    """Build one job per step 1 marker that has not yet been rendered.

    Args:
        store_path: the GeoZarr store holding both arrays.
        artifact_path: the fitted global PCA artifact.
        source_completed_paths: step 1 marker directories, one per reference year.
        completed_path: directory for this step's markers.
        patch_size: the encoder patch size used when the embeddings were written.

    Returns:
        a list of worker argument lists, one per unrendered block.
    """
    done = set()
    completed_upath = UPath(completed_path)
    if completed_upath.exists():
        done = {fname.name for fname in completed_upath.iterdir()}

    jobs: list[list[str]] = []
    total = 0
    for source_path in source_completed_paths:
        root = UPath(source_path)
        if not root.exists():
            logger.warning("source marker directory %s does not exist", source_path)
            continue
        for fname in root.iterdir():
            if not fname.name.endswith(".json"):
                continue
            total += 1
            if fname.name in done:
                continue
            jobs.append(
                [
                    "--store_path",
                    store_path,
                    "--artifact_path",
                    artifact_path,
                    "--source_marker",
                    str(fname),
                    "--completed_path",
                    completed_path,
                    "--patch_size",
                    str(patch_size),
                ]
            )
    logger.info("%d source marker(s), %d still to render", total, len(jobs))
    return jobs


def write_render_jobs(
    store_path: str,
    artifact_path: str,
    source_completed_paths: list[str],
    completed_path: str,
    queue_name: str,
    patch_size: int = 1,
) -> None:
    """Enqueue step 3 jobs on a Beaker queue.

    The artifact must already exist: rendering against a missing or refitted basis
    would produce pixels that do not match the rest of the archive.

    Args:
        store_path: the GeoZarr store holding both arrays.
        artifact_path: the fitted global PCA artifact from step 2.
        source_completed_paths: step 1 marker directories, one per reference year.
        completed_path: directory for this step's markers.
        queue_name: the Beaker queue to write job entries to.
        patch_size: the encoder patch size used when the embeddings were written.
    """
    # Fail before enqueuing anything rather than after every worker has started.
    artifact = PcaArtifact.load(artifact_path)
    logger.info(
        "rendering with a basis over %d dimensions fitted at %s",
        artifact.mean.shape[0],
        artifact.metadata.get("geoemb:pca_fitted_at", "unknown time"),
    )

    jobs = get_render_jobs(
        store_path=store_path,
        artifact_path=artifact_path,
        source_completed_paths=source_completed_paths,
        completed_path=completed_path,
        patch_size=patch_size,
    )
    if not jobs:
        logger.info("nothing to render; every source marker already has output")
        return
    rslp.common.worker.write_jobs(
        queue_name, "large_scale_embeddings", "render_pca", jobs
    )


def annotate_pca_array(
    store_path: str,
    artifact_path: str,
    zone_numbers: list[int] | None = None,
    storage_options: dict | None = None,
) -> None:
    """Record the basis provenance onto each zone's pca_rgb array attributes.

    The RGB pixels are meaningless without knowing which basis produced them, so the
    artifact metadata is copied onto the array itself. Run once after step 2.

    Args:
        store_path: the GeoZarr store to annotate.
        artifact_path: the fitted artifact whose metadata to record.
        zone_numbers: zones to annotate; defaults to every zone group present.
        storage_options: fsspec storage options for remote stores.
    """
    artifact = PcaArtifact.load(artifact_path)
    metadata = dict(artifact.metadata)
    metadata["geoemb:pca_artifact_path"] = artifact_path
    metadata["geoemb:pca_annotated_at"] = datetime.now().astimezone().isoformat()

    root = zarr.open_group(store=store_path, mode="r+", storage_options=storage_options)
    if zone_numbers is None:
        zones = [name for name in root.group_keys() if name.startswith("utm")]
    else:
        zones = [zone_group_name(z) for z in zone_numbers]

    annotated = 0
    for name in zones:
        group = root[name]
        if PCA_ARRAY not in group:
            continue
        group[PCA_ARRAY].attrs.update(metadata)
        annotated += 1
    logger.info("annotated %d pca_rgb array(s) in %s", annotated, store_path)
