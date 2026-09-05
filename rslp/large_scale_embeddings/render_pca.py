"""Render the false-color pyramid from the written embeddings.

``predict`` writes the int8 embeddings, ``fit_pca`` fits a global basis on exactly the
data it will be applied to, and this module reads the embeddings back and writes the
multiscale ``pca_rgb`` pyramid into a separate store.

It is separate from ``predict`` because the basis cannot exist until there is data
to fit on, and because it is cheap: no model, no GPU, just a read, three dot products
per pixel, and a write, so it runs on ordinary CPU workers.

The output is a sibling store rather than another array in the embeddings store, so the
basis can be refit without touching the source and two renders can coexist during a
cutover. Put the basis version in its path.

The pyramid is what makes the store directly servable: a client picks a level by zoom
and reads a bounded number of chunks at any extent. Every level keeps one shard per
window footprint, so a window is still a whole object owned by a single writer and
concurrent renders of disjoint windows need no locking.

Work is enumerated from ``predict``'s completion markers, each of which lists exactly
the windows that were written, so this stage covers precisely that output.
"""

import json
from datetime import datetime

import numpy as np
import zarr
from rslearn.utils.geometry import Projection
from upath import UPath

import rslp.common.worker
from rslp.large_scale_embeddings.pca import PcaArtifact, build_pyramid, project_to_rgb
from rslp.large_scale_embeddings.zarr_store import (
    DEFAULT_PCA_MAX_LEVEL,
    EMBEDDINGS_ARRAY,
    pca_level_array_name,
    write_pca_window_levels,
    zone_group_name,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)


def pca_marker_name(source_fname: UPath) -> str:
    """Build this stage's marker name for a predict marker, unique across years.

    The source's parent directory must be part of the name. ``predict`` puts the year in the
    *directory* (completed_2022/, completed_2023/, ...) and gives every year the same
    file name for a given block, so keying only on the file name collapses all years
    onto one marker: the first year written makes every later year look already done and
    the block is silently skipped.

    Derived from the path alone rather than from the marker's ``time_index`` field, so
    enumeration can compute it without opening every source marker.

    Args:
        source_fname: the predict marker being rendered.

    Returns:
        the marker file name for this unit of work.
    """
    return f"{source_fname.parent.name}_{source_fname.name}"


def get_pca_marker_fname(completed_path: str, source_fname: UPath) -> UPath:
    """Locate this stage's completion marker for a given predict marker.

    Args:
        completed_path: the directory holding this stage's markers.
        source_fname: the predict marker being rendered.

    Returns:
        the marker path for this unit of work.
    """
    return UPath(completed_path) / pca_marker_name(source_fname)


def render_pca_pipeline(
    store_path: str,
    pca_store_path: str,
    artifact_path: str,
    source_marker: str,
    completed_path: str,
    patch_size: int = 1,
    max_level: int = DEFAULT_PCA_MAX_LEVEL,
    storage_options: dict | None = None,
) -> None:
    """Render one block's worth of windows into the pca store's pyramid.

    Idempotent: returns immediately if this block's marker already exists, so jobs can
    be re-enqueued freely after a worker dies.

    Args:
        store_path: the GeoZarr store holding the embeddings. Opened read-only.
        pca_store_path: the sibling store to write the false-color pyramid into.
        artifact_path: the fitted global PCA artifact from ``fit_pca``.
        source_marker: path to the predict completion marker naming the windows to
            render.
        completed_path: directory for this step's own completion markers.
        patch_size: the encoder patch size used when the embeddings were written.
        max_level: deepest pyramid level to write, downsampled 2**max_level.
        storage_options: fsspec storage options for remote stores.
    """
    source_fname = UPath(source_marker)
    with source_fname.open() as f:
        source = json.load(f)
    written: list[list[int]] = source.get("written") or []
    time_index = source["time_index"]

    marker_fname = get_pca_marker_fname(completed_path, source_fname)
    if marker_fname.exists():
        logger.info("marker file %s already exists", marker_fname)
        return

    projection = Projection.deserialize(source["projection"])
    epsg = projection.crs.to_epsg()
    if epsg is None:
        raise ValueError(f"projection in {source_fname.name} has no EPSG code")
    zone_number = epsg % 100

    artifact = PcaArtifact.load(artifact_path)
    group = zarr.open_group(
        store=store_path,
        path=zone_group_name(zone_number),
        mode="r",
        storage_options=storage_options,
    )
    transform = group.attrs["spatial:transform"]
    origin_x = round(transform[2] / transform[0])
    origin_y = round(transform[5] / transform[4])
    embeddings_array = group[EMBEDDINGS_ARRAY]
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
            # Nothing valid here; leave the shards unwritten so the arrays stay sparse.
            empty.append([x, y])
            continue
        write_pca_window_levels(
            pca_store_path=pca_store_path,
            zone_number=zone_number,
            window_bounds=(x, y, x + window_size, y + window_size),
            time_index=time_index,
            levels=build_pyramid(rgb, max_level),
            patch_size=patch_size,
            storage_options=storage_options,
        )
        rendered.append([x, y])

    logger.info(
        "rendered %d window(s) at levels 0..%d, skipped %d empty, for %s",
        len(rendered),
        max_level,
        len(empty),
        source_fname.name,
    )

    marker = {
        "source_marker": str(source_fname),
        "pca_store_path": pca_store_path,
        "artifact_path": artifact_path,
        "time_index": time_index,
        "max_level": max_level,
        "levels": [pca_level_array_name(k) for k in range(max_level + 1)],
        "rendered": rendered,
        "skipped_empty": empty,
    }
    marker_fname.parent.mkdir(parents=True, exist_ok=True)
    with marker_fname.open("w") as f:
        json.dump(marker, f)
    logger.info("wrote marker file %s", marker_fname)


def get_render_jobs(
    store_path: str,
    pca_store_path: str,
    artifact_path: str,
    source_completed_paths: list[str],
    completed_path: str,
    patch_size: int = 1,
    max_level: int = DEFAULT_PCA_MAX_LEVEL,
) -> list[list[str]]:
    """Build one job per predict marker that has not yet been rendered.

    Args:
        store_path: the GeoZarr store holding the embeddings.
        pca_store_path: the sibling store to write the pyramid into.
        artifact_path: the fitted global PCA artifact.
        source_completed_paths: predict marker directories, one per reference year.
        completed_path: directory for this step's markers.
        patch_size: the encoder patch size used when the embeddings were written.
        max_level: deepest pyramid level to write.

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
            if pca_marker_name(fname) in done:
                continue
            jobs.append(
                [
                    "--store_path",
                    store_path,
                    "--pca_store_path",
                    pca_store_path,
                    "--artifact_path",
                    artifact_path,
                    "--source_marker",
                    str(fname),
                    "--completed_path",
                    completed_path,
                    "--patch_size",
                    str(patch_size),
                    "--max_level",
                    str(max_level),
                ]
            )
    logger.info("%d source marker(s), %d still to render", total, len(jobs))
    return jobs


def write_render_jobs(
    store_path: str,
    pca_store_path: str,
    artifact_path: str,
    source_completed_paths: list[str],
    completed_path: str,
    queue_name: str,
    patch_size: int = 1,
    max_level: int = DEFAULT_PCA_MAX_LEVEL,
) -> None:
    """Enqueue render_pca jobs on a Beaker queue.

    The artifact must already exist: rendering against a missing or refitted basis would
    produce pixels that do not match the rest of the store.

    Args:
        store_path: the GeoZarr store holding the embeddings.
        pca_store_path: the sibling store to write the pyramid into.
        artifact_path: the fitted global PCA artifact from ``fit_pca``.
        source_completed_paths: predict marker directories, one per reference year.
        completed_path: directory for this step's markers.
        queue_name: the Beaker queue to write job entries to.
        patch_size: the encoder patch size used when the embeddings were written.
        max_level: deepest pyramid level to write.
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
        pca_store_path=pca_store_path,
        artifact_path=artifact_path,
        source_completed_paths=source_completed_paths,
        completed_path=completed_path,
        patch_size=patch_size,
        max_level=max_level,
    )
    if not jobs:
        logger.info("nothing to render; every source marker already has output")
        return
    rslp.common.worker.write_jobs(
        queue_name, "large_scale_embeddings", "render_pca", jobs
    )


def annotate_pca_store(
    pca_store_path: str,
    artifact_path: str,
    zone_numbers: list[int] | None = None,
    max_level: int = DEFAULT_PCA_MAX_LEVEL,
    storage_options: dict | None = None,
) -> None:
    """Record the basis provenance onto the pca store's arrays.

    The RGB pixels are meaningless without knowing which basis produced them, so the
    artifact metadata is copied onto every pyramid level. Run once after ``fit_pca``.

    Args:
        pca_store_path: the pca store to annotate.
        artifact_path: the fitted artifact whose metadata to record.
        zone_numbers: zones to annotate; defaults to every zone group present.
        max_level: deepest pyramid level to annotate.
        storage_options: fsspec storage options for remote stores.
    """
    artifact = PcaArtifact.load(artifact_path)
    metadata = dict(artifact.metadata)
    metadata["geoemb:pca_artifact_path"] = artifact_path
    metadata["geoemb:pca_annotated_at"] = datetime.now().astimezone().isoformat()

    root = zarr.open_group(
        store=pca_store_path, mode="r+", storage_options=storage_options
    )
    if zone_numbers is None:
        zones = [name for name in root.group_keys() if name.startswith("utm")]
    else:
        zones = [zone_group_name(z) for z in zone_numbers]

    annotated = 0
    for name in zones:
        group = root[name]
        for level in range(max_level + 1):
            array_name = pca_level_array_name(level)
            if array_name not in group:
                continue
            group[array_name].attrs.update(metadata)
            annotated += 1
    # init_pca_store consolidates metadata at creation, and readers prefer that
    # snapshot over each array's own zarr.json. Without re-consolidating, everything
    # written above stays invisible to a default reader.
    zarr.consolidate_metadata(root.store)
    logger.info(
        "annotated %d pca array(s) and re-consolidated metadata in %s",
        annotated,
        pca_store_path,
    )
