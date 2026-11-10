"""Enqueue OlmoEarth embedding prediction jobs on a Beaker queue.

Each UTM zone is divided into TILE_SIZE tiles, and each tile into job_size blocks. A
block becomes one ``predict`` job unless it lies outside its zone's wedge, has no
covered crops, or already has a completion marker.
"""

import concurrent.futures
import functools
import json
import multiprocessing
import os
import random
from collections.abc import Generator
from datetime import datetime

import shapely
import shapely.geometry
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from upath import UPath

import rslp.common.worker
from rslp.large_scale_embeddings import zarr_store
from rslp.large_scale_embeddings.predict_pipeline import (
    EMBEDDING_DIM,
    PATCH_SIZE,
    PREFETCH,
    RESOLUTION,
    EmbeddingInputs,
    get_marker_fname,
)
from rslp.large_scale_embeddings.tiling import (
    UTM_MAX_LAT,
    UTM_MIN_LAT,
    bounds_intersect_wedge,
    get_zone_grid,
    get_zone_wedge,
    list_kept_crops,
)
from rslp.log_utils import get_logger

logger = get_logger(__name__)

# The zone grid is snapped to this; job_size must divide it.
TILE_SIZE = 32768
DEFAULT_JOB_SIZE = 8192

# Cap on enumeration processes. Each loads its own coverage mask, and in a container
# os.cpu_count() reports the host's cores rather than the job's.
ENUMERATION_MAX_PROCESSES = 4
# Fewer zones than this are enumerated inline.
ENUMERATION_POOL_MIN_ZONES = 8

# Longest edge segment, in degrees, of a footprint before it is reprojected, so its
# edges stay close to the curves they stand for in UTM.
ZONE_EDGE_SEGMENT_DEG = 0.1


def enumerate_tiles_in_zone(zone_number: int) -> Generator[tuple[int, int], None, None]:
    """List the (column, row) of all TILE_SIZE tiles within a UTM zone.

    Args:
        zone_number: the UTM zone number (1-60).

    Returns:
        generator of (column, row) of the tiles that are needed.
    """
    _, (origin_x, origin_y), (height, width) = get_zone_grid(
        zone_number, RESOLUTION, TILE_SIZE
    )
    for col in range(origin_x // TILE_SIZE, (origin_x + width) // TILE_SIZE):
        for row in range(origin_y // TILE_SIZE, (origin_y + height) // TILE_SIZE):
            yield (col, row)


def _zone_extent(zone_number: int) -> shapely.Polygon:
    """The zone's WGS84 extent across both hemispheres."""
    zone_lon_min = -180 + (zone_number - 1) * 6
    return shapely.box(zone_lon_min, UTM_MIN_LAT, zone_lon_min + 6, UTM_MAX_LAT)


def _enumerate_zone(
    zone_number: int,
    job_size: int,
    geojson_shapes: list[shapely.Geometry] | None,
    wgs84_bounds: tuple[float, float, float, float] | None,
) -> list[tuple[Projection, PixelBounds]]:
    """Enumerate one UTM zone's blocks. See `enumerate_blocks`.

    Args:
        zone_number: the UTM zone number (1-60).
        job_size: the pixel size of each block.
        geojson_shapes: footprint restriction in WGS84, if any.
        wgs84_bounds: bounding box restriction, if any.

    Returns:
        (projection, bounds) per block in the zone.
    """
    tasks: list[tuple[Projection, PixelBounds]] = []
    projection, _, _ = get_zone_grid(zone_number, RESOLUTION, TILE_SIZE)
    wedge = get_zone_wedge(projection.crs, RESOLUTION)
    zone_shp = _zone_extent(zone_number)

    # Clip restrictions to the zone before reprojecting, since geometry far outside a
    # zone does not reproject meaningfully into it.
    zone_geojson_shapes: list[shapely.Geometry] | None = None
    if geojson_shapes is not None:
        zone_geojson_shapes = []
        for shp in geojson_shapes:
            zone_intersect_shp = shp.intersection(zone_shp)
            if zone_intersect_shp.is_empty:
                continue
            zone_intersect_shp = shapely.segmentize(
                zone_intersect_shp, max_segment_length=ZONE_EDGE_SEGMENT_DEG
            )
            zone_geojson_shapes.append(
                STGeometry(WGS84_PROJECTION, zone_intersect_shp, None)
                .to_projection(projection)
                .shp
            )
        if len(zone_geojson_shapes) == 0:
            return tasks

    user_bounds_in_proj: PixelBounds | None = None
    if wgs84_bounds is not None:
        intersect_shp = shapely.box(*wgs84_bounds).intersection(zone_shp)
        if intersect_shp.is_empty:
            return tasks
        dst_geom = STGeometry(WGS84_PROJECTION, intersect_shp, None).to_projection(
            projection
        )
        user_bounds_in_proj = (
            int(dst_geom.shp.bounds[0]),
            int(dst_geom.shp.bounds[1]),
            int(dst_geom.shp.bounds[2]),
            int(dst_geom.shp.bounds[3]),
        )

    def in_area(bounds: PixelBounds) -> bool:
        if user_bounds_in_proj is not None and (
            bounds[2] < user_bounds_in_proj[0]
            or bounds[0] >= user_bounds_in_proj[2]
            or bounds[3] < user_bounds_in_proj[1]
            or bounds[1] >= user_bounds_in_proj[3]
        ):
            return False
        if zone_geojson_shapes is None:
            return True
        box = shapely.box(*bounds)
        return any(shp.intersects(box) for shp in zone_geojson_shapes)

    for col, row in enumerate_tiles_in_zone(zone_number):
        bounds = (
            col * TILE_SIZE,
            row * TILE_SIZE,
            (col + 1) * TILE_SIZE,
            (row + 1) * TILE_SIZE,
        )
        if not in_area(bounds) or not bounds_intersect_wedge(wedge, bounds):
            continue
        tile_crops = list_kept_crops(projection, bounds, PATCH_SIZE, wedge=wedge)
        if len(tile_crops) == 0:
            continue

        for sub_x in range(bounds[0], bounds[2], job_size):
            for sub_y in range(bounds[1], bounds[3], job_size):
                sub_bounds = (sub_x, sub_y, sub_x + job_size, sub_y + job_size)
                if not in_area(sub_bounds):
                    continue
                if not bounds_intersect_wedge(wedge, sub_bounds):
                    continue
                if not any(
                    sub_bounds[0] <= crop[0] < sub_bounds[2]
                    and sub_bounds[1] <= crop[1] < sub_bounds[3]
                    for crop in tile_crops
                ):
                    continue
                tasks.append((projection, sub_bounds))
    return tasks


def _zone_reaches(
    zone_number: int,
    geojson_shapes: list[shapely.Geometry] | None,
    wgs84_bounds: tuple[float, float, float, float] | None,
) -> bool:
    """Whether a zone's extent meets the area restriction.

    Args:
        zone_number: the UTM zone number (1-60).
        geojson_shapes: footprint restriction in WGS84, if any.
        wgs84_bounds: bounding box restriction, if any.

    Returns:
        False only when the zone can have no blocks.
    """
    zone_shp = _zone_extent(zone_number)
    if geojson_shapes is not None and not any(
        shp.intersects(zone_shp) for shp in geojson_shapes
    ):
        return False
    return wgs84_bounds is None or shapely.box(*wgs84_bounds).intersects(zone_shp)


def enumerate_blocks(
    job_size: int = DEFAULT_JOB_SIZE,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    geojson_fname: str | None = None,
    blocks_fname: str | None = None,
) -> list[tuple[Projection, PixelBounds]]:
    """Every block the run covers, whether or not it has a marker yet.

    Args:
        job_size: the pixel size of each block.
        epsg_code: limit to the zone of this UTM EPSG code.
        wgs84_bounds: limit to blocks intersecting these WGS84 bounds.
        geojson_fname: limit to blocks intersecting a feature in this GeoJSON file.
        blocks_fname: limit to the blocks named in this JSON list of `block_id`s. Use
            this for an exact selection: neighboring zones' blocks overlap on the
            ground, so a footprint selects blocks in both.

    Returns:
        (projection, bounds) per block.
    """
    zone_numbers = [epsg_code % 100] if epsg_code else list(range(1, 61))

    geojson_shapes: list[shapely.Geometry] | None = None
    if geojson_fname is not None:
        with UPath(geojson_fname).open() as f:
            feature_collection = json.load(f)
        geojson_shapes = [
            shapely.geometry.shape(feature["geometry"])
            for feature in feature_collection["features"]
        ]

    zone_numbers = [
        zone
        for zone in zone_numbers
        if _zone_reaches(zone, geojson_shapes, wgs84_bounds)
    ]
    tasks: list[tuple[Projection, PixelBounds]] = []
    if len(zone_numbers) < ENUMERATION_POOL_MIN_ZONES:
        for zone in zone_numbers:
            tasks.extend(_enumerate_zone(zone, job_size, geojson_shapes, wgs84_bounds))
    else:
        # Spawned rather than forked, so a caller holding gRPC channels cannot hang.
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(
                len(zone_numbers), os.cpu_count() or 1, ENUMERATION_MAX_PROCESSES
            ),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            for zone_tasks in tqdm.tqdm(
                pool.map(
                    functools.partial(
                        _enumerate_zone,
                        job_size=job_size,
                        geojson_shapes=geojson_shapes,
                        wgs84_bounds=wgs84_bounds,
                    ),
                    zone_numbers,
                ),
                total=len(zone_numbers),
                desc="Enumerating tasks across UTM zones",
            ):
                tasks.extend(zone_tasks)

    logger.info("Got %d total tasks", len(tasks))
    if blocks_fname is not None:
        tasks = _filter_to_blocks(tasks, blocks_fname)
    return tasks


def block_id(projection: Projection, bounds: PixelBounds) -> str:
    """Name a block the way its completion marker is named, without the extension.

    Args:
        projection: the block's projection.
        bounds: the block's pixel bounds.

    Returns:
        "{crs}_{x}_{y}", unique per block within one job_size.
    """
    return f"{projection.crs!s}_{bounds[0]}_{bounds[1]}"


def _filter_to_blocks(
    tasks: list[tuple[Projection, PixelBounds]], blocks_fname: str
) -> list[tuple[Projection, PixelBounds]]:
    """Keep only the blocks a list names.

    Args:
        tasks: the enumerated blocks.
        blocks_fname: JSON list of `block_id`s.

    Returns:
        the named blocks, in enumeration order.

    Raises:
        ValueError: if the list names a block the enumeration does not have, which
            means it was built for another job_size or coverage mask.
    """
    with UPath(blocks_fname).open() as f:
        wanted = set(json.load(f))
    kept = [task for task in tasks if block_id(*task) in wanted]
    if len(kept) != len(wanted):
        raise ValueError(
            f"{blocks_fname} names {len(wanted) - len(kept)} block(s) this enumeration "
            "does not have; it was likely built for another job_size or coverage mask"
        )
    logger.info("Kept %d of %d tasks named in %s", len(kept), len(tasks), blocks_fname)
    return kept


def _check_marker_job_size(completed_upath: UPath, job_size: int) -> None:
    """Check that existing markers were written for this job_size.

    Marker names hold only a block's corner, so blocks of different sizes sharing a
    corner share a name too.

    Args:
        completed_upath: the marker directory.
        job_size: the job size of this run.

    Raises:
        ValueError: if a marker records a different block size.
    """
    marker = next((f for f in completed_upath.iterdir() if f.suffix == ".json"), None)
    if marker is None:
        return
    with marker.open() as f:
        bounds = json.load(f)["bounds"]
    if bounds[2] - bounds[0] != job_size:
        raise ValueError(
            f"markers in {completed_upath} are for job_size {bounds[2] - bounds[0]}, "
            f"not {job_size}; use a separate completed_path per job_size"
        )


def get_jobs(
    inputs: EmbeddingInputs,
    timestamp: datetime,
    store_path: str,
    completed_path: str,
    checkpoint_path: str,
    time_index: int,
    patch_size: int = 2,
    location_query_patch_size: int = 1,
    window_size: int = 480,
    overlap_size: int = 32,
    context_pixels: int = 48,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    geojson_fname: str | None = None,
    blocks_fname: str | None = None,
    count: int | None = None,
    job_size: int = DEFAULT_JOB_SIZE,
) -> list[list[str]]:
    """Get the prediction jobs for every uncompleted block.

    Args:
        inputs: which input variant to use.
        timestamp: the reference timestamp (start of the one-year input period). Must
            have a timezone.
        store_path: the GeoZarr store to write embeddings into.
        completed_path: the directory for completion markers.
        checkpoint_path: the OlmoEarth checkpoint or released model bundle.
        time_index: the index into the store's time axis for this reference year.
        patch_size: the encoder's token patch size.
        location_query_patch_size: input pixels per output embedding.
        window_size: the size of the crops the model operates on.
        overlap_size: overlap in pixels between adjacent crops.
        context_pixels: imagery margin around each window that is read but not
            written.
        epsg_code: limit tasks to the zone of this UTM EPSG code; default all zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        geojson_fname: limit tasks to blocks intersecting a feature in this WGS84
            GeoJSON file.
        blocks_fname: limit tasks to the blocks named in this JSON list of
            `block_id`s. See `enumerate_blocks`.
        count: limit to this many tasks (randomly sampled).
        job_size: the pixel size of each block; a divisor of TILE_SIZE and a multiple
            of PATCH_SIZE.

    Returns:
        a list of worker argument lists, one per block.
    """
    if job_size % PATCH_SIZE != 0:
        raise ValueError(f"job_size {job_size} must be a multiple of {PATCH_SIZE}")
    if TILE_SIZE % job_size != 0:
        raise ValueError(f"job_size {job_size} must divide TILE_SIZE {TILE_SIZE}")
    tasks = enumerate_blocks(
        job_size=job_size,
        epsg_code=epsg_code,
        wgs84_bounds=wgs84_bounds,
        geojson_fname=geojson_fname,
        blocks_fname=blocks_fname,
    )

    completed_upath = UPath(completed_path)
    if completed_upath.exists():
        _check_marker_job_size(completed_upath, job_size)
        existing_marker_fnames = {fname.name for fname in completed_upath.iterdir()}
        tasks = [
            (projection, bounds)
            for projection, bounds in tasks
            if get_marker_fname(completed_path, projection, bounds).name
            not in existing_marker_fnames
        ]
    logger.info("Got %d tasks that are uncompleted", len(tasks))

    if count is not None and len(tasks) > count:
        tasks = random.sample(tasks, count)
        logger.info("Randomly sampled %d tasks", len(tasks))

    time_range_json = json.dumps([timestamp.isoformat(), timestamp.isoformat()])
    return [
        [
            "--inputs",
            inputs.name,
            "--projection_json",
            json.dumps(projection.serialize()),
            "--bounds",
            json.dumps(list(bounds)),
            "--time_range",
            time_range_json,
            "--store_path",
            store_path,
            "--completed_path",
            completed_path,
            "--checkpoint_path",
            checkpoint_path,
            "--time_index",
            str(time_index),
            "--patch_size",
            str(patch_size),
            "--location_query_patch_size",
            str(location_query_patch_size),
            "--window_size",
            str(window_size),
            "--overlap_size",
            str(overlap_size),
            "--context_pixels",
            str(context_pixels),
        ]
        for projection, bounds in tasks
    ]


def write_jobs(
    inputs: EmbeddingInputs,
    timestamp: datetime,
    store_path: str,
    completed_path: str,
    queue_name: str,
    checkpoint_path: str,
    patch_size: int = 2,
    location_query_patch_size: int = 1,
    window_size: int = 480,
    overlap_size: int = 32,
    context_pixels: int = 48,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    geojson_fname: str | None = None,
    blocks_fname: str | None = None,
    count: int | None = None,
    job_size: int = DEFAULT_JOB_SIZE,
) -> None:
    """Enumerate blocks for one reference timestamp and write jobs to a Beaker queue.

    The store must already exist (see init_store) and include the timestamp's year.

    Args:
        inputs: which input variant to use.
        timestamp: the reference timestamp (start of the one-year input period). Must
            have a timezone.
        store_path: the GeoZarr store to write embeddings into.
        completed_path: the directory for completion markers.
        queue_name: the Beaker queue to write the job entries to.
        checkpoint_path: the OlmoEarth checkpoint or released model bundle.
        patch_size: the encoder's token patch size.
        location_query_patch_size: input pixels per output embedding.
        window_size: the size of the crops the model operates on.
        overlap_size: overlap in pixels between adjacent crops.
        context_pixels: imagery margin around each window that is read but not
            written.
        epsg_code: limit tasks to the zone of this UTM EPSG code; default all zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        geojson_fname: limit tasks to blocks intersecting a feature in this WGS84
            GeoJSON file.
        blocks_fname: limit tasks to the blocks named in this JSON list of
            `block_id`s. See `enumerate_blocks`.
        count: limit to this many tasks (randomly sampled).
        job_size: the pixel size of each block (see get_jobs).
    """
    years = zarr_store.get_store_years(store_path)
    if timestamp.year not in years:
        raise ValueError(
            f"store {store_path} has years {years} but timestamp year "
            f"{timestamp.year} is not among them (run init_store first)"
        )

    jobs = get_jobs(
        inputs=inputs,
        timestamp=timestamp,
        store_path=store_path,
        completed_path=completed_path,
        checkpoint_path=checkpoint_path,
        time_index=years.index(timestamp.year),
        patch_size=patch_size,
        location_query_patch_size=location_query_patch_size,
        window_size=window_size,
        overlap_size=overlap_size,
        context_pixels=context_pixels,
        epsg_code=epsg_code,
        wgs84_bounds=wgs84_bounds,
        geojson_fname=geojson_fname,
        blocks_fname=blocks_fname,
        count=count,
        job_size=job_size,
    )
    # Shuffled so output appears across the world rather than zone by zone.
    random.shuffle(jobs)
    rslp.common.worker.write_jobs(
        queue_name, "large_scale_embeddings", "predict", jobs, prefetch=PREFETCH
    )


def init_store(
    store_path: str,
    years: list[int],
    model_url: str,
    license_url: str,
    matryoshka_dims: list[int] = zarr_store.DEFAULT_MATRYOSHKA_DIMS,
    inputs: EmbeddingInputs = EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
    source_data: list[str] | None = None,
    zone_numbers: list[int] | None = None,
    location_query_patch_size: int = 1,
    band_chunk: int = zarr_store.DEFAULT_BAND_CHUNK,
    build_version: str | None = None,
    zstd_level: int = zarr_store.DEFAULT_ZSTD_LEVEL,
    overwrite: bool = False,
) -> None:
    """Create the GeoZarr store before enqueuing prediction jobs.

    Args:
        store_path: the GeoZarr store path or URL to create.
        years: the annual reference years, defining the time axis.
        model_url: what produced the embeddings, recorded as geoemb:model: a
            HuggingFace URL for a release, or the checkpoint path.
        license_url: URL of the licence text.
        matryoshka_dims: prefix widths the encoder is trained to emit.
        inputs: the input variant, used to record the source datasets.
        source_data: URLs of the source datasets. Derived from `inputs` if unset.
        zone_numbers: the UTM zone numbers to create; defaults to all of 1-60.
        location_query_patch_size: input pixels per output embedding. Sets the
            store's grid, so it must match the run.
        band_chunk: dimensions per inner chunk along the band axis.
        build_version: version of the software that built the store; detected if
            unset.
        zstd_level: zstd compression level for the arrays.
        overwrite: whether to overwrite an existing store.
    """
    if zone_numbers is None:
        zone_numbers = list(range(1, 61))
    if PATCH_SIZE % location_query_patch_size != 0:
        raise ValueError(
            f"location_query_patch_size must divide {PATCH_SIZE}, got "
            f"{location_query_patch_size}"
        )
    # The store grid is at the output resolution; one window is one shard.
    output_shard_size = PATCH_SIZE // location_query_patch_size
    zarr_store.init_store(
        store_path=store_path,
        zone_numbers=zone_numbers,
        years=years,
        model_url=model_url,
        license_url=license_url,
        source_data=(
            source_data
            if source_data is not None
            else zarr_store.source_data_for(inputs.value)
        ),
        resolution=RESOLUTION * location_query_patch_size,
        tile_size=TILE_SIZE // location_query_patch_size,
        dimensions=EMBEDDING_DIM,
        band_chunk=band_chunk,
        matryoshka_dims=matryoshka_dims,
        chunk_size=min(zarr_store.DEFAULT_CHUNK_SIZE, output_shard_size),
        shard_size=output_shard_size,
        zstd_level=zstd_level,
        build_version=build_version,
        overwrite=overwrite,
    )
