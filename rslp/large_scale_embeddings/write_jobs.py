"""Enqueue OlmoEarth embedding prediction jobs on a Beaker queue.

The world is divided into TILE_SIZE x TILE_SIZE UTM tiles. Each tile becomes one
prediction job for a single user-provided reference timestamp. Tiles that don't
intersect their zone's canonical wedge, or that contain no crops to process (e.g.
entirely ocean), are excluded, along with tiles whose completion marker already
exists. Jobs are written to a Beaker queue, where they are processed by rslp.common
workers running the ``predict`` workflow.

The tile size is fixed to 32768x32768 here; the prediction pipeline itself accepts
any tile size that is a multiple of PATCH_SIZE.
"""

import json
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
from rslp.log_utils import get_logger

from . import zarr_store
from .predict_pipeline import (
    EMBEDDING_DIM,
    PATCH_SIZE,
    RESOLUTION,
    EmbeddingInputs,
    get_marker_fname,
)
from .tiling import (
    UTM_MAX_LAT,
    UTM_MIN_LAT,
    bounds_intersect_wedge,
    get_zone_grid,
    get_zone_wedge,
    list_kept_crops,
)

logger = get_logger(__name__)

# Fixed tile size for this job-writer (the prediction pipeline supports any multiple
# of PATCH_SIZE).
TILE_SIZE = 32768


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


def get_jobs(
    inputs: EmbeddingInputs,
    timestamp: datetime,
    store_path: str,
    completed_path: str,
    checkpoint_path: str,
    time_index: int,
    patch_size: int = 1,
    window_size: int = 16,
    overlap_size: int = 4,
    compile_model: bool = True,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    geojson_fname: str | None = None,
    count: int | None = None,
    job_size: int = TILE_SIZE,
) -> list[list[str]]:
    """Get the prediction jobs (one per job_size block).

    Each UTM zone number (1-60) is processed once in its northern CRS (EPSG:326NN),
    spanning both hemispheres. Tiles whose completion markers already exist are
    excluded, along with tiles that don't intersect their zone's canonical wedge or
    contain no crops to process.

    Args:
        inputs: which input variant to use. Different variants produce different
            embeddings so they must use different stores.
        timestamp: the reference timestamp (start of the one-year input period). Must
            have timezone.
        store_path: the GeoZarr store to write embeddings into.
        completed_path: the directory for per-tile completion markers.
        checkpoint_path: the OlmoEarth checkpoint to compute embeddings with.
            Different checkpoints produce different embeddings so they must use
            different store_path/completed_path (same for patch_size, window_size, and
            overlap_size below).
        time_index: the index into the store's time axis for this reference year.
        patch_size: the encoder patch size; yields one embedding per patch_size x
            patch_size pixels.
        window_size: the size of the crops the model operates on.
        overlap_size: overlap in pixels between adjacent crops.
        compile_model: whether to compile the encoder transformer blocks.
        epsg_code: limit tasks to the zone of this UTM EPSG code (326NN or 327NN both
            map to zone NN); default all UTM zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        geojson_fname: limit tasks to tiles intersecting a feature in this GeoJSON
            file (features must be in WGS84 coordinates).
        count: limit to this many tasks (randomly sampled).
        job_size: the pixel size of each job, a divisor of TILE_SIZE and a multiple
            of PATCH_SIZE. Defaults to one job per TILE_SIZE tile. Smaller jobs cost
            more fixed overhead (model load and compile per job) but each finishes
            far sooner, which matters on preemptible workers: a job that outlives the
            gaps between preemptions never completes at all.

    Returns:
        a list of worker argument lists, one per job_size block.
    """
    if job_size % PATCH_SIZE != 0:
        raise ValueError(f"job_size {job_size} must be a multiple of {PATCH_SIZE}")
    if TILE_SIZE % job_size != 0:
        raise ValueError(f"job_size {job_size} must divide TILE_SIZE {TILE_SIZE}")
    if epsg_code:
        zone_numbers = [epsg_code % 100]
    else:
        zone_numbers = list(range(1, 61))

    geojson_shapes: list[shapely.Geometry] | None = None
    if geojson_fname is not None:
        with UPath(geojson_fname).open() as f:
            feature_collection = json.load(f)
        geojson_shapes = [
            shapely.geometry.shape(feature["geometry"])
            for feature in feature_collection["features"]
        ]

    tasks: list[tuple[Projection, PixelBounds]] = []
    for zone_number in tqdm.tqdm(
        zone_numbers, desc="Enumerating tasks across UTM zones"
    ):
        projection, _, _ = get_zone_grid(zone_number, RESOLUTION, TILE_SIZE)
        wedge = get_zone_wedge(projection.crs, RESOLUTION)
        # The zone's WGS84 extent, spanning both hemispheres. Deliberately not
        # get_wgs84_bounds(projection.crs): the projection is the zone's *northern* CRS,
        # so that returns 0..84N and would silently drop every southern shape.
        zone_lon_min = -180 + (zone_number - 1) * 6
        zone_shp = shapely.box(zone_lon_min, UTM_MIN_LAT, zone_lon_min + 6, UTM_MAX_LAT)

        # Intersect the GeoJSON shapes with the WGS84 extent of the current UTM zone
        # and project them into the zone's pixel coordinate system (skipping the zone
        # if no shape intersects it). Reprojecting geometry far outside the zone's
        # extent fails (or yields meaningless bounds), so we only reproject the
        # portion of each shape that falls within the zone.
        zone_geojson_shapes: list[shapely.Geometry] | None = None
        if geojson_shapes is not None:
            zone_geojson_shapes = []
            for shp in geojson_shapes:
                zone_intersect_shp = shp.intersection(zone_shp)
                if zone_intersect_shp.is_empty:
                    continue
                zone_geojson_shapes.append(
                    STGeometry(WGS84_PROJECTION, zone_intersect_shp, None)
                    .to_projection(projection)
                    .shp
                )
            if len(zone_geojson_shapes) == 0:
                continue

        user_bounds_in_proj: PixelBounds | None = None
        if wgs84_bounds is not None:
            # Intersect the user bounds with the zone extent for the same reason as
            # the GeoJSON shapes above.
            intersect_shp = shapely.box(*wgs84_bounds).intersection(zone_shp)
            if intersect_shp.is_empty:
                continue
            dst_geom = STGeometry(WGS84_PROJECTION, intersect_shp, None).to_projection(
                projection
            )
            user_bounds_in_proj = (
                int(dst_geom.shp.bounds[0]),
                int(dst_geom.shp.bounds[1]),
                int(dst_geom.shp.bounds[2]),
                int(dst_geom.shp.bounds[3]),
            )

        for col, row in enumerate_tiles_in_zone(zone_number):
            if user_bounds_in_proj is not None:
                if (col + 1) * TILE_SIZE < user_bounds_in_proj[0]:
                    continue
                if col * TILE_SIZE >= user_bounds_in_proj[2]:
                    continue
                if (row + 1) * TILE_SIZE < user_bounds_in_proj[1]:
                    continue
                if row * TILE_SIZE >= user_bounds_in_proj[3]:
                    continue

            bounds = (
                col * TILE_SIZE,
                row * TILE_SIZE,
                (col + 1) * TILE_SIZE,
                (row + 1) * TILE_SIZE,
            )

            # Skip tiles that don't intersect any GeoJSON feature.
            if zone_geojson_shapes is not None:
                tile_box = shapely.box(*bounds)
                if not any(shp.intersects(tile_box) for shp in zone_geojson_shapes):
                    continue

            # Skip tiles outside the zone's canonical wedge (they are covered by the
            # neighboring UTM zone).
            if not bounds_intersect_wedge(wedge, bounds):
                continue
            # Skip tiles with no crops to process (e.g. entirely ocean).
            if len(list_kept_crops(projection, bounds, PATCH_SIZE, wedge=wedge)) == 0:
                continue

            # Split the tile into job_size blocks. Subdividing the TILE_SIZE grid
            # (rather than re-gridding the zone) keeps every block on the same
            # absolute pixel coordinates the store was created with.
            for sub_x in range(bounds[0], bounds[2], job_size):
                for sub_y in range(bounds[1], bounds[3], job_size):
                    sub_bounds = (sub_x, sub_y, sub_x + job_size, sub_y + job_size)
                    if zone_geojson_shapes is not None and not any(
                        shp.intersects(shapely.box(*sub_bounds))
                        for shp in zone_geojson_shapes
                    ):
                        continue
                    if not bounds_intersect_wedge(wedge, sub_bounds):
                        continue
                    if (
                        len(
                            list_kept_crops(
                                projection, sub_bounds, PATCH_SIZE, wedge=wedge
                            )
                        )
                        == 0
                    ):
                        continue
                    tasks.append((projection, sub_bounds))

    logger.info("Got %d total tasks", len(tasks))

    # Remove tasks where the completion marker already exists.
    completed_upath = UPath(completed_path)
    if completed_upath.exists():
        existing_marker_fnames = {fname.name for fname in completed_upath.iterdir()}
        tasks = [
            (projection, bounds)
            for projection, bounds in tasks
            if get_marker_fname(completed_path, projection, bounds).name
            not in existing_marker_fnames
        ]
    logger.info("Got %d tasks that are uncompleted", len(tasks))

    # Sample down to count if requested.
    if count is not None and len(tasks) > count:
        tasks = random.sample(tasks, count)
        logger.info("Randomly sampled %d tasks", len(tasks))

    # Convert tasks to worker jobs (one per tile).
    time_range_json = json.dumps([timestamp.isoformat(), timestamp.isoformat()])
    jobs = []
    for projection, bounds in tasks:
        cur_args = [
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
            "--window_size",
            str(window_size),
            "--overlap_size",
            str(overlap_size),
            "--compile_model",
            "true" if compile_model else "false",
        ]
        jobs.append(cur_args)

    return jobs


def write_jobs(
    inputs: EmbeddingInputs,
    timestamp: datetime,
    store_path: str,
    completed_path: str,
    queue_name: str,
    checkpoint_path: str,
    patch_size: int = 1,
    window_size: int = 16,
    overlap_size: int = 4,
    compile_model: bool = True,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    geojson_fname: str | None = None,
    count: int | None = None,
    job_size: int = TILE_SIZE,
) -> None:
    """Enumerate tiles for one reference timestamp and write jobs to a Beaker queue.

    The store must already be initialized (see init_store); its time axis determines
    the time index for this timestamp's year.

    Args:
        inputs: which input variant to use. Different variants produce different
            embeddings so they must use different stores.
        timestamp: the reference timestamp (start of the one-year input period). Must
            have timezone.
        store_path: the GeoZarr store to write embeddings into.
        completed_path: the directory for per-tile completion markers.
        queue_name: the Beaker queue to write the job entries to.
        checkpoint_path: the OlmoEarth checkpoint to compute embeddings with.
            Different checkpoints produce different embeddings so they must use
            different store_path/completed_path (same for patch_size, window_size, and
            overlap_size below).
        patch_size: the encoder patch size; yields one embedding per patch_size x
            patch_size pixels.
        window_size: the size of the crops the model operates on.
        overlap_size: overlap in pixels between adjacent crops.
        compile_model: whether to compile the encoder transformer blocks.
        epsg_code: limit tasks to the zone of this UTM EPSG code; default all zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        geojson_fname: limit tasks to tiles intersecting a feature in this GeoJSON
            file (features must be in WGS84 coordinates).
        count: limit to this many tasks (randomly sampled).
        job_size: the pixel size of each job (see get_jobs). Defaults to one job per
            TILE_SIZE tile.
    """
    years = zarr_store.get_store_years(store_path)
    if timestamp.year not in years:
        raise ValueError(
            f"store {store_path} has years {years} but timestamp year "
            f"{timestamp.year} is not among them (run init_store first)"
        )
    time_index = years.index(timestamp.year)

    jobs = get_jobs(
        inputs=inputs,
        timestamp=timestamp,
        store_path=store_path,
        completed_path=completed_path,
        checkpoint_path=checkpoint_path,
        time_index=time_index,
        patch_size=patch_size,
        window_size=window_size,
        overlap_size=overlap_size,
        compile_model=compile_model,
        epsg_code=epsg_code,
        wgs84_bounds=wgs84_bounds,
        geojson_fname=geojson_fname,
        count=count,
        job_size=job_size,
    )
    # Shuffle so outputs start appearing from random parts of the world (aids
    # debugging).
    random.shuffle(jobs)
    rslp.common.worker.write_jobs(queue_name, "large_scale_embeddings", "predict", jobs)


def init_store(
    store_path: str,
    years: list[int],
    model_url: str,
    source_data: list[str],
    zone_numbers: list[int] | None = None,
    patch_size: int = 1,
    band_chunk: int = zarr_store.DEFAULT_BAND_CHUNK,
    matryoshka_dims: list[int] | None = None,
    build_version: str = "0.0.1",
    zstd_level: int = zarr_store.DEFAULT_ZSTD_LEVEL,
    overwrite: bool = False,
) -> None:
    """Initialize the GeoZarr store for a variant before enqueuing prediction jobs.

    Creates the root group and one group per UTM zone with an empty sharded int8
    embedding array spanning the given years. Run once per store, before write_jobs.

    Args:
        store_path: the GeoZarr store path or URL to create.
        years: the annual reference years, defining the time axis.
        model_url: URL reference to the encoder model (e.g. a HuggingFace repo).
        source_data: URLs of the source datasets.
        zone_numbers: the UTM zone numbers to create; defaults to all of 1-60.
        patch_size: the encoder patch size; the store grid is at 1/patch_size of the
            input resolution, so it must match the patch_size used by write_jobs.
        band_chunk: dimensions per inner chunk along the band axis. Makes Matryoshka
            prefix reads proportionally cheaper at negligible storage cost.
        matryoshka_dims: prefix widths the model supports, recorded in the store's
            provenance so a reader knows which truncations are valid.
        build_version: version of the software that built the store.
        zstd_level: zstd compression level for the arrays.
        overwrite: whether to overwrite an existing store.
    """
    if zone_numbers is None:
        zone_numbers = list(range(1, 61))
    if PATCH_SIZE % patch_size != 0:
        raise ValueError(f"patch_size must divide {PATCH_SIZE}, got {patch_size}")
    # The store grid is at the output (embedding) resolution, which is 1/patch_size of
    # the input resolution. One output window (= one shard) is PATCH_SIZE / patch_size
    # pixels, and the tile size scales down the same way.
    output_resolution = RESOLUTION * patch_size
    output_tile_size = TILE_SIZE // patch_size
    output_shard_size = PATCH_SIZE // patch_size
    zarr_store.init_store(
        store_path=store_path,
        zone_numbers=zone_numbers,
        years=years,
        model_url=model_url,
        source_data=source_data,
        resolution=output_resolution,
        tile_size=output_tile_size,
        dimensions=EMBEDDING_DIM,
        band_chunk=band_chunk,
        matryoshka_dims=matryoshka_dims,
        chunk_size=min(zarr_store.DEFAULT_CHUNK_SIZE, output_shard_size),
        shard_size=output_shard_size,
        zstd_level=zstd_level,
        build_version=build_version,
        overwrite=overwrite,
    )
