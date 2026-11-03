"""Enqueue OlmoEarth LCC prediction jobs on a Beaker queue.

The world is divided into TILE_SIZE x TILE_SIZE tiles, one UTM zone number at a time,
each in its northern CRS (EPSG:326NN) spanning both hemispheres (see tiling.py). Only
tiles intersecting their zone's canonical wedge are kept, so each location is
processed in (nearly) one zone. Each tile becomes one prediction task for a single
user-provided reference timestamp. Tasks are batched and written to a Beaker queue,
where they are processed by rslp.common workers running the ``predict_multi``
workflow.

The tile size is fixed to 16384x16384 here; the prediction pipeline itself accepts any
tile size that is a multiple of PATCH_SIZE.
"""

import json
import random
from datetime import datetime

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_wgs84_bounds
from upath import UPath

import rslp.common.worker
from rslp.log_utils import get_logger

from . import geozarr
from .predict_pipeline import (
    MODEL_CONFIG_FNAME,
    PredictTaskArgs,
    get_output_fname,
)
from .tiling import (
    RESOLUTION,
    TILE_SIZE,
    UTM_MAX_LAT,
    UTM_MIN_LAT,
    enumerate_zone_tiles,
    get_zone_projection,
    zone_number_from_epsg,
)

logger = get_logger(__name__)

# Densify the user's WGS84 box to this many degrees per segment before reprojecting,
# since straight edges in WGS84 are curved in UTM.
WGS84_SEGMENT_DEGREES = 0.1


def _user_bounds_in_zone(
    zone_number: int, wgs84_bounds: tuple[float, float, float, float]
) -> PixelBounds | None:
    """Get the pixel bounds of the user's WGS84 box within one zone's wedge.

    Args:
        zone_number: the UTM zone number (1-60).
        wgs84_bounds: the user's (min_lon, min_lat, max_lon, max_lat).

    Returns:
        the pixel bounds in the zone's northern projection, or None if the box does
        not intersect the zone's wedge.
    """
    projection = get_zone_projection(zone_number, RESOLUTION)
    min_lon, _, max_lon, _ = get_wgs84_bounds(projection.crs)
    zone_shp = shapely.box(min_lon, UTM_MIN_LAT, max_lon, UTM_MAX_LAT)
    intersect_shp = shapely.box(*wgs84_bounds).intersection(zone_shp)
    if intersect_shp.is_empty:
        return None
    intersect_shp = shapely.segmentize(intersect_shp, WGS84_SEGMENT_DEGREES)
    dst_geom = STGeometry(WGS84_PROJECTION, intersect_shp, None).to_projection(
        projection
    )
    return (
        int(dst_geom.shp.bounds[0]),
        int(dst_geom.shp.bounds[1]),
        int(dst_geom.shp.bounds[2]),
        int(dst_geom.shp.bounds[3]),
    )


def get_tasks(
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
) -> list[tuple[Projection, PixelBounds]]:
    """Enumerate the TILE_SIZE tiles to process.

    Args:
        epsg_code: limit tasks to this UTM zone, given as its 326NN or 327NN EPSG
            code (both select the same zone); default all UTM zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.

    Returns:
        list of (projection, bounds) of the tiles.
    """
    if epsg_code:
        zone_numbers = [zone_number_from_epsg(epsg_code)]
    else:
        zone_numbers = list(range(1, 61))

    tasks: list[tuple[Projection, PixelBounds]] = []
    for zone_number in tqdm.tqdm(zone_numbers, desc="Enumerating tasks across zones"):
        projection = get_zone_projection(zone_number, RESOLUTION)

        user_bounds_in_proj: PixelBounds | None = None
        if wgs84_bounds is not None:
            user_bounds_in_proj = _user_bounds_in_zone(zone_number, wgs84_bounds)
            if user_bounds_in_proj is None:
                continue

        for col, row in enumerate_zone_tiles(zone_number, RESOLUTION, TILE_SIZE):
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
            tasks.append((projection, bounds))

    return tasks


def get_jobs(
    timestamp: datetime,
    out_path: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    batch_size: int = 1,
    count: int | None = None,
    write_raster: bool = False,
    write_summary_raster: bool = False,
    zarr_store_path: str | None = None,
    write_geojson: bool = True,
) -> list[list[str]]:
    """Get batches of prediction tasks.

    Tasks whose outputs already exist are excluded: the per-tile GeoJSON normally,
    or the per-tile ``_meta.json`` (written last) when writing to a GeoZarr store.

    Args:
        timestamp: the reference timestamp. Must have timezone.
        out_path: the output directory.
        epsg_code: limit tasks to this UTM zone (326NN or 327NN EPSG code); default
            all UTM zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        batch_size: how many tasks to run per worker job.
        count: limit to this many tasks (randomly sampled).
        write_raster: whether workers should also write the full merged raster.
        write_summary_raster: whether workers should also write the compact uint8
            summary raster.
        zarr_store_path: GeoZarr store that workers should write into.
        write_geojson: whether workers should write the per-tile GeoJSON.

    Returns:
        a list of worker argument lists, one per batch of TILE_SIZE tiles.
    """
    time_range = (timestamp, timestamp)
    tasks = get_tasks(epsg_code=epsg_code, wgs84_bounds=wgs84_bounds)
    logger.info("Got %d total tasks", len(tasks))

    # Remove tasks where outputs are already computed.
    out_upath = UPath(out_path)
    if out_upath.exists():
        existing_output_fnames = {fname.name for fname in out_upath.iterdir()}
        tasks = [
            (projection, bounds)
            for projection, bounds in tasks
            if get_output_fname(
                out_path, projection, bounds, meta=zarr_store_path is not None
            ).name
            not in existing_output_fnames
        ]
    logger.info("Got %d tasks that are uncompleted", len(tasks))

    # Sample down to count if requested.
    if count is not None and len(tasks) > count:
        tasks = random.sample(tasks, count)
        logger.info("Randomly sampled %d tasks", len(tasks))

    # Convert tasks to batched worker jobs.
    jobs = []
    for i in range(0, len(tasks), batch_size):
        cur_tasks = tasks[i : i + batch_size]
        predict_tasks = [
            PredictTaskArgs(
                projection_json=projection.serialize(),
                bounds=bounds,
                time_range=time_range,
            )
            for projection, bounds in cur_tasks
        ]
        cur_args = [
            "--out_path",
            out_path,
            "--scratch_path",
            "/tmp/scratch/",
            "--tasks",
            json.dumps([predict_task.serialize() for predict_task in predict_tasks]),
        ]
        if write_raster:
            cur_args += ["--write_raster", "true"]
        if write_summary_raster:
            cur_args += ["--write_summary_raster", "true"]
        if zarr_store_path is not None:
            cur_args += ["--zarr_store_path", zarr_store_path]
        if not write_geojson:
            cur_args += ["--write_geojson", "false"]
        jobs.append(cur_args)

    return jobs


def init_geozarr(
    store_path: str,
    timestamp: datetime,
    chunk_size: int = geozarr.DEFAULT_CHUNK_SIZE,
    compressor: str = geozarr.DEFAULT_COMPRESSOR,
    level: int = geozarr.DEFAULT_COMPRESSION_LEVEL,
    overwrite: bool = False,
) -> None:
    """Create a GeoZarr store for one reference timestamp with custom settings.

    write_jobs creates the store with default settings if it is missing, so this is
    only needed to use a non-default chunk size or compressor.

    Args:
        store_path: the Zarr store path or URL.
        timestamp: the reference timestamp. Must have timezone.
        chunk_size: inner chunk spatial size; must divide the 2048 shard size.
        compressor: one of geozarr.COMPRESSORS.
        level: the compression level.
        overwrite: whether to overwrite an existing store.
    """
    geozarr.init_store(
        store_path,
        timestamp,
        model_config=MODEL_CONFIG_FNAME,
        chunk_size=chunk_size,
        compressor=compressor,
        level=level,
        overwrite=overwrite,
    )


def write_jobs(
    timestamp: datetime,
    out_path: str,
    queue_name: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    batch_size: int = 1,
    count: int | None = None,
    write_raster: bool = False,
    write_summary_raster: bool = False,
    zarr_store_path: str | None = None,
    write_geojson: bool = True,
) -> None:
    """Enumerate tiles for one reference timestamp and write jobs to a Beaker queue.

    Args:
        timestamp: the reference timestamp. Must have timezone.
        out_path: the output directory.
        queue_name: the Beaker queue to write the job entries to.
        epsg_code: limit tasks to this UTM zone (326NN or 327NN EPSG code); default
            all UTM zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        batch_size: how many tasks to run per worker job.
        count: limit to this many tasks (randomly sampled).
        write_raster: whether workers should also write the full merged raster.
        write_summary_raster: whether workers should also write the compact uint8
            summary raster.
        zarr_store_path: also write outputs into this GeoZarr store (one store per
            reference timestamp). It is created with default settings if missing; an
            existing store must have been created for the same timestamp.
        write_geojson: whether workers should write the per-tile GeoJSON.
    """
    if zarr_store_path is None and not write_geojson:
        raise ValueError("write_geojson=false requires zarr_store_path")
    if zarr_store_path is not None:
        geozarr.ensure_store(
            zarr_store_path, timestamp, model_config=MODEL_CONFIG_FNAME
        )

    jobs = get_jobs(
        timestamp=timestamp,
        out_path=out_path,
        epsg_code=epsg_code,
        wgs84_bounds=wgs84_bounds,
        batch_size=batch_size,
        count=count,
        write_raster=write_raster,
        write_summary_raster=write_summary_raster,
        zarr_store_path=zarr_store_path,
        write_geojson=write_geojson,
    )
    # Shuffle so outputs start appearing from random parts of the world (aids debugging).
    random.shuffle(jobs)
    rslp.common.worker.write_jobs(queue_name, "olmoearth_lcc", "predict_multi", jobs)
