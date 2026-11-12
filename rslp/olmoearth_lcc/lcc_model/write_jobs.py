"""Write OlmoEarth LCC prediction tasks to a job directory.

The world is divided into TILE_SIZE x TILE_SIZE tiles, one UTM zone number at a time,
each in its northern CRS (EPSG:326NN) spanning both hemispheres (see tiling.py). Only
tiles intersecting their zone's canonical wedge are kept, so each location is
processed in (nearly) one zone. Each tile becomes one prediction task for a single
user-provided reference timestamp. Tasks are written to a job directory (see
job_dir.py), where workers (see worker.py) claim and predict them one tile at a
time.

The tile size is fixed to 16384x16384 here; the prediction pipeline itself accepts any
tile size that is a multiple of PATCH_SIZE.
"""

from datetime import datetime

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_wgs84_bounds

from rslp.log_utils import get_logger

from . import geozarr, geozarr_pyramid
from .job_dir import JobConfig, PredictTaskArgs, check_local_path, write_job_dir
from .predict_pipeline import MODEL_CONFIG_FNAME
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


def write_tasks(
    job_dir: str,
    tasks: list[PredictTaskArgs],
    out_path: str,
    zarr_store_path: str,
    zarr_detail: str = geozarr.DETAIL_COMPACT,
    zarr_pyramid_path: str | None = None,
    count: int | None = None,
    fast_inference: bool = True,
) -> None:
    """Create the output stores if needed and write tasks to a job directory.

    Tiles whose ``_meta.json`` already exists in out_path, or that already have a
    task file, are skipped.

    Args:
        job_dir: the job directory (see job_dir.py), a local POSIX path that the
            workers mount.
        tasks: the tasks to write.
        out_path: the output directory for the per-tile _meta.json.
        zarr_store_path: the GeoZarr store, created if missing; an existing store
            must have been created with the same detail mode.
        zarr_detail: the detail mode of the store, one of geozarr.DETAILS.
        zarr_pyramid_path: also write each tile's overviews into this pyramid store
            (see geozarr_pyramid.py), created if missing.
        count: write at most this many of the remaining tasks (randomly sampled).
        fast_inference: whether workers should use fast inference (compiled
            encoder with cuDNN attention, see fast_inference.py); set false to run
            the model eagerly.
    """
    check_local_path(job_dir)
    geozarr.ensure_store(
        zarr_store_path, model_config=MODEL_CONFIG_FNAME, detail=zarr_detail
    )
    if zarr_pyramid_path is not None:
        geozarr_pyramid.ensure_pyramid_store(zarr_pyramid_path, zarr_store_path)
    config = JobConfig(
        out_path=out_path,
        zarr_store_path=zarr_store_path,
        zarr_pyramid_path=zarr_pyramid_path,
        fast_inference=fast_inference,
    )
    write_job_dir(job_dir, config, tasks, count=count)


def write_jobs(
    timestamp: datetime,
    out_path: str,
    job_dir: str,
    zarr_store_path: str,
    zarr_detail: str = geozarr.DETAIL_COMPACT,
    zarr_pyramid_path: str | None = None,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    count: int | None = None,
    fast_inference: bool = True,
) -> None:
    """Enumerate tiles for one reference timestamp and write them to a job directory.

    Tiles whose ``_meta.json`` already exists in out_path, or that already have a
    task file, are skipped, so re-running resumes or extends the job.

    Args:
        timestamp: the reference timestamp. Must have timezone.
        out_path: the output directory for the per-tile _meta.json.
        job_dir: the job directory (see job_dir.py), a local POSIX path that the
            workers mount.
        zarr_store_path: write outputs into this GeoZarr store (normally one store
            per reference timestamp). It is created if missing; an existing store
            must have been created with the same detail mode.
        zarr_detail: the detail mode of the store, one of geozarr.DETAILS.
        zarr_pyramid_path: also write each tile's overviews into this pyramid store
            (see geozarr_pyramid.py), created if missing.
        epsg_code: limit tasks to this UTM zone (326NN or 327NN EPSG code); default
            all UTM zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        count: write at most this many of the remaining tasks (randomly sampled).
        fast_inference: whether workers should use fast inference (compiled
            encoder with cuDNN attention, see fast_inference.py); set false to run
            the model eagerly.
    """
    tasks = get_tasks(epsg_code=epsg_code, wgs84_bounds=wgs84_bounds)
    logger.info("Got %d total tasks", len(tasks))
    write_tasks(
        job_dir,
        [
            PredictTaskArgs(
                projection=projection, bounds=bounds, time_range=(timestamp, timestamp)
            )
            for projection, bounds in tasks
        ],
        out_path=out_path,
        zarr_store_path=zarr_store_path,
        zarr_detail=zarr_detail,
        zarr_pyramid_path=zarr_pyramid_path,
        count=count,
        fast_inference=fast_inference,
    )
