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
from rasterio.crs import CRS
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_proj_bounds
from upath import UPath

import rslp.common.worker
from rslp.log_utils import get_logger

from .predict_pipeline import (
    PATCH_SIZE,
    RESOLUTION,
    EmbeddingInputs,
    get_marker_fname,
)
from .tiling import bounds_intersect_wedge, get_zone_wedge, list_kept_crops

logger = get_logger(__name__)

# Fixed tile size for this job-writer (the prediction pipeline supports any multiple
# of PATCH_SIZE).
TILE_SIZE = 32768

# When filtering tiles by GeoJSON, skip a UTM zone entirely if no feature's WGS84
# bounding box comes within this many degrees longitude of the zone. This avoids
# projecting shapes into distant UTM zones where the transform is unreliable.
GEOJSON_ZONE_LONGITUDE_MARGIN = 6.0


def enumerate_tiles_in_zone(utm_zone: CRS) -> Generator[tuple[int, int], None, None]:
    """List the (column, row) of all TILE_SIZE tiles within a UTM zone.

    Args:
        utm_zone: the CRS which must correspond to a UTM EPSG.

    Returns:
        generator of (column, row) of the tiles that are needed.
    """
    crs_bbox = STGeometry(
        Projection(utm_zone, 1, 1),
        shapely.box(*get_proj_bounds(utm_zone)),
        None,
    )
    projection = Projection(utm_zone, RESOLUTION, -RESOLUTION)
    pixel_bbox = crs_bbox.to_projection(projection)
    zone_bounds = tuple(int(value) for value in pixel_bbox.shp.bounds)

    for col in range(zone_bounds[0] // TILE_SIZE, zone_bounds[2] // TILE_SIZE + 1):
        for row in range(zone_bounds[1] // TILE_SIZE, zone_bounds[3] // TILE_SIZE + 1):
            yield (col, row)


def get_jobs(
    inputs: EmbeddingInputs,
    timestamp: datetime,
    out_path: str,
    completed_path: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    geojson_fname: str | None = None,
    count: int | None = None,
) -> list[list[str]]:
    """Get the prediction jobs (one per tile).

    Tiles whose completion markers already exist are excluded, along with tiles that
    don't intersect their zone's canonical wedge or contain no crops to process.

    Args:
        inputs: which input variant to use. Different variants produce different
            embeddings so they must use different out_path/completed_path.
        timestamp: the reference timestamp (start of the one-year input period). Must
            have timezone.
        out_path: the directory to write the embedding GeoTIFFs.
        completed_path: the directory for per-tile completion markers.
        epsg_code: limit tasks to this UTM zone (EPSG code); default all UTM zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        geojson_fname: limit tasks to tiles intersecting a feature in this GeoJSON
            file (features must be in WGS84 coordinates).
        count: limit to this many tasks (randomly sampled).

    Returns:
        a list of worker argument lists, one per TILE_SIZE tile.
    """
    if epsg_code:
        utm_zones = [CRS.from_epsg(epsg_code)]
    else:
        utm_zones = [CRS.from_epsg(code) for code in range(32601, 32661)]
        utm_zones += [CRS.from_epsg(code) for code in range(32701, 32761)]

    geojson_shapes: list[shapely.Geometry] | None = None
    if geojson_fname is not None:
        with UPath(geojson_fname).open() as f:
            feature_collection = json.load(f)
        geojson_shapes = [
            shapely.geometry.shape(feature["geometry"])
            for feature in feature_collection["features"]
        ]

    tasks: list[tuple[Projection, PixelBounds]] = []
    for utm_zone in tqdm.tqdm(utm_zones, desc="Enumerating tasks across UTM zones"):
        projection = Projection(utm_zone, RESOLUTION, -RESOLUTION)
        wedge = get_zone_wedge(utm_zone, RESOLUTION)

        # Project the GeoJSON shapes near this zone into the zone's pixel coordinate
        # system (skipping the zone if there are none nearby).
        zone_geojson_shapes: list[shapely.Geometry] | None = None
        if geojson_shapes is not None:
            zone_number = utm_zone.to_epsg() % 100
            zone_lon_min = -180 + (zone_number - 1) * 6
            zone_lon_max = zone_lon_min + 6
            zone_geojson_shapes = []
            for shp in geojson_shapes:
                shp_bounds = shp.bounds
                if shp_bounds[2] < zone_lon_min - GEOJSON_ZONE_LONGITUDE_MARGIN:
                    continue
                if shp_bounds[0] > zone_lon_max + GEOJSON_ZONE_LONGITUDE_MARGIN:
                    continue
                zone_geojson_shapes.append(
                    STGeometry(WGS84_PROJECTION, shp, None)
                    .to_projection(projection)
                    .shp
                )
            if len(zone_geojson_shapes) == 0:
                continue

        user_bounds_in_proj: PixelBounds | None = None
        if wgs84_bounds is not None:
            dst_geom = STGeometry(
                WGS84_PROJECTION, shapely.box(*wgs84_bounds), None
            ).to_projection(projection)
            user_bounds_in_proj = (
                int(dst_geom.shp.bounds[0]),
                int(dst_geom.shp.bounds[1]),
                int(dst_geom.shp.bounds[2]),
                int(dst_geom.shp.bounds[3]),
            )

        for col, row in enumerate_tiles_in_zone(utm_zone):
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

            tasks.append((projection, bounds))

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
            "--out_path",
            out_path,
            "--completed_path",
            completed_path,
        ]
        jobs.append(cur_args)

    return jobs


def write_jobs(
    inputs: EmbeddingInputs,
    timestamp: datetime,
    out_path: str,
    completed_path: str,
    queue_name: str,
    epsg_code: int | None = None,
    wgs84_bounds: tuple[float, float, float, float] | None = None,
    geojson_fname: str | None = None,
    count: int | None = None,
) -> None:
    """Enumerate tiles for one reference timestamp and write jobs to a Beaker queue.

    Args:
        inputs: which input variant to use. Different variants produce different
            embeddings so they must use different out_path/completed_path.
        timestamp: the reference timestamp (start of the one-year input period). Must
            have timezone.
        out_path: the directory to write the embedding GeoTIFFs.
        completed_path: the directory for per-tile completion markers.
        queue_name: the Beaker queue to write the job entries to.
        epsg_code: limit tasks to this UTM zone (EPSG code); default all UTM zones.
        wgs84_bounds: limit tasks to ones intersecting these WGS84 bounds.
        geojson_fname: limit tasks to tiles intersecting a feature in this GeoJSON
            file (features must be in WGS84 coordinates).
        count: limit to this many tasks (randomly sampled).
    """
    jobs = get_jobs(
        inputs=inputs,
        timestamp=timestamp,
        out_path=out_path,
        completed_path=completed_path,
        epsg_code=epsg_code,
        wgs84_bounds=wgs84_bounds,
        geojson_fname=geojson_fname,
        count=count,
    )
    # Shuffle so outputs start appearing from random parts of the world (aids
    # debugging).
    random.shuffle(jobs)
    rslp.common.worker.write_jobs(queue_name, "large_scale_embeddings", "predict", jobs)
