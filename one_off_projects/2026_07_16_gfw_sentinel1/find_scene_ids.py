"""Find Sentinel-1 scene IDs intersecting ocean that we should run vessel detection on.

We enumerate every Sentinel-1 GRD IW DV scene from the requested time range (default:
April 2026) that intersects the ocean. The scene IDs are the full ``.SAFE`` names that
``rslp.sentinel1_vessels`` expects (it resolves them via the Copernicus/AWS data source
in ``setup_dataset_with_scene_ids``).

Approach (mirrors ``one_off_projects/2025_12_africa_vessels/get_scene_ids.py``):
- Tile the globe into a grid (default 5x5 degrees).
- Keep only tiles that touch ocean (using ``global_land_mask``), to avoid pointless API
  queries over purely-inland tiles.
- For each ocean tile, query the rslearn AWS Sentinel-1 data source with
  ``SpaceMode.INTERSECTS`` to list all scenes intersecting the tile in the time range.
- Drop scenes whose footprint does not actually touch ocean.
- De-duplicate scene IDs across tiles (a scene footprint spans multiple tiles).

The prediction pipeline requires every scene in a batch to share the same orbit
direction, so we query each orbit direction (ascending/descending) separately and write
a separate output file per orbit direction (the orbit direction is inserted into the
``--out_fname``/``--geojson_fname`` before the extension).

Credentials: the AWS Sentinel-1 data source uses the public Copernicus OData API for the
metadata search, but constructing it still reads ``COPERNICUS_USERNAME``/``COPERNICUS_PASSWORD``
(or ``COPERNICUS_ACCESS_TOKEN``) and needs AWS credentials for its boto3 client. Source
``rslearn/.env`` before running.

Note: the 0.7 confidence threshold requested for this run is a prediction-time parameter
(applied later by ``rslp.sentinel1_vessels``), not part of scene enumeration.

Example:

    python one_off_projects/2026_07_16_gfw_sentinel1/find_scene_ids.py \
        --start_time 2026-04-01 \
        --end_time 2026-05-01 \
        --out_fname scene_ids.json \
        --geojson_fname scene_geoms.geojson \
        --workers 16
"""

import argparse
import json
import multiprocessing
import time
from datetime import UTC, datetime

import numpy as np
import shapely
import tqdm
from global_land_mask import globe
from rslearn.config import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.aws_sentinel1 import Sentinel1
from rslearn.data_sources.copernicus import Sentinel1OrbitDirection
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

# Spacing (degrees) used to sample points when testing whether a polygon touches ocean.
OCEAN_SAMPLE_SPACING = 0.25

# Number of retries for a single tile's API query before giving up.
NUM_RETRIES = 5

# The prediction pipeline requires every scene in a batch to share the same orbit
# direction, so we enumerate each orbit direction separately and write separate files.
ORBIT_DIRECTIONS = [
    Sentinel1OrbitDirection.ASCENDING,
    Sentinel1OrbitDirection.DESCENDING,
]

# One AWS Sentinel-1 data source is constructed lazily per orbit direction per worker
# process (the orbit direction is applied as a server-side search filter).
_data_sources: dict[str, Sentinel1] = {}


def get_data_source(orbit_direction: Sentinel1OrbitDirection) -> Sentinel1:
    """Get a per-process cached AWS Sentinel-1 data source for an orbit direction."""
    if orbit_direction not in _data_sources:
        _data_sources[orbit_direction] = Sentinel1(orbit_direction=orbit_direction)
    return _data_sources[orbit_direction]


def polygon_touches_ocean(shp: shapely.Geometry) -> bool:
    """Return whether the given WGS84 polygon intersects any ocean.

    We sample a grid of points across the polygon's bounds (plus its exterior
    vertices) and return True if any point that falls inside the polygon is ocean.

    Args:
        shp: the polygon in WGS84 lon/lat coordinates.

    Returns:
        True if the polygon appears to touch ocean.
    """
    minx, miny, maxx, maxy = shp.bounds

    # Footprints that wrap the antimeridian produce a bogus full-width bounds; just keep
    # them (they are near the dateline in open sea and rare).
    if maxx - minx > 180:
        return True

    xs = np.arange(minx, maxx + OCEAN_SAMPLE_SPACING, OCEAN_SAMPLE_SPACING)
    ys = np.arange(miny, maxy + OCEAN_SAMPLE_SPACING, OCEAN_SAMPLE_SPACING)
    lon_grid, lat_grid = np.meshgrid(xs, ys)
    lons = lon_grid.ravel()
    lats = lat_grid.ravel()

    # Also include the exterior vertices so small footprints are not missed.
    if isinstance(shp, shapely.Polygon):
        coords = np.array(shp.exterior.coords)
        lons = np.concatenate([lons, coords[:, 0]])
        lats = np.concatenate([lats, coords[:, 1]])

    lons = np.clip(lons, -180, 180)
    lats = np.clip(lats, -90, 90)

    is_ocean = ~globe.is_land(lats, lons)
    if not is_ocean.any():
        return False

    ocean_points = shapely.points(lons[is_ocean], lats[is_ocean])
    return shapely.contains(shp, ocean_points).any() or shp.intersects(
        shapely.MultiPoint(ocean_points.tolist())
    )


def tile_touches_ocean(minx: int, miny: int, size: int) -> bool:
    """Return whether the given integer-degree tile touches ocean.

    Args:
        minx: minimum longitude of the tile.
        miny: minimum latitude of the tile.
        size: the tile size in degrees.

    Returns:
        True if any sampled point in the tile is ocean.
    """
    xs = np.arange(minx, minx + size + 1)
    ys = np.arange(miny, miny + size + 1)
    lon_grid, lat_grid = np.meshgrid(xs, ys)
    is_ocean = ~globe.is_land(lat_grid.ravel(), lon_grid.ravel())
    return bool(is_ocean.any())


def get_scenes(
    tile_bounds: tuple[int, int, int, int],
    orbit_direction: Sentinel1OrbitDirection,
    start_time: datetime,
    end_time: datetime,
    filter_scene_ocean: bool,
) -> list[tuple[str, str]]:
    """Get Sentinel-1 scenes intersecting the tile in the time range.

    Args:
        tile_bounds: (minx, miny, maxx, maxy) of the tile in WGS84 degrees.
        orbit_direction: only return scenes with this orbit direction.
        start_time: start of the search time range.
        end_time: end of the search time range.
        filter_scene_ocean: whether to drop scenes whose footprint does not touch ocean.

    Returns:
        list of (scene_id, footprint_wkt) tuples.
    """
    minx, miny, maxx, maxy = tile_bounds
    geom = STGeometry(
        WGS84_PROJECTION,
        shapely.box(minx, miny, maxx, maxy),
        (start_time, end_time),
    )
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=100000)

    data_source = get_data_source(orbit_direction)
    last_exc: Exception | None = None
    for attempt in range(NUM_RETRIES):
        try:
            groups = data_source.get_items([geom], query_config)[0]
            break
        except Exception as e:  # noqa: BLE001 -- transient API errors, retry with backoff
            print(f"got error {e} for geometry {geom}, retrying...")
            last_exc = e
            time.sleep(2**attempt)
    else:
        raise RuntimeError(
            f"failed to query tile {tile_bounds} after {NUM_RETRIES} attempts"
        ) from last_exc

    results: list[tuple[str, str]] = []
    for group in groups:
        if len(group.items) != 1:
            raise ValueError(
                "expected each item group to have one item with INTERSECTS space mode"
            )
        item = group.items[0]
        shp = item.geometry.shp
        if filter_scene_ocean and not polygon_touches_ocean(shp):
            continue
        results.append((item.name, shp.wkt))
    return results


def build_tiles(
    bbox: tuple[int, int, int, int], grid_size: int
) -> list[tuple[int, int, int, int]]:
    """Build the list of ocean-touching tiles covering the bbox.

    Args:
        bbox: (minx, miny, maxx, maxy) region to cover, in integer degrees.
        grid_size: tile size in degrees.

    Returns:
        list of (minx, miny, maxx, maxy) tiles that touch ocean.
    """
    tiles: list[tuple[int, int, int, int]] = []
    for lon in range(bbox[0], bbox[2], grid_size):
        for lat in range(bbox[1], bbox[3], grid_size):
            if not tile_touches_ocean(lon, lat, grid_size):
                continue
            tiles.append((lon, lat, lon + grid_size, lat + grid_size))
    return tiles


def add_orbit_suffix(fname: str, orbit_direction: Sentinel1OrbitDirection) -> str:
    """Insert the orbit direction into a filename before its extension.

    For example ``scene_ids.json`` becomes ``scene_ids_ascending.json``.

    Args:
        fname: the base filename (may include a directory and/or URI scheme).
        orbit_direction: the orbit direction to encode.

    Returns:
        the filename with the orbit direction inserted before the extension.
    """
    suffix = orbit_direction.value.lower()
    dot = fname.rfind(".")
    slash = max(fname.rfind("/"), fname.rfind("\\"))
    if dot > slash:
        return f"{fname[:dot]}_{suffix}{fname[dot:]}"
    return f"{fname}_{suffix}"


def main() -> None:
    """Enumerate ocean Sentinel-1 scene IDs and write them out."""
    parser = argparse.ArgumentParser(
        description="Find Sentinel-1 scene IDs intersecting ocean for a time range.",
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default="2026-04-01",
        help="Start of the search time range (YYYY-MM-DD), inclusive. Default 2026-04-01.",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        default="2026-05-01",
        help="End of the search time range (YYYY-MM-DD), exclusive. Default 2026-05-01.",
    )
    parser.add_argument(
        "--bbox",
        type=str,
        default="-180,-85,180,85",
        help="Region to search as minx,miny,maxx,maxy integer degrees. Default whole globe.",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=5,
        help="Tile size in degrees used to query the data source. Default 5.",
    )
    parser.add_argument(
        "--out_fname",
        type=str,
        required=True,
        help="Base filename to write the JSON list of scene IDs. The orbit direction is "
        "inserted before the extension, producing one file per orbit direction "
        "(e.g. scene_ids_ascending.json and scene_ids_descending.json).",
    )
    parser.add_argument(
        "--geojson_fname",
        type=str,
        default=None,
        help="Optional base filename to write scene footprints as GeoJSON. The orbit "
        "direction is inserted before the extension, one file per orbit direction.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel worker processes. Default 16.",
    )
    parser.add_argument(
        "--no_filter_scene_ocean",
        action="store_true",
        help="Keep scenes even if their footprint does not touch ocean.",
    )
    args = parser.parse_args()

    start_time = datetime.strptime(args.start_time, "%Y-%m-%d").replace(tzinfo=UTC)
    end_time = datetime.strptime(args.end_time, "%Y-%m-%d").replace(tzinfo=UTC)
    bbox = tuple(int(v) for v in args.bbox.split(","))
    assert len(bbox) == 4
    filter_scene_ocean = not args.no_filter_scene_ocean

    tiles = build_tiles(bbox, args.grid_size)
    print(f"Got {len(tiles)} ocean tiles to query")

    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)

    # Query each orbit direction separately and write a separate output file for each,
    # since the prediction pipeline requires a batch to share one orbit direction.
    p = multiprocessing.Pool(args.workers)
    try:
        for orbit_direction in ORBIT_DIRECTIONS:
            scene_ids: set[str] = set()
            features: list[Feature] = []

            jobs = [
                dict(
                    tile_bounds=tile,
                    orbit_direction=orbit_direction,
                    start_time=start_time,
                    end_time=end_time,
                    filter_scene_ocean=filter_scene_ocean,
                )
                for tile in tiles
            ]
            outputs = star_imap_unordered(p, get_scenes, jobs)
            for scene_list in tqdm.tqdm(
                outputs, total=len(jobs), desc=orbit_direction.value
            ):
                for name, wkt in scene_list:
                    if name in scene_ids:
                        continue
                    scene_ids.add(name)
                    features.append(
                        Feature(
                            STGeometry(WGS84_PROJECTION, shapely.from_wkt(wkt), None),
                            {"scene_id": name},
                        )
                    )

            out_fname = add_orbit_suffix(args.out_fname, orbit_direction)
            print(
                f"{orbit_direction.value}: got {len(scene_ids)} scene IDs after "
                f"de-duplication, writing to {out_fname}"
            )
            with UPath(out_fname).open("w") as f:
                json.dump(sorted(scene_ids), f)

            if args.geojson_fname:
                geojson_fname = add_orbit_suffix(args.geojson_fname, orbit_direction)
                vector_format.encode_to_file(UPath(geojson_fname), features)
    finally:
        p.close()
        p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
