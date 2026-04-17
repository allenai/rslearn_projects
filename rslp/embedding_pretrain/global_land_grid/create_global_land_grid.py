"""Create a global adaptive grid at 2 km resolution, filtered to land areas.

The globe is divided into processing tiles (default 10 x 10 degrees) to cap
peak memory.  Within each tile the grid is built with an adaptive-grid
logic (step size in *meters* so cell ground area stays roughly constant across
latitudes).  Only cells that intersect a user-supplied land-area polygon are kept.
The output stores cell centroids (points) rather than full polygons to
keep the file compact.

Usage
-----
    python create_global_land_grid.py \
        --land-geojson /path/to/land_polygons.geojson \
        --output /path/to/global_land_grid.gdb \
        --step 2000 \
        --tile-size 10
"""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Geod
from shapely.geometry import Polygon, box
from shapely.prepared import prep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TILE_SIZE_DEG = 10.0
LAYER_NAME = "global_land_grid"


def adaptive_grid_polygons(
    minx: float,
    maxx: float,
    miny: float,
    maxy: float,
    stepx: int,
    stepy: int,
) -> list[Polygon]:
    """Return adaptive grid cells as raw Shapely polygons.

    Reimplements the logic of
    ``data_preproc_script.preprocess.utils.adaptive_grid`` but returns a plain
    list instead of a GeoDataFrame so the caller can filter before allocating
    the frame.
    """
    geod = Geod(ellps="WGS84")

    # Clamp latitude to avoid the geodesic singularity at the poles where
    # "east" is undefined and geod.fwd returns degenerate longitudes.
    # One cell width (~0.02° ≈ 2.2 km) avoids thin slivers at the pole edge.
    _POLE_EPS = 0.02
    miny = max(miny, -90.0 + _POLE_EPS)
    maxy = min(maxy, 90.0 - _POLE_EPS)

    lat_grid: list[float] = [miny]
    while lat_grid[-1] < maxy:
        _, lat, _ = geod.fwd(minx, lat_grid[-1], 0, stepy)
        lat_grid.append(min(lat, maxy))
        if lat >= maxy:
            break

    lon_grid: list[list[float]] = []
    for lat in lat_grid:
        lon_row: list[float] = [minx]
        while lon_row[-1] < maxx:
            lon, _, _ = geod.fwd(lon_row[-1], lat, 90, stepx)
            if lon < lon_row[-1]:  # wrapped past antimeridian
                lon_row.append(maxx)
                break
            lon_row.append(min(lon, maxx))
            if lon >= maxx:
                break
        lon_grid.append(lon_row)

    polygons: list[Polygon] = []
    for i in range(len(lat_grid) - 1):
        for j in range(len(lon_grid[i]) - 1):
            polygons.append(
                Polygon(
                    [
                        (lon_grid[i][j], lat_grid[i]),
                        (lon_grid[i][j + 1], lat_grid[i]),
                        (lon_grid[i][j + 1], lat_grid[i + 1]),
                        (lon_grid[i][j], lat_grid[i + 1]),
                    ]
                )
            )
    return polygons


def create_global_land_grid(
    land_geojson_path: str | Path,
    output_path: str | Path,
    step_m: int = 2000,
    tile_size_deg: float = TILE_SIZE_DEG,
) -> None:
    """Generate a global adaptive grid filtered to land and write to GDB.

    Args:
        land_geojson_path: GeoJSON with a (Multi)Polygon of land areas.
        output_path: Destination File Geodatabase (.gdb) path.
        step_m: Cell size in meters (default 2000 for 2 km).
        tile_size_deg: Processing-tile width/height in degrees (default 10).
    """
    output_path = Path(output_path)
    if output_path.suffix != ".gdb":
        output_path = output_path.with_suffix(".gdb")

    logger.info("Loading land geometry from %s", land_geojson_path)
    land_gdf = gpd.read_file(land_geojson_path)
    land_union = land_gdf.union_all()
    land_prepared = prep(land_union)
    logger.info("Land geometry loaded and indexed")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(output_path)

    lat_starts = np.arange(-90.0, 90.0, tile_size_deg)
    lon_starts = np.arange(-180.0, 180.0, tile_size_deg)

    total_cells = 0
    first_write = True
    t0 = time.time()

    for row_idx, lat_start in enumerate(lat_starts):
        lat_end = min(lat_start + tile_size_deg, 90.0)
        row_cells = 0

        for lon_start in lon_starts:
            lon_end = min(lon_start + tile_size_deg, 180.0)

            tile_bbox = box(lon_start, lat_start, lon_end, lat_end)
            if not land_prepared.intersects(tile_bbox):
                continue

            polys = adaptive_grid_polygons(
                lon_start, lon_end, lat_start, lat_end, step_m, step_m
            )

            land_polys = [p for p in polys if land_prepared.intersects(p)]
            if not land_polys:
                continue

            centroids = [p.centroid for p in land_polys]
            gdf = gpd.GeoDataFrame(geometry=centroids, crs="EPSG:4326")
            gdf["id"] = range(total_cells + 1, total_cells + 1 + len(gdf))

            mode = "w" if first_write else "a"
            gdf.to_file(output_path, layer=LAYER_NAME, mode=mode)
            first_write = False

            row_cells += len(land_polys)
            total_cells += len(land_polys)

        elapsed = time.time() - t0
        logger.info(
            "lat %+6.1f to %+6.1f  |  row %2d/%d  |  "
            "+%d cells (total %d)  |  %.0fs elapsed",
            lat_start,
            lat_end,
            row_idx + 1,
            len(lat_starts),
            row_cells,
            total_cells,
            elapsed,
        )

    logger.info(
        "Done: %d land cells written to %s  (%.0fs)",
        total_cells,
        output_path,
        time.time() - t0,
    )


def main() -> None:
    """Parse CLI arguments and create the global land grid."""
    parser = argparse.ArgumentParser(
        description="Create a global land grid at a given resolution.",
    )
    parser.add_argument(
        "--land-geojson",
        required=True,
        help="Path to a GeoJSON file containing the land-area (Multi)Polygon.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the File Geodatabase (.gdb).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2000,
        help="Grid cell size in meters (default: 2000).",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=TILE_SIZE_DEG,
        help="Processing tile size in degrees (default: 10).",
    )
    args = parser.parse_args()

    create_global_land_grid(
        land_geojson_path=args.land_geojson,
        output_path=args.output,
        step_m=args.step,
        tile_size_deg=args.tile_size,
    )


if __name__ == "__main__":
    main()
