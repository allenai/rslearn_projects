"""Generate rslearn windows centred on global land-grid centroids.

Reads the centroid GDB produced by ``create_global_land_grid.py`` and creates
one rslearn ``Window`` per selected cell.  A stride / stride-index scheme lets
multiple jobs split the grid without overlap.

Each window is assigned a random year sampled uniformly from 2016-2025
(deterministic given the seed) with a Jan 1 – Dec 31 time range.

Multiple machines can safely run this script against the same output
directory.  Input indices are shuffled upfront so each machine (with a
different random seed) naturally covers different parts of the grid.
Windows are written iteratively in batches — already-existing windows
are skipped before building, with a second guard just before writing.

Usage
-----
    python globe_window_generation.py \
        --grid-gdb /path/to/global_land_grid.gdb \
        --output /path/to/rslearn_dataset \
        --window-size 128 \
        --stride 4 --stride-index 0
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from rslearn.dataset import Window
from rslearn.dataset.storage.file import FileWindowStorage
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LAYER_NAME = "global_land_grid"
RESOLUTION = 10  # meters per pixel
WINDOW_SIZE = 128  # pixels
YEAR_MIN = 2016
YEAR_MAX = 2025
IO_WORKERS = 32
BATCH_SIZE = 5000


def _window_exists(ds_path: UPath, group: str, name: str) -> bool:
    """Return True if the window's metadata.json already exists on disk."""
    return (Window.get_window_root(ds_path, group, name) / "metadata.json").exists()


def _make_window(
    row_geometry_x: float,
    row_geometry_y: float,
    cell_id: int,
    year: int,
    storage: FileWindowStorage,
    group: str,
    half_m: float,
    resolution: int,
    transformer_cache: dict,
) -> Window:
    """Build a single Window object from centroid coordinates."""
    lon, lat = row_geometry_x, row_geometry_y

    utm_proj = get_utm_ups_projection(lon, lat, resolution, -resolution)
    crs_key = str(utm_proj.crs)
    if crs_key not in transformer_cache:
        transformer_cache[crs_key] = (
            utm_proj,
            Transformer.from_crs("EPSG:4326", utm_proj.crs, always_xy=True),
        )
    _, transformer = transformer_cache[crs_key]

    cx, cy = transformer.transform(lon, lat)

    min_x = cx - half_m
    min_y = cy - half_m
    max_x = cx + half_m
    max_y = cy + half_m

    # y_resolution is negative so dividing max_y gives the smaller pixel row
    bounds = (
        int(min_x / utm_proj.x_resolution),
        int(max_y / utm_proj.y_resolution),
        int(max_x / utm_proj.x_resolution),
        int(min_y / utm_proj.y_resolution),
    )

    time_range = (
        datetime(year, 1, 1, tzinfo=timezone.utc),
        datetime(year, 12, 31, tzinfo=timezone.utc),
    )

    return Window(
        storage=storage,
        group=group,
        name=f"globe_{cell_id}",
        projection=utm_proj,
        bounds=bounds,
        time_range=time_range,
        options={"cell_id": cell_id, "year": year},
    )


def generate_windows(
    grid_gdb_path: str,
    output_path: str,
    group: str = "default",
    window_size: int = WINDOW_SIZE,
    resolution: int = RESOLUTION,
    stride: int = 1,
    stride_index: int = 0,
    seed: int | None = None,
    layer: str = LAYER_NAME,
    io_workers: int = IO_WORKERS,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Create rslearn windows from global land-grid centroids.

    Safe to run concurrently from multiple machines against the same
    *output_path*.  Input indices are shuffled upfront (based on seed)
    so each machine covers different parts of the grid first.  Windows
    are built and written iteratively in batches — already-existing
    windows are skipped before building, with a race-condition guard
    just before writing.

    Args:
        grid_gdb_path: Path to the .gdb produced by create_global_land_grid.py.
        output_path: Root path for the rslearn dataset (FileWindowStorage).
        group: Window group name.
        window_size: Window width and height in pixels.
        resolution: Meters per pixel.
        stride: Write one window every *stride* cells.
        stride_index: Which index within the stride to process (0-based).
        seed: Random seed for year sampling.  When *None* a random seed is
            chosen and logged, so that concurrent machines naturally diverge.
        layer: Layer name inside the GDB to read.
        io_workers: Number of threads for parallel window saving.
        batch_size: Number of windows to process per batch.
    """
    logger.info("Reading grid from %s (layer=%s)", grid_gdb_path, layer)
    gdf = gpd.read_file(grid_gdb_path, layer=layer)
    logger.info("Loaded %d centroids", len(gdf))

    if stride_index >= stride:
        raise ValueError(f"stride-index ({stride_index}) must be < stride ({stride})")

    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**63))
        logger.info("No seed provided; using random seed: %d", seed)
    else:
        logger.info("Using seed: %d", seed)

    ds_path = UPath(output_path)
    storage = FileWindowStorage(ds_path)
    rng = np.random.default_rng(seed)

    half_m = window_size * resolution / 2.0
    t0 = time.time()

    # -- Collect eligible row indices and shuffle them ---------------------
    eligible_indices = np.arange(stride_index, len(gdf), stride, dtype=np.int64)
    total = len(eligible_indices)
    logger.info(
        "%d eligible cells (stride=%d, stride_index=%d)", total, stride, stride_index
    )

    rng.shuffle(eligible_indices)
    logger.info("Shuffled input indices for parallel-safe processing")

    # Pre-draw years for all eligible cells (one per index, order matches
    # the shuffled eligible_indices so year assignment is seed-dependent).
    years = rng.integers(YEAR_MIN, YEAR_MAX + 1, size=total).astype(int)

    # Extract geometry and id arrays for fast positional access.
    geom_x = gdf.geometry.x.values
    geom_y = gdf.geometry.y.values
    cell_ids = gdf["id"].values

    # -- Iterate in batches: skip existing, build, write -------------------
    transformer_cache: dict = {}
    saved = 0
    skipped = 0

    def _save_if_new(w: Window) -> bool:
        if _window_exists(ds_path, w.group, w.name):
            return False
        w.save()
        return True

    for batch_start in range(0, total, batch_size):
        batch_indices = eligible_indices[batch_start : batch_start + batch_size]
        batch_years = years[batch_start : batch_start + batch_size]

        # Build windows for this batch, skipping already-existing ones.
        batch_windows: list[Window] = []
        for grid_idx, year in zip(batch_indices, batch_years):
            cell_id = int(cell_ids[grid_idx])
            name = f"globe_{cell_id}"
            if _window_exists(ds_path, group, name):
                skipped += 1
                continue
            batch_windows.append(
                _make_window(
                    row_geometry_x=float(geom_x[grid_idx]),
                    row_geometry_y=float(geom_y[grid_idx]),
                    cell_id=cell_id,
                    year=int(year),
                    storage=storage,
                    group=group,
                    half_m=half_m,
                    resolution=resolution,
                    transformer_cache=transformer_cache,
                )
            )

        # Write the batch in parallel.
        batch_saved = 0
        batch_existed = 0
        if batch_windows:
            with ThreadPoolExecutor(max_workers=io_workers) as pool:
                futures = {pool.submit(_save_if_new, w): w for w in batch_windows}
                for future in as_completed(futures):
                    if future.result():
                        batch_saved += 1
                    else:
                        batch_existed += 1

        saved += batch_saved
        skipped += batch_existed
        remaining = total - (batch_start + len(batch_indices))
        logger.info(
            "Batch %d–%d done: %d saved, %d skipped (existed), "
            "%d remaining, %.0fs elapsed",
            batch_start,
            batch_start + len(batch_indices) - 1,
            saved,
            skipped,
            remaining,
            time.time() - t0,
        )

    logger.info(
        "Done: %d windows saved, %d skipped (already existed), " "%d total in %.0fs",
        saved,
        skipped,
        total,
        time.time() - t0,
    )


def main() -> None:
    """Parse CLI arguments and generate rslearn windows from grid centroids."""
    parser = argparse.ArgumentParser(
        description="Generate rslearn windows from global land-grid centroids.",
    )
    parser.add_argument(
        "--grid-gdb",
        required=True,
        help="Path to the .gdb produced by create_global_land_grid.py.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Root path for the rslearn dataset.",
    )
    parser.add_argument(
        "--group",
        default="default",
        help="Window group name (default: 'default').",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="Window width/height in pixels (default: 128).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=RESOLUTION,
        help="Meters per pixel (default: 10).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process 1 window every N cells (default: 1).",
    )
    parser.add_argument(
        "--stride-index",
        type=int,
        default=0,
        help="Which index within the stride to process, 0-based (default: 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for year sampling (default: random).",
    )
    parser.add_argument(
        "--layer",
        default=LAYER_NAME,
        help="Layer name to read from the GDB (default: 'global_land_grid').",
    )
    parser.add_argument(
        "--io-workers",
        type=int,
        default=IO_WORKERS,
        help=f"Number of threads for parallel window saving (default: {IO_WORKERS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Windows per batch (progress is logged after each batch, default: {BATCH_SIZE}).",
    )
    args = parser.parse_args()

    generate_windows(
        grid_gdb_path=args.grid_gdb,
        output_path=args.output,
        group=args.group,
        window_size=args.window_size,
        resolution=args.resolution,
        stride=args.stride,
        stride_index=args.stride_index,
        seed=args.seed,
        layer=args.layer,
        io_workers=args.io_workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
