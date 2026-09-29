"""Select forest loss events from GLAD-S2 alerts for labeling.

This reuses the event extraction from the olmoearth_projects forest loss driver
pipeline (olmoearth_projects.projects.forest_loss_driver.extract_alerts), but slices the
time range into calendar months rather than fixed-length day slices.

There are two stages:

1. extract: for each GLAD tile, read the rasters once and, for each calendar month,
   extract connected components of alert pixels dated in that month, restricted to the
   target countries, and randomly sample up to --max_per_tile_month of them. Each tile's
   candidates are written to {out_dir}/candidates/{tile}.geojson.
2. select: pool the candidates, shuffle them, and greedily accept events as long as the
   (country, month) has fewer than --max_per_country_month accepted events and there is
   no already-accepted event (in any month) whose center point is within
   --min_distance_m meters. The output is {out_dir}/selected_events.geojson in the same
   format as extract_alerts output, plus {out_dir}/counts.csv.

Example:

    python select_events.py extract --out_dir /path/to/out --workers 7
    python select_events.py select --out_dir /path/to/out
"""

import argparse
import csv
import dataclasses
import logging
import math
import multiprocessing
import os
import random
import resource
import time
import zlib
from collections import Counter
from datetime import UTC, datetime

# extract_events_for_window is called per block, so disable its per-call progress
# bars. This must be set before tqdm is imported.
os.environ["TQDM_DISABLE"] = "1"

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.feature import Feature
from rslearn.utils.fsspec import open_rasterio_upath_reader
from rslearn.utils.geometry import STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import get_raster_projection_and_bounds
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

from olmoearth_projects.projects.forest_loss_driver.extract_alerts import (
    BASE_DATETIME,
    ExtractAlertsArgs,
    extract_events_for_window,
    load_country_polygons,
)

# Likewise, silence the per-call debug logging about skipped shapes.
logging.getLogger(
    "olmoearth_projects.projects.forest_loss_driver.extract_alerts"
).setLevel(logging.INFO)

# Same tiles and countries as olmoearth_run_data/forest_loss_driver/deploy.yaml.
TILES = [
    "040W_10S_030W_00N.tif",
    "040W_20S_030W_10S.tif",
    "050W_00N_040W_10N.tif",
    "050W_10S_040W_00N.tif",
    "050W_20S_040W_10S.tif",
    "060W_00N_050W_10N.tif",
    "060W_10S_050W_00N.tif",
    "060W_20S_050W_10S.tif",
    "070W_00N_060W_10N.tif",
    "070W_10S_060W_00N.tif",
    "070W_20S_060W_10S.tif",
    "080W_00N_070W_10N.tif",
    "080W_10S_070W_00N.tif",
    "080W_20S_070W_10S.tif",
]
COUNTRIES = ["PE", "BR", "CO", "EC", "BO"]
COUNTRY_DATA_PATH = "/weka/dfive-default/rslearn-eai/artifacts/natural_earth_countries/20240830/ne_10m_admin_0_countries.shp"

EARTH_RADIUS_M = 6371008.8

VECTOR_FORMAT = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)


def get_months(start: str, end: str) -> list[tuple[str, datetime, datetime]]:
    """Get the calendar months in [start, end), each as (YYYY-MM, start, end).

    Args:
        start: the first month, as YYYY-MM.
        end: the exclusive end month, as YYYY-MM.
    """
    months = []
    cur = datetime.strptime(start, "%Y-%m").replace(tzinfo=UTC)
    end_dt = datetime.strptime(end, "%Y-%m").replace(tzinfo=UTC)
    while cur < end_dt:
        if cur.month == 12:
            nxt = cur.replace(year=cur.year + 1, month=1)
        else:
            nxt = cur.replace(month=cur.month + 1)
        months.append((cur.strftime("%Y-%m"), cur, nxt))
        cur = nxt
    return months


def extract_tile(
    args: ExtractAlertsArgs,
    tif_fname: str,
    months: list[tuple[str, datetime, datetime]],
    country_wgs84_shps: dict[str, shapely.Geometry],
    out_fname: UPath,
    block_size: int,
    block_margin: int,
) -> tuple[str, int, float, float]:
    """Extract candidate events from one GLAD tile for each month.

    This mirrors extract_events_for_tile but reads the rasters once and then, for each
    calendar month, calls extract_events_for_window on overlapping blocks of the tile.
    Processing the 100k x 100k tile in one call allocates ~80 GB of temporary arrays
    per month, which with several workers causes heavy kernel memory reclaim.

    Each block is extended by block_margin pixels on each side, and we keep only the
    events whose center pixel falls in the block core, so events are not duplicated
    and connected components smaller than block_margin are extracted exactly as they
    would be on the whole tile.

    Returns:
        (tif_fname, number of events, elapsed seconds, peak RSS in GB)
    """
    start_time = time.time()
    # Seed per tile so the sampling is reproducible regardless of worker scheduling.
    random.seed(zlib.crc32(tif_fname.encode()))
    # We sample per tile-month after merging the blocks.
    block_args = dataclasses.replace(args, max_number_of_events=None)

    with open_rasterio_upath_reader(UPath(args.conf_prefix) / tif_fname) as src:
        conf_data = src.read(1)
        projection, bounds = get_raster_projection_and_bounds(src)
    with open_rasterio_upath_reader(UPath(args.date_prefix) / tif_fname) as src:
        date_data = src.read(1)
    print(f"{tif_fname}: read rasters in {time.time() - start_time:.0f} s", flush=True)

    height, width = date_data.shape
    events: list[Feature] = []
    for month_str, month_start, month_end in months:
        events_for_month: list[Feature] = []
        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                row0 = max(row - block_margin, 0)
                col0 = max(col - block_margin, 0)
                row1 = min(row + block_size + block_margin, height)
                col1 = min(col + block_size + block_margin, width)
                block_bounds = (
                    bounds[0] + col0,
                    bounds[1] + row0,
                    bounds[0] + col1,
                    bounds[1] + row1,
                )
                block_events = extract_events_for_window(
                    args=block_args,
                    tif_fname=tif_fname,
                    conf_data=conf_data[row0:row1, col0:col1],
                    date_data=date_data[row0:row1, col0:col1],
                    projection=projection,
                    bounds=block_bounds,
                    country_wgs84_shps=country_wgs84_shps,
                    min_days=(month_start - BASE_DATETIME).days,
                    max_days=(month_end - BASE_DATETIME).days,
                )
                for feat in block_events:
                    # center_pixel is relative to the block, make it relative to the
                    # tile like in extract_events_for_tile.
                    center_col = feat.properties["center_pixel"][0] + col0
                    center_row = feat.properties["center_pixel"][1] + row0
                    if not (
                        col <= center_col < col + block_size
                        and row <= center_row < row + block_size
                    ):
                        continue
                    feat.properties["center_pixel"] = (center_col, center_row)
                    events_for_month.append(feat)

        num_before_sampling = len(events_for_month)
        if len(events_for_month) > args.max_number_of_events:
            events_for_month = random.sample(
                events_for_month, args.max_number_of_events
            )

        for feat in events_for_month:
            # Record the center point in WGS84 for the distance filter.
            col, row = feat.properties["center_pixel"]
            center = STGeometry(
                projection, shapely.Point(col + bounds[0], row + bounds[1]), None
            ).to_projection(WGS84_PROJECTION)
            feat.properties["center_lon"] = center.shp.x
            feat.properties["center_lat"] = center.shp.y
            feat.properties["month"] = month_str
        print(
            f"{tif_fname} {month_str}: {len(events_for_month)} events "
            f"(of {num_before_sampling}, {time.time() - start_time:.0f} s elapsed)",
            flush=True,
        )
        events.extend(events_for_month)

    # Write to a temporary file first so a partially written file is never mistaken
    # for a completed tile.
    tmp_fname = out_fname.parent / (out_fname.name + ".tmp")
    VECTOR_FORMAT.encode_to_file(tmp_fname, events)
    tmp_fname.rename(out_fname)
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    return tif_fname, len(events), time.time() - start_time, peak_rss_gb


def run_extract(cli_args: argparse.Namespace) -> None:
    """Run the extract stage over all tiles."""
    out_dir = UPath(cli_args.out_dir)
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    months = get_months(cli_args.start_month, cli_args.end_month)
    tiles = cli_args.tiles or TILES
    args = ExtractAlertsArgs(
        gcs_tiff_filenames=tiles,
        out_fname=str(out_dir / "unused.geojson"),
        country_data_path=COUNTRY_DATA_PATH,
        countries=COUNTRIES,
        max_number_of_events=cli_args.max_per_tile_month,
    )
    country_wgs84_shps = load_country_polygons(
        UPath(args.country_data_path), COUNTRIES
    )

    jobs = []
    for tif_fname in tiles:
        out_fname = cand_dir / tif_fname.replace(".tif", ".geojson")
        if out_fname.exists():
            print(f"skipping {tif_fname} since {out_fname} exists")
            continue
        jobs.append(
            dict(
                args=args,
                tif_fname=tif_fname,
                months=months,
                country_wgs84_shps=country_wgs84_shps,
                out_fname=out_fname,
                block_size=cli_args.block_size,
                block_margin=cli_args.block_margin,
            )
        )

    p = multiprocessing.Pool(cli_args.workers)
    for tif_fname, num_events, elapsed, peak_rss_gb in star_imap_unordered(
        p, extract_tile, jobs
    ):
        print(
            f"finished {tif_fname}: {num_events} events in {elapsed / 60:.1f} min "
            f"(peak RSS {peak_rss_gb:.1f} GB)",
            flush=True,
        )
    p.close()


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters between two WGS84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def run_select(cli_args: argparse.Namespace) -> None:
    """Run the select stage: shuffle, then apply the distance and count limits."""
    out_dir = UPath(cli_args.out_dir)
    cand_fnames = sorted((out_dir / "candidates").glob("*.geojson"))
    candidates: list[Feature] = []
    for fname in tqdm.tqdm(cand_fnames, desc="loading candidates"):
        candidates.extend(VECTOR_FORMAT.decode_from_file(fname))
    print(f"loaded {len(candidates)} candidates from {len(cand_fnames)} tiles")

    # Sort before shuffling so the result does not depend on file loading order.
    candidates.sort(
        key=lambda f: (
            f.properties["tif_fname"],
            f.properties["month"],
            tuple(f.properties["center_pixel"]),
        )
    )
    random.Random(cli_args.seed).shuffle(candidates)

    # Grid index over accepted center points. The cell size (in degrees) must span at
    # least min_distance_m in both axes so that we only need to check the 3x3
    # neighborhood. Longitude degrees shrink by cos(lat), so compute the cell size
    # from the maximum absolute latitude of the candidates.
    max_abs_lat = max(abs(f.properties["center_lat"]) for f in candidates)
    m_per_deg = math.pi * EARTH_RADIUS_M / 180
    cell_deg = cli_args.min_distance_m / (m_per_deg * math.cos(math.radians(max_abs_lat)))
    cell_deg *= 1.01
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}

    counts: Counter[tuple[str, str]] = Counter()
    cand_counts: Counter[tuple[str, str]] = Counter()
    distance_rejects: Counter[tuple[str, str]] = Counter()
    selected: list[Feature] = []
    for feat in tqdm.tqdm(candidates, desc="selecting"):
        key = (feat.properties["country"], feat.properties["month"])
        cand_counts[key] += 1
        if counts[key] >= cli_args.max_per_country_month:
            continue
        lon = feat.properties["center_lon"]
        lat = feat.properties["center_lat"]
        cell = (math.floor(lon / cell_deg), math.floor(lat / cell_deg))
        too_close = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_lon, other_lat in grid.get((cell[0] + dx, cell[1] + dy), []):
                    if haversine_m(lon, lat, other_lon, other_lat) < cli_args.min_distance_m:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break
        if too_close:
            distance_rejects[key] += 1
            continue
        grid.setdefault(cell, []).append((lon, lat))
        counts[key] += 1
        selected.append(feat)

    print(f"selected {len(selected)} events")
    VECTOR_FORMAT.encode_to_file(out_dir / "selected_events.geojson", selected)

    months = sorted({month for _, month in cand_counts})
    with (out_dir / "counts.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["country", "month", "candidates", "distance_rejected", "selected"]
        )
        for country in COUNTRIES:
            for month in months:
                key = (country, month)
                writer.writerow(
                    [country, month, cand_counts[key], distance_rejects[key], counts[key]]
                )
                if counts[key] < cli_args.max_per_country_month:
                    print(
                        f"shortfall {country} {month}: selected {counts[key]} "
                        f"of {cand_counts[key]} candidates"
                    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="stage", required=True)

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--out_dir", required=True)
    extract_parser.add_argument("--start_month", default="2025-01")
    extract_parser.add_argument(
        "--end_month", default="2026-08", help="exclusive end month"
    )
    extract_parser.add_argument("--max_per_tile_month", type=int, default=10000)
    extract_parser.add_argument("--workers", type=int, default=7)
    extract_parser.add_argument("--block_size", type=int, default=10000)
    extract_parser.add_argument("--block_margin", type=int, default=500)
    extract_parser.add_argument(
        "--tiles", nargs="*", default=None, help="override the tile list (for testing)"
    )

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--out_dir", required=True)
    select_parser.add_argument("--max_per_country_month", type=int, default=1000)
    select_parser.add_argument("--min_distance_m", type=float, default=250)
    select_parser.add_argument("--seed", type=int, default=0)

    cli_args = parser.parse_args()
    if cli_args.stage == "extract":
        run_extract(cli_args)
    else:
        run_select(cli_args)
