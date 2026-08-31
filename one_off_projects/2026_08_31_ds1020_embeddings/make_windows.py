"""Create one 224x224 px rslearn window per ds1020 survey point.

Bulk equivalent of `rslearn dataset add_windows --utm --resolution 10
--window_size 224` (58,825 CLI invocations is not practical): per point we pick
the point's UTM zone, project the lat/lon, and center a fixed 224x224 pixel
window on it, mirroring add_windows_from_geometries's int-truncation centering.

Each window's time range is a single instant, the first day of the month BEFORE
the observation month; the dataset config's duration=90d / period_duration=30d
expands that into three 30-day mosaics covering (approximately) the previous
month, the event month and the following month -- the same mosaic recipe as
pretraining, just 3 periods instead of 12. For the points_fixed.csv (every date
is 2018-06-15) that start is 2018-05-01, i.e. May/June/July 2018.

Windows are assigned to shard groups {prefix}_{NN} of --shard_size rows each, in
CSV order, so prepare/materialize/predict can be split across Beaker jobs with
--group.

Window names are the survey_id, with whitespace and other unsafe characters
collapsed to "_" (17 ids in points_fixed.csv contain " - "); uniqueness is
asserted after sanitizing.

Usage (config.json must already be at the dataset root):
    python make_windows.py --csv ds1020_consolidated_survey_points_2017.csv \
        --root $DS --prefix y2017
    python make_windows.py --csv ds1020_consolidated_survey_points_fixed.csv \
        --root $DS --prefix fixed
"""

import argparse
import csv
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import tqdm
from pyproj import Transformer
from rasterio.crs import CRS
from upath import UPath

from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, get_utm_ups_crs

WGS84 = CRS.from_epsg(4326)


def month_before_start(date_str: str) -> datetime:
    """First day (UTC) of the month before the observation's month."""
    d = datetime.strptime(date_str, "%Y-%m-%d")
    year, month = (d.year - 1, 12) if d.month == 1 else (d.year, d.month - 1)
    return datetime(year, month, 1, tzinfo=timezone.utc)


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="survey-point CSV")
    parser.add_argument("--root", required=True, help="rslearn dataset root")
    parser.add_argument(
        "--prefix", required=True, help="group prefix, e.g. y2017 or fixed"
    )
    parser.add_argument("--window_size", type=int, default=224)
    parser.add_argument(
        "--shard_size",
        type=int,
        default=2000,
        help="windows per group; groups are the sharding unit for Beaker jobs",
    )
    parser.add_argument("--resolution", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N rows (smoke test)"
    )
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]

    names = [sanitize(r["survey_id"]) for r in rows]
    if len(set(names)) != len(names):
        seen: set[str] = set()
        dups = {n for n in names if n in seen or seen.add(n)}  # type: ignore[func-returns-value]
        raise SystemExit(f"{len(dups)} duplicate window names after sanitizing: {sorted(dups)[:10]}")

    dataset = Dataset(UPath(args.root))
    # One transformer per UTM zone (the points span only a couple of zones);
    # constructing a pyproj Transformer per point is what makes the naive
    # STGeometry route slow.
    transformers: dict[str, Transformer] = {}
    half = args.window_size // 2

    def build_window(i: int) -> Window:
        row = rows[i]
        lon, lat = float(row["longitude"]), float(row["latitude"])
        utm_crs = get_utm_ups_crs(lon, lat)
        key = utm_crs.to_string()
        if key not in transformers:
            transformers[key] = Transformer.from_crs(WGS84, utm_crs, always_xy=True)
        x, y = transformers[key].transform(lon, lat)
        projection = Projection(utm_crs, args.resolution, -args.resolution)
        # Same convention as add_windows_from_geometries: truncate the projected
        # pixel coordinate, then take +/- window_size//2.
        col, row_px = int(x / args.resolution), int(y / -args.resolution)
        bounds = (col - half, row_px - half, col + half, row_px + half)
        start = month_before_start(row["date"])
        return Window(
            storage=dataset.storage,
            group=f"{args.prefix}_{i // args.shard_size:02d}",
            name=names[i],
            projection=projection,
            bounds=bounds,
            time_range=(start, start),
            data_factory=dataset.window_data_storage_factory,
        )

    def make(i: int) -> None:
        build_window(i).save()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(tqdm.tqdm(pool.map(make, range(len(rows))), total=len(rows)))

    n_groups = (len(rows) + args.shard_size - 1) // args.shard_size
    print(
        f"created {len(rows)} windows in groups "
        f"{args.prefix}_00 .. {args.prefix}_{n_groups - 1:02d}"
    )


if __name__ == "__main__":
    main()
