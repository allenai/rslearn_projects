"""Build an LCC prediction dataset directly from an evaluation CSV.

This replaces the ``rslearn dataset add_windows`` step in the prediction flow for
point-based evaluation. Each CSV row becomes one prediction window centered on the
point, in that point's own UTM zone (which add_windows cannot do in a single call).

For each row, the reference "as of" time is the beginning of dst_year. The window
time_range is set to (T, T) with T = {dst_year}-01-01 minus 60 days, so the
config_predict.json layers derive:
- sentinel2_quarterly: [T-1800d, T]
- sentinel2_frequent_0: [T, T+60d] (the four 15-day periods, ending at dst_year start)

After running this, materialize and predict as usual:
    rslearn dataset prepare     --root <ds> --workers 32
    rslearn dataset materialize --root <ds> --workers 128
    rslearn model predict --config data/change_finder_v2/lcc_model/config_predict.yaml \
        --data.init_args.path=<ds>
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import shapely
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

PREDICTION_GROUP = "predict"
WINDOW_SIZE = 128
RESOLUTION = 10

# The frequent block covers [T, T+60d]; offsetting T back by 60d places the
# reference time (block end) at the beginning of dst_year.
REFERENCE_OFFSET = timedelta(days=60)

# Default dataset config, relative to the repo root (rslp/change_finder_v2/evaluation/).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_JSON = (
    _REPO_ROOT / "data" / "change_finder_v2" / "lcc_model" / "config_predict.json"
)


def _bounds_for_point(lon: float, lat: float) -> tuple[Any, tuple[int, int, int, int]]:
    """Return the UTM projection and a WINDOW_SIZE bounds centered on the point."""
    projection = get_utm_ups_projection(lon, lat, RESOLUTION, -RESOLUTION)
    center = STGeometry(
        WGS84_PROJECTION, shapely.Point(lon, lat), time_range=None
    ).to_projection(projection)
    center_col = math.floor(center.shp.x)
    center_row = math.floor(center.shp.y)
    half = WINDOW_SIZE // 2
    bounds = (
        center_col - half,
        center_row - half,
        center_col + half,
        center_row + half,
    )
    return projection, bounds


def create_dataset(
    csv_path: Path,
    ds_path: str,
) -> int:
    """Create prediction windows from the CSV. Returns the number of windows created."""
    ds_upath = UPath(ds_path)
    ds_upath.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEFAULT_CONFIG_JSON, ds_upath / "config.json")

    dataset = Dataset(ds_upath)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    for idx, row in enumerate(rows):
        lon = float(row["lon"])
        lat = float(row["lat"])
        dst_year = int(row["dst_year"])

        projection, bounds = _bounds_for_point(lon, lat)

        reference = datetime(dst_year, 1, 1, tzinfo=timezone.utc)
        t = reference - REFERENCE_OFFSET

        window = Window(
            storage=dataset.storage,
            group=PREDICTION_GROUP,
            name=f"eval_{idx:06d}",
            projection=projection,
            bounds=bounds,
            time_range=(t, t),
            options={
                "row_index": idx,
                "lon": lon,
                "lat": lat,
                "src_year": int(row["src_year"]),
                "dst_year": dst_year,
                "has_changed": row["has_changed"].strip().lower() == "true",
                "src_category": row.get("src_category", ""),
                "dst_category": row.get("dst_category", ""),
            },
        )
        window.save()

    print(f"Created {len(rows)} windows in {ds_upath} (group {PREDICTION_GROUP})")
    return len(rows)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Create an LCC prediction dataset from an evaluation CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Input evaluation CSV (from export_annotations_to_csv.py).",
    )
    parser.add_argument(
        "--ds-path",
        required=True,
        help="Output rslearn dataset path (config.json is written here).",
    )
    args = parser.parse_args()

    create_dataset(
        csv_path=args.csv,
        ds_path=args.ds_path,
    )


if __name__ == "__main__":
    main()
