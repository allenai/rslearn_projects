"""Build a WorldCover prediction dataset from an evaluation CSV.

The WorldCover model is a single-time land-cover segmentation model, so to evaluate
it on change we predict land cover at two times per point. Each CSV row therefore
becomes TWO prediction windows, both in the point's own UTM zone (which
``rslearn dataset add_windows`` cannot do in a single call):

- ``eval_{idx:06d}_src`` with ``time_range = ({src_year}-01-01, {src_year}-12-31)``
- ``eval_{idx:06d}_dst`` with ``time_range = ({dst_year}-01-01, {dst_year}-12-31)``

The full-calendar-year ranges mirror how the WorldCover training windows were built
(see ``rslp/worldcover/create_windows.py``); the ``sentinel2`` layer then yields ~12
monthly mosaics per window.

The script copies ``config_predict.json`` into the dataset as ``config.json``, so you
can immediately run the standard prepare/materialize/predict steps:

    EVAL_DS=/path/to/worldcover_eval_ds/
    python -m rslp.change_finder_v2.evaluation.worldcover.create_prediction_dataset_from_csv \
        --csv eval.csv --ds-path "$EVAL_DS"

    rslearn dataset prepare     --root "$EVAL_DS" --workers 32
    rslearn dataset materialize --root "$EVAL_DS" --workers 128
    rslearn model predict \
        --config data/land_cover_change/worldcover_change/config.yaml \
        --ckpt_path <ckpt> \
        --data.init_args.path="$EVAL_DS"

Imagery uses the OlmoEarth Datasets source, so the API env vars must be set (see the
main change_finder_v2 README's Prerequisites).
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import shapely
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

PREDICTION_GROUP = "predict"
WINDOW_SIZE = 64
RESOLUTION = 10

# Default dataset config, relative to the repo root.
# (rslp/change_finder_v2/evaluation/worldcover/ -> repo root is 4 parents up.)
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_JSON = (
    _REPO_ROOT
    / "data"
    / "land_cover_change"
    / "worldcover_change"
    / "config_predict.json"
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


def _year_range(year: int) -> tuple[datetime, datetime]:
    """Full calendar-year time range for a given year (mirrors training windows)."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 12, 31, tzinfo=timezone.utc)
    return start, end


def create_dataset(csv_path: Path, ds_path: str) -> int:
    """Create src/dst prediction windows from the CSV. Returns window count."""
    ds_upath = UPath(ds_path)
    ds_upath.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(DEFAULT_CONFIG_JSON, ds_upath / "config.json")

    dataset = Dataset(ds_upath)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    num_windows = 0
    for idx, row in enumerate(rows):
        lon = float(row["lon"])
        lat = float(row["lat"])
        src_year = int(row["src_year"])
        dst_year = int(row["dst_year"])

        projection, bounds = _bounds_for_point(lon, lat)

        common_options = {
            "row_index": idx,
            "lon": lon,
            "lat": lat,
            "src_year": src_year,
            "dst_year": dst_year,
            "has_changed": row["has_changed"].strip().lower() == "true",
            "src_category": row.get("src_category", ""),
            "dst_category": row.get("dst_category", ""),
        }

        for kind, year in (("src", src_year), ("dst", dst_year)):
            window = Window(
                storage=dataset.storage,
                group=PREDICTION_GROUP,
                name=f"eval_{idx:06d}_{kind}",
                projection=projection,
                bounds=bounds,
                time_range=_year_range(year),
                options={**common_options, "kind": kind, "year": year},
            )
            window.save()
            num_windows += 1

    print(
        f"Created {num_windows} windows ({len(rows)} points x 2) in {ds_upath} "
        f"(group {PREDICTION_GROUP})"
    )
    return num_windows


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Create a WorldCover prediction dataset from an evaluation CSV."
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

    create_dataset(csv_path=args.csv, ds_path=args.ds_path)


if __name__ == "__main__":
    main()
