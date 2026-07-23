"""Build an embeddings prediction dataset from evaluation (and training) CSVs.

Each CSV row becomes TWO windows (one per year) so we can embed the point at src_year
and dst_year and score change from the two embeddings. Windows are created in the
point's own UTM zone with a full-calendar-year time_range (mirroring the WorldCover
flow).

- ``--csv`` (eval points) -> group ``predict``, window names ``eval_{idx:06d}_{kind}``.
- ``--train-csv`` (optional, labeled training points) -> group ``train``, window names
  ``train_{idx:06d}_{kind}``. Produce this CSV by running export_annotations_to_csv.py on
  the training v2 annotation JSONs (the same ones that feed lcc_model.prepare). The
  labeled training points get embedded in the same dataset so the linear probe can fit
  on them.

``--source`` selects which dataset config to copy in:
- ``alphaearth``: the embeddings layer is the GoogleSatelliteEmbeddingV1 data source, so
  just prepare/materialize (no model).
- ``olmoearth``: materialize Sentinel-2, then ``rslearn model predict`` with
  olmoearth_model.yaml writes the embeddings layer.

    EMBED_DS=/path/to/embeddings_eval_ds/
    python -m rslp.change_finder_v2.evaluation.embeddings.create_prediction_dataset_from_csv \
        --source alphaearth --csv eval.csv --train-csv train.csv --ds-path "$EMBED_DS"
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
TRAIN_GROUP = "train"
WINDOW_SIZE = 64
RESOLUTION = 10

# Dataset configs live alongside this module's data directory.
# (rslp/change_finder_v2/evaluation/embeddings/ -> repo root is 4 parents up.)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIG_DIR = _REPO_ROOT / "data" / "change_finder_v2" / "evaluation" / "embeddings"
CONFIG_BY_SOURCE = {
    "alphaearth": _CONFIG_DIR / "alphaearth_config.json",
    "olmoearth": _CONFIG_DIR / "olmoearth_config.json",
}


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
    """Full calendar-year time range for a given year."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year, 12, 31, tzinfo=timezone.utc)
    return start, end


def _add_windows(dataset: Dataset, csv_path: Path, group: str, name_prefix: str) -> int:
    """Create src/dst windows for every row of one CSV. Returns window count."""
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
                group=group,
                name=f"{name_prefix}{idx:06d}_{kind}",
                projection=projection,
                bounds=bounds,
                time_range=_year_range(year),
                options={**common_options, "kind": kind, "year": year},
            )
            window.save()
            num_windows += 1

    print(f"Created {num_windows} windows ({len(rows)} points x 2) in group {group}")
    return num_windows


def create_dataset(
    source: str,
    csv_path: Path,
    ds_path: str,
    train_csv_path: Path | None = None,
) -> None:
    """Create the embeddings prediction dataset from the eval (and train) CSVs."""
    ds_upath = UPath(ds_path)
    ds_upath.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CONFIG_BY_SOURCE[source], ds_upath / "config.json")

    dataset = Dataset(ds_upath)
    _add_windows(dataset, csv_path, PREDICTION_GROUP, "eval_")
    if train_csv_path is not None:
        _add_windows(dataset, train_csv_path, TRAIN_GROUP, "train_")

    print(f"\nDataset config ({source}) written to {ds_upath / 'config.json'}")
    if source == "alphaearth":
        print(
            "Next:\n"
            f"  rslearn dataset prepare     --root {ds_path} --workers 32\n"
            f"  rslearn dataset materialize --root {ds_path} --workers 128\n"
            "  python -m rslp.change_finder_v2.evaluation.embeddings.predict_change ...\n"
            "  python -m rslp.change_finder_v2.evaluation.embeddings.linear_probe ..."
        )
    else:
        print(
            "Next (set OEDATASETS_API_URL / DATASETS_API_TOKEN):\n"
            f"  rslearn dataset prepare     --root {ds_path} --workers 32\n"
            f"  rslearn dataset materialize --root {ds_path} --workers 128\n"
            f"  EMBED_DS={ds_path} rslearn model predict \\\n"
            "      --config data/change_finder_v2/evaluation/embeddings/olmoearth_model.yaml\n"
            "  python -m rslp.change_finder_v2.evaluation.embeddings.predict_change ...\n"
            "  python -m rslp.change_finder_v2.evaluation.embeddings.linear_probe ..."
        )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Create an embeddings prediction dataset from an evaluation CSV."
    )
    parser.add_argument(
        "--source",
        required=True,
        choices=sorted(CONFIG_BY_SOURCE),
        help="Embedding source (selects the dataset config to copy in).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Evaluation CSV (from export_annotations_to_csv.py) -> group 'predict'.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help=(
            "Optional labeled training CSV (export_annotations_to_csv.py on the training "
            "v2 JSONs) -> group 'train', used to fit the linear probe."
        ),
    )
    parser.add_argument(
        "--ds-path",
        required=True,
        help="Output rslearn dataset path (config.json is written here).",
    )
    args = parser.parse_args()

    create_dataset(
        source=args.source,
        csv_path=args.csv,
        ds_path=args.ds_path,
        train_csv_path=args.train_csv,
    )


if __name__ == "__main__":
    main()
