"""Create rslearn windows for the Baleares points using the collaborators' splits.

Uses the OFFICIAL island splits from the collaborators (not our longitude estimate).

Input: baleares_sample_points_with_island.geojson -- 40,000 WGS84 points, each with
properties:
  * ``mode``: 0 = non-seagrass (negative), 1 = dense seagrass (positive)
              (NOTE: this differs from the earlier file, which used 2 = dense.)
  * ``island``: Formentera | Ibiza | Mallorca | Menorca
  * ``island_group``: Pitiusas (=Formentera+Ibiza) | Mallorca | Menorca

Each point becomes one point-centered window (single center-pixel ``label_raster``,
rest = 255 ignore), mirroring create_baleares_windows.py, but with two changes:
  1. The island_group is taken directly from the file (authoritative), NOT estimated
     from longitude. Our earlier longitude guess was substantially wrong.
  2. ``mode`` is mapped to the dataset's canonical raster label so the existing
     class_id_mapping {0:0, 2:1, 255:255} keeps working:
         mode 0 (non-seagrass) -> raster label 0 (background)
         mode 1 (dense)         -> raster label 2 (dense_seagrass)

Per-fold tags ``fold_<group> in {train,val,test}`` are written for the three
leave-one-island-group-out folds (mallorca, menorca, pitiusas): the held-out group's
points are ``test``, the other two groups' points are ``train``/``val`` (hash split).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import tqdm
from rasterio.crs import CRS
from rslearn.config.dataset import StorageConfig
from rslearn.dataset import Window
from rslearn.utils import Projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

DEFAULT_GEOJSON = Path(
    "/weka/dfive-default/piperw/scripts/seagrass/baleares_sample_points_with_island.geojson"
)
DEFAULT_DATASET_PATH = Path("/weka/dfive-default/piperw/rslearn_projects/data/seagrass")

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label_raster"
LABEL_BAND = "label"
# canonical dataset raster labels (config.json class_names index)
CANON = {0: "background", 1: "sparse_seagrass", 2: "dense_seagrass"}
# input mode -> canonical raster label
MODE_TO_LABEL = {0: 0, 1: 2}
# the three leave-one-island-group-out folds (lowercased island_group)
FOLDS = ["mallorca", "menorca", "pitiusas"]


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """UTM EPSG code for a lon/lat point."""
    import math

    zone = max(1, min(60, int(math.floor((lon + 180.0) / 6.0)) + 1))
    return (32600 if lat >= 0 else 32700) + zone


_TRANSFORMERS: dict[int, Any] = {}


def _transformer(epsg: int) -> Any:
    """Cache pyproj transformers from WGS84 to a UTM EPSG."""
    if epsg not in _TRANSFORMERS:
        _TRANSFORMERS[epsg] = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    return _TRANSFORMERS[epsg]


def calculate_bounds(cx: int, cy: int, size: int) -> tuple[int, int, int, int]:
    """Pixel bounds for a window of `size` centered on (cx, cy) (even size)."""
    return (cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2)


def split_role(sample_id: str, seed: str, val_fraction: float) -> str:
    """Stable train/val assignment (for the non-held-out groups of a fold)."""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    return "val" if (int(digest[:8], 16) / 0xFFFFFFFF) < val_fraction else "train"


def create_window(**job: Any) -> str:
    """Create one point-centered window with a center-pixel label and fold tags."""
    lon, lat = job["lon"], job["lat"]
    mode = job["mode"]
    if mode not in MODE_TO_LABEL:
        raise ValueError(f"Unexpected mode {mode}")
    label_value = MODE_TO_LABEL[mode]
    sample_id = job["sample_id"]
    size = job["window_size"]
    group_lc = job["island_group"].lower()
    if group_lc not in FOLDS:
        raise ValueError(f"Unexpected island_group {job['island_group']}")

    epsg = utm_epsg_for_lonlat(lon, lat)
    x_utm, y_utm = _transformer(epsg).transform(lon, lat)
    projection = Projection(CRS.from_epsg(epsg), WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    cx = int(round(x_utm / WINDOW_RESOLUTION))
    cy = int(round(y_utm / -WINDOW_RESOLUTION))
    bounds = calculate_bounds(cx, cy, size)

    # per-fold roles: the held-out group's points are test, else train/val by hash.
    fold_tags = {}
    for held in FOLDS:
        if group_lc == held:
            fold_tags[f"fold_{held}"] = "test"
        else:
            fold_tags[f"fold_{held}"] = split_role(
                sample_id, job["split_seed"], job["val_fraction"]
            )

    options = {
        "sample_id": sample_id,
        "mode": mode,
        "label": label_value,
        "label_name": CANON[label_value],
        "longitude": lon,
        "latitude": lat,
        "island": job["island"],
        "island_group": job["island_group"],
        "split_seed": job["split_seed"],
        **fold_tags,
    }

    window = Window(
        storage=StorageConfig()
        .instantiate_window_storage_factory()
        .get_storage(job["ds_path"]),
        group=job["group"],
        name=f"sample_{sample_id}",
        projection=projection,
        bounds=bounds,
        time_range=(
            datetime(job["year"], 1, 1, tzinfo=timezone.utc),
            datetime(job["year"], 12, 31, tzinfo=timezone.utc),
        ),
        options=options,
    )
    window.save()

    raster = np.full((1, size, size), 255, dtype=np.uint8)
    raster[0, size // 2, size // 2] = label_value
    raster_dir = window.get_raster_dir(LABEL_LAYER, [LABEL_BAND])
    GeotiffRasterFormat().encode_raster(
        raster_dir, window.projection, window.bounds, RasterArray(chw_array=raster)
    )
    window.mark_layer_completed(LABEL_LAYER)
    return job["island_group"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geojson", default=str(DEFAULT_GEOJSON))
    p.add_argument("--ds_path", default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--group", default="baleares_official_2025")
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--split_seed", default="baleares_official_2025")
    p.add_argument("--workers", type=int, default=32)
    return p.parse_args()


def main() -> None:
    """Create all official-split Baleares point windows."""
    args = parse_args()
    ds_path = UPath(args.ds_path)
    gj = json.load(Path(args.geojson).open())
    jobs = []
    for idx, ft in enumerate(gj["features"]):
        lon, lat = ft["geometry"]["coordinates"]
        props = ft["properties"]
        jobs.append(
            dict(
                sample_id=f"{idx:06d}",
                lon=float(lon),
                lat=float(lat),
                mode=int(props["mode"]),
                island=props["island"],
                island_group=props["island_group"],
                ds_path=ds_path,
                group=args.group,
                window_size=args.window_size,
                year=args.year,
                val_fraction=args.val_fraction,
                split_seed=args.split_seed,
            )
        )
    print(f"creating {len(jobs)} windows in {ds_path}/windows/{args.group} ...")
    from collections import Counter

    with multiprocessing.Pool(args.workers) as pool:
        groups = list(
            tqdm.tqdm(star_imap_unordered(pool, create_window, jobs), total=len(jobs))
        )
    print("island_group counts:", dict(Counter(groups)))


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
