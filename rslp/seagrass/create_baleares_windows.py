"""Create rslearn windows for the Baleares seagrass point dataset.

Input: a GeoJSON of binary-labeled points (property ``mode``: 0=no_seagrass,
2=dense_seagrass) in WGS84. Each point becomes one point-centered window (single
center-pixel ``label_raster``, rest = 255 ignore), mirroring
create_windows.py:create_sample_window for the Jamaica points.

Each window is assigned an island group (Mallorca / Menorca / Pitiusas) by
longitude (the three groups are separated by clear sea-channel gaps), and gets
per-fold tags ``fold_<island> in {train,val,test}`` so a leave-one-island-group-out
config can select 2 islands to train on and 1 to evaluate on via tags:

    train_config: tags {fold_menorca: train}   # Mallorca + Pitiusas
    val_config:   tags {fold_menorca: val}
    test_config:  tags {fold_menorca: test}     # Menorca only
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
    "/weka/dfive-default/piperw/scripts/seagrass/baleares_samples_binary_points.geojson"
)
DEFAULT_DATASET_PATH = Path("/weka/dfive-default/piperw/rslearn_projects/data/seagrass")

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label_raster"
LABEL_BAND = "label"
LABEL_NAMES = {0: "background", 1: "sparse_seagrass", 2: "dense_seagrass"}
ISLANDS = ["mallorca", "menorca", "pitiusas"]
# Longitude thresholds separating the island groups (clear gaps in the data).
PITIUSAS_MAX_LON = 1.95
MALLORCA_MAX_LON = 3.55


def assign_island(lon: float) -> str:
    """Assign an island group from longitude."""
    if lon < PITIUSAS_MAX_LON:
        return "pitiusas"
    if lon < MALLORCA_MAX_LON:
        return "mallorca"
    return "menorca"


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
    """Stable train/val assignment (for the non-held-out islands of a fold)."""
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest()
    return "val" if (int(digest[:8], 16) / 0xFFFFFFFF) < val_fraction else "train"


def create_window(**job: Any) -> str:
    """Create one point-centered window with a center-pixel label and fold tags."""
    lon, lat = job["lon"], job["lat"]
    label_value = job["mode"]
    if label_value not in LABEL_NAMES:
        raise ValueError(f"Unexpected label {label_value}")
    sample_id = job["sample_id"]
    size = job["window_size"]

    epsg = utm_epsg_for_lonlat(lon, lat)
    x_utm, y_utm = _transformer(epsg).transform(lon, lat)
    projection = Projection(CRS.from_epsg(epsg), WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    cx = int(round(x_utm / WINDOW_RESOLUTION))
    cy = int(round(y_utm / -WINDOW_RESOLUTION))
    bounds = calculate_bounds(cx, cy, size)

    island = assign_island(lon)
    # per-fold roles: test on the held-out island, else train/val from a hash.
    fold_tags = {}
    for held in ISLANDS:
        if island == held:
            fold_tags[f"fold_{held}"] = "test"
        else:
            fold_tags[f"fold_{held}"] = split_role(
                sample_id, job["split_seed"], job["val_fraction"]
            )

    options = {
        "sample_id": sample_id,
        "mode": label_value,
        "label": label_value,
        "label_name": LABEL_NAMES[label_value],
        "longitude": lon,
        "latitude": lat,
        "island": island,
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
    return island


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geojson", default=str(DEFAULT_GEOJSON))
    p.add_argument("--ds_path", default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--group", default="baleares_2025")
    p.add_argument("--window_size", type=int, default=64)
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--split_seed", default="baleares_2025")
    p.add_argument("--workers", type=int, default=32)
    return p.parse_args()


def main() -> None:
    """Create all Baleares point windows."""
    args = parse_args()
    ds_path = UPath(args.ds_path)
    gj = json.load(Path(args.geojson).open())
    jobs = []
    for idx, ft in enumerate(gj["features"]):
        lon, lat = ft["geometry"]["coordinates"]
        jobs.append(
            dict(
                sample_id=f"{idx:06d}",
                lon=float(lon),
                lat=float(lat),
                mode=int(ft["properties"]["mode"]),
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
        islands = list(
            tqdm.tqdm(star_imap_unordered(pool, create_window, jobs), total=len(jobs))
        )
    print("island counts:", dict(Counter(islands)))


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
