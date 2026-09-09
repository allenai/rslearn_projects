"""Create change finder windows restricted to a set of African countries.

This mirrors :mod:`rslp.change_finder.create_windows` but only keeps windows
whose center falls inside one of a fixed set of target countries. Random
lat/lon points are rejection-sampled within the bounding box of the target
countries and tested against their (unioned) polygons, so nearly every sampled
point results in an in-country candidate before the land/Sentinel-2 coverage
checks run.
"""

import json
import multiprocessing
import random
import urllib.request
from pathlib import Path

import numpy as np
import shapely
import tqdm
from rslearn.dataset import Dataset
from rslearn.utils.mp import star_imap_unordered
from shapely.geometry import shape
from upath import UPath

from .create_windows import (
    _CELL_RADIUS,
    BASE_MONTHS,
    WINDOW_SIZE,
    _compute_cell,
    _save_window,
)

# Natural Earth 1:50m admin-0 countries (used for the point-in-country test).
NE_COUNTRIES_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_50m_admin_0_countries.geojson"
)
_CACHE_PATH = (
    Path.home() / ".cache" / "change_finder" / "ne_50m_admin_0_countries.geojson"
)

# Target countries, keyed by their Natural Earth ``ADMIN`` name. Some names
# differ from their common name (e.g. Tanzania, Côte d'Ivoire), and a few
# (Guinea, Niger) must match exactly to exclude Guinea-Bissau / Nigeria.
TARGET_COUNTRIES = {
    "Democratic Republic of the Congo",
    "Angola",
    "Zambia",
    "United Republic of Tanzania",
    "Uganda",
    "Nigeria",
    "Ghana",
    "Benin",
    "Togo",
    "Cameroon",
    "Central African Republic",
    "Gabon",
    "Equatorial Guinea",
    "South Sudan",
    "Ethiopia",
    "Kenya",
    "Malawi",
    "Mozambique",
    "Chad",
    "Burkina Faso",
    "Guinea",
    "Ivory Coast",
    "Niger",
    "Sierra Leone",
    "Senegal",
}


def _load_country_geometry() -> (
    tuple[shapely.Geometry, tuple[float, float, float, float]]
):
    """Load and union the target country polygons.

    Returns:
        A tuple of (prepared union geometry, (minx, miny, maxx, maxy) bounds).
    """
    if not _CACHE_PATH.exists():
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(
            NE_COUNTRIES_URL, headers={"User-Agent": "change-finder"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            _CACHE_PATH.write_bytes(resp.read())

    with _CACHE_PATH.open() as f:
        gj = json.load(f)

    geoms = [
        shape(feat["geometry"])
        for feat in gj["features"]
        if feat["properties"].get("ADMIN") in TARGET_COUNTRIES
    ]
    found = {
        feat["properties"].get("ADMIN")
        for feat in gj["features"]
        if feat["properties"].get("ADMIN") in TARGET_COUNTRIES
    }
    missing = TARGET_COUNTRIES - found
    if missing:
        raise ValueError(
            f"Could not match all target countries; missing: {sorted(missing)}"
        )

    union = shapely.union_all(geoms)
    shapely.prepare(union)
    return union, union.bounds


def _sample_points_in_countries(
    rng: np.random.Generator,
    union: shapely.Geometry,
    bounds: tuple[float, float, float, float],
    num_samples: int,
) -> list[dict]:
    """Rejection-sample lat/lon points that fall inside the target countries."""
    minx, miny, maxx, maxy = bounds
    lats: list[float] = []
    lons: list[float] = []
    base_time_rng = random.Random()

    while len(lats) < num_samples:
        need = num_samples - len(lats)
        batch = max(need * 3, 1000)
        cand_lon = rng.uniform(minx, maxx, size=batch)
        cand_lat = rng.uniform(miny, maxy, size=batch)
        pts = shapely.points(cand_lon, cand_lat)
        inside = shapely.contains(union, pts)
        lons.extend(cand_lon[inside].tolist())
        lats.extend(cand_lat[inside].tolist())

    return [
        dict(lat=lat, lon=lon, base_time=base_time_rng.choice(BASE_MONTHS))
        for lat, lon in zip(lats[:num_samples], lons[:num_samples])
    ]


def create_windows_africa(
    ds_path: str,
    group: str = "africa",
    num_samples: int = 50000,
    workers: int = 32,
) -> None:
    """Create windows by sampling random locations within the target countries.

    Each window is a 128x128 pixel tile (10 m/pixel) in UTM. Locations are
    rejection-sampled within the bounding box of the target countries and kept
    only if they fall inside one of the country polygons; each point is then
    snapped to the nearest grid-aligned tile so that bounds are always multiples
    of WINDOW_SIZE. Points on ocean/water or lacking Sentinel-2 coverage are
    dropped, and windows too close to existing ones are skipped.

    Args:
        ds_path: output dataset path, e.g.
            gs://rslearn-eai/datasets/change_finder/dataset_v1/20260403/
        group: window group name.
        num_samples: number of in-country points to sample and process.
        workers: number of worker processes.
    """
    union, bounds = _load_country_geometry()

    ds_upath = UPath(ds_path)
    dataset = Dataset(ds_upath)
    rng = np.random.default_rng()

    # Build index of occupied grid cells from existing windows. Two cells are
    # "too close" if centers are < 256 px apart (i.e. < 128 px between borders),
    # which with WINDOW_SIZE=128 equals a 3×3 neighborhood in cell-index space.
    occupied: set[tuple[int, int, int]] = set()  # (epsg, cell_x, cell_y)
    print("Loading existing windows...")
    for w in dataset.load_windows(workers=workers, show_progress=True):
        epsg = w.projection.crs.to_epsg()
        if epsg is None:
            continue
        occupied.add((epsg, w.bounds[0] // WINDOW_SIZE, w.bounds[1] // WINDOW_SIZE))
    print(f"Found {len(occupied)} existing cells")

    # 1) Rejection-sample random lat/lon points inside the target countries.
    print(f"Sampling {num_samples} points within target countries...")
    samples = _sample_points_in_countries(rng, union, bounds, num_samples)

    # 2) Compute grid cells in parallel (reprojection + land/coverage checks).
    print(f"Computing grid cells for {len(samples)} samples with {workers} workers...")
    p = multiprocessing.Pool(workers)
    cell_results = list(
        tqdm.tqdm(
            star_imap_unordered(p, _compute_cell, samples),
            total=len(samples),
            desc="Computing cells",
        )
    )
    p.close()

    # 3) Sequentially filter by proximity to existing/planned windows.
    jobs: list[dict] = []
    skipped = 0
    for cell in cell_results:
        if cell is None:
            continue
        epsg, cell_x, cell_y, base_time = cell
        if any(
            (epsg, cell_x + dx, cell_y + dy) in occupied
            for dx in range(-_CELL_RADIUS + 1, _CELL_RADIUS)
            for dy in range(-_CELL_RADIUS + 1, _CELL_RADIUS)
        ):
            skipped += 1
            continue

        occupied.add((epsg, cell_x, cell_y))
        jobs.append(
            dict(
                dataset=dataset,
                group=group,
                epsg=epsg,
                cell_x=cell_x,
                cell_y=cell_y,
                base_time=base_time,
            )
        )

    # 4) Save windows in parallel (STAC checks + write).
    print(f"Skipped {skipped} samples too close to existing/planned windows")
    print(f"Processing {len(jobs)} samples with {workers} workers")
    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, _save_window, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs), desc="Creating windows"):
        pass
    p.close()
