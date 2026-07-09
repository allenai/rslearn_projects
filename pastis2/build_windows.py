"""Phase 3: create stratified 128x128 @10m rslearn windows over all territories.

Reads the normalized RPG GPKGs (download_rpg.py) and, per territory, samples cropland
tiles stratified by département x crop-class rarity, guaranteeing a minimum quota per
overseas territory so islands are never sampled to zero. Each sampled cell is tiled
into 128x128 windows via rslearn's add_windows_from_geometries with use_utm=True (rslearn
assigns each window its correct UTM zone -- so metropole, Corsica and every DROM are
handled without per-territory CRS juggling here).

PASTIS geometry: 128x128 px @ 10 m = 1.28 km tiles; time_range = the S2 growing season
(Sep <year-1> .. Nov <year>). Windows land under <dataset>/windows/<group>/.

This is a scaffold: the density-grid + stratification specifics are marked TODO, and it
requires the rslearn env + an existing dataset root with config.json (Phase 4/5). It is
written against the verified rslearn API but not yet run end-to-end.

Run:  python build_windows.py --dataset /path/to/rslearn_ds --year 2019 \
          --per-territory-min 50 --target-total 20000
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from upath import UPath

from rasterio.crs import CRS
from rslearn.dataset.add_windows import add_windows_from_geometries
from rslearn.dataset.dataset import Dataset
from rslearn.utils import Projection, STGeometry
from territories import TERRITORIES

HERE = Path(__file__).parent
RPG = Path(os.environ.get("PASTIS2_RPG_DIR", HERE / "data" / "rpg"))

PATCH_PX = 128          # PASTIS patch size
RESOLUTION = 10.0       # m/px (Sentinel-2 baseline)
BIN_M = PATCH_PX * RESOLUTION   # 1280 m = one window footprint (the sampling cell)
RARITY_POWER = 1.0      # cell weight = (1/class_freq)**RARITY_POWER; >0 over-samples rare


def growing_season(year: int, fetch_months: int = 12) -> tuple[datetime, datetime]:
    """Acquisition window ending Aug <year>, spanning ``fetch_months`` months back.

    The eval targets the 12-month season Sep <year-1> .. Aug <year>. Set fetch_months
    to 18 or 24 to OVER-fetch: S2 (period_duration=30d) then yields up to that many
    monthly composites, and make_tensors reduces them to 12 calendar-month slots,
    back-filling any cloudy target-year month from an adjacent year. This is how we end
    up with a full 12 months despite cloud dropouts.
    """
    end = datetime(year, 8, 31, tzinfo=timezone.utc)
    # Go back fetch_months (approx via 30-day months is fine; rslearn clips by scene date).
    start_total = (year * 12 + 8) - fetch_months  # month index of Aug<year> minus span
    start = datetime(start_total // 12, start_total % 12 + 1, 1, tzinfo=timezone.utc)
    return (start, end)


def sample_cells(gdf: gpd.GeoDataFrame, n: int, seed: int = 0) -> list[Point]:
    """Stratified sample of ~window-sized cell centers, over-weighting rare crops.

    Bins positive-class parcels into ``BIN_M`` (1.28 km) cells — one window each — scores
    each cell by the rarity of its rarest class (inverse global class frequency), and draws
    ``n`` cells without replacement with probability proportional to that score. Rare crops
    (sorghum, soybean, tropical, ...) get represented instead of being drowned out by the
    dominant meadow/cereal classes, while sampling many distinct cells still gives broad
    geographic spread. Returns cell-center Points in the territory CRS (meters).

    (Replaces the earlier random-parcel placeholder. Département-level stratification could
    be layered on later via an admin join; rarity + spatial spread covers the main need.)
    """
    pos = gdf[gdf["class_id"] > 0]
    if len(pos) == 0:
        return []
    b = pos.geometry.bounds
    cx = (((b["minx"] + b["maxx"]) / 2) // BIN_M).astype("int64").to_numpy()
    cy = (((b["miny"] + b["maxy"]) / 2) // BIN_M).astype("int64").to_numpy()
    freq = pos["class_id"].map(pos["class_id"].value_counts(normalize=True)).to_numpy()
    rarity = freq ** (-RARITY_POWER)
    cells = (
        pd.DataFrame({"cx": cx, "cy": cy, "rarity": rarity})
        .groupby(["cx", "cy"])["rarity"].max().reset_index()
    )
    n = min(n, len(cells))
    p = cells["rarity"].to_numpy()
    p = p / p.sum()
    idx = np.random.default_rng(seed).choice(len(cells), size=n, replace=False, p=p)
    sel = cells.iloc[idx]
    return [Point((cx_ + 0.5) * BIN_M, (cy_ + 0.5) * BIN_M)
            for cx_, cy_ in zip(sel["cx"], sel["cy"])]


def build_for_territory(
    ds: Dataset, key: str, n_cells: int, year: int, fetch_months: int
) -> int:
    gpkg = RPG / f"{key}.gpkg"
    if not gpkg.exists():
        print(f"[{key}] missing {gpkg}; run download_rpg.py first — skipping")
        return 0
    gdf = gpd.read_file(gpkg)
    epsg = gdf.crs.to_epsg()
    cells = sample_cells(gdf, n_cells)
    crs = CRS.from_epsg(epsg)
    # Input geometries are in CRS meters at resolution (1, 1) -- matching rslearn's own
    # add_windows_from_file idiom (Projection(crs, 1, 1)); the y-res MUST be +1 here or
    # the northing sign flips and the window is mislocated. The OUTPUT projection carries
    # the real 10 m resolution (CRS overridden per-window by use_utm).
    src_proj = Projection(crs, 1, 1)
    out_proj = Projection(crs, RESOLUTION, -RESOLUTION)
    geoms = [STGeometry(src_proj, c, None) for c in cells]
    windows = add_windows_from_geometries(
        dataset=ds,
        group=f"rpg_{year}",
        geometries=geoms,
        projection=out_proj,
        window_size=PATCH_PX,             # one 128px window centered on each sampled cell
        time_range=growing_season(year, fetch_months),
        use_utm=True,                     # per-window UTM -> covers metropole + all DROM
    )
    print(f"[{key}] epsg={epsg} cells={len(cells)} -> {len(windows)} windows")
    return len(windows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="rslearn dataset root (has config.json)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--target-total", type=int, default=20000,
                    help="approx total windows across metropole")
    ap.add_argument("--per-territory-min", type=int, default=50,
                    help="minimum sampled cells per DROM so islands are covered")
    ap.add_argument("--fetch-months", type=int, default=12,
                    help="acquisition-window span in months (18/24 to over-fetch so 12 "
                         "calendar slots fill despite cloud dropouts). Also set the S2 "
                         "layer's query_config.max_matches >= this in config.json.")
    args = ap.parse_args()

    ds = Dataset(UPath(args.dataset))
    total = 0
    for t in TERRITORIES:
        # Metropole gets the bulk; each DROM gets at least the min quota.
        n_cells = args.target_total if t.key == "metropole" else args.per_territory_min
        total += build_for_territory(ds, t.key, n_cells, args.year, args.fetch_months)
    print(f"created ~{total} windows across {len(TERRITORIES)} territories")


if __name__ == "__main__":
    main()
