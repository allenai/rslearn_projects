"""Create rslearn EVAL windows tiling each held-out island group's polygon footprint.

This builds the ground-truth raster for the collaborators' polygon-based LOIO
evaluation: run inference over the entire held-out island and compute pixel-F1 within
the survey footprint (threshold 0.5; dense = positive, sparse = positive, non = negative).

Input: baleares_ground_truth.shp -- 12 polygons (4 islands x 3 classes: dense/sparse/non
seagrass), EPSG:4326. Island groups: Mallorca, Menorca, Pitiusas (=Ibiza+Formentera).

For each island group we:
  1. Union all 3 class polygons -> the survey FOOTPRINT (the evaluable area).
  2. Tile the footprint's bbox on a fixed UTM (EPSG:32631) grid of TILE px @ 10 m,
     keeping only tiles that intersect the footprint.
  3. Rasterize the GT into each tile's label_raster:
        outside footprint         -> 255 (ignore, not scored)
        non seagrass              -> 0   (negative)
        dense OR sparse seagrass  -> 2   (positive; canonical dense label)
     (label 2 keeps the dataset class_id_mapping {0:0, 2:1, 255:255} valid, matching
      the training windows.)

Each window is tagged with its island_group so a fold's model (trained holding out that
group) evaluates exactly its held-out tiles: test_config tags {eval_group: mallorca}.
"""

from __future__ import annotations

import argparse
import multiprocessing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fiona
import numpy as np
import pyproj
import rasterio.features
import tqdm
from affine import Affine
from rasterio.crs import CRS
from rslearn.config.dataset import StorageConfig
from rslearn.dataset import Window
from rslearn.utils import Projection
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from shapely.geometry import box, mapping, shape
from shapely.ops import transform, unary_union
from upath import UPath

DEFAULT_SHP = Path(
    "/weka/dfive-default/piperw/scripts/seagrass/baleares_ground_truth.shp"
)
DEFAULT_DATASET_PATH = Path("/weka/dfive-default/piperw/rslearn_projects/data/seagrass")

UTM = 32631  # UTM 31N covers all of the Baleares (1.2-4.35 E)
RES = 10
LABEL_LAYER = "label_raster"
LABEL_BAND = "label"
GROUPS = {
    "mallorca": ["Mallorca"],
    "menorca": ["Menorca"],
    "pitiusas": ["Ibiza", "Formentera"],
}
# seagrass_c -> canonical raster label; sparse counts as positive (=dense label 2) in eval
CLASS_LABEL = {"non seagrass": 0, "sparse seagrass": 2, "dense seagrass": 2}


def load_group_geoms(shp: Path) -> dict[str, dict[str, Any]]:
    """Load polygons grouped by island_group, reprojected to UTM.

    Returns {group_lc: {"footprint": geom, "burn": [(geom, label), ...]}}.
    """
    to_utm = pyproj.Transformer.from_crs(4326, UTM, always_xy=True).transform
    by_island: dict[str, list[tuple[str, Any]]] = {}
    with fiona.open(shp) as src:
        for f in src:
            isl = f["properties"]["island"]
            g = transform(to_utm, shape(f["geometry"]))
            by_island.setdefault(isl, []).append((f["properties"]["seagrass_c"], g))

    out: dict[str, dict[str, Any]] = {}
    for grp, islands in GROUPS.items():
        burn = []
        allg = []
        for isl in islands:
            for cls, g in by_island[isl]:
                burn.append((g, CLASS_LABEL[cls]))
                allg.append(g)
        out[grp] = {"footprint": unary_union(allg), "burn": burn}
    return out


def tile_origins(footprint: Any, tile_px: int) -> list[tuple[int, int]]:
    """UTM (col,row) pixel origins of TILE-sized tiles whose bbox intersects footprint.

    Origins are snapped to a global grid that is a multiple of tile_px*RES so tiles are
    reproducible and non-overlapping.
    """
    minx, miny, maxx, maxy = footprint.bounds
    step = tile_px * RES
    origins = []
    # snap down to grid
    x0 = (int(minx) // step) * step
    y0 = (int(miny) // step) * step
    x = x0
    while x < maxx:
        y = y0
        while y < maxy:
            if footprint.intersects(box(x, y, x + step, y + step)):
                # pixel coords: col = x/RES ; row uses negative y res, row = -y/RES upper edge
                origins.append((x, y))
            y += step
        x += step
    return origins


def create_tile(**job: Any) -> dict[int, int]:
    """Create one eval tile window with a rasterized GT label; return pixel-class counts."""
    x_utm, y_utm = job["x"], job["y"]  # lower-left UTM meters
    tile_px = job["tile_px"]
    step = tile_px * RES
    projection = Projection(CRS.from_epsg(UTM), RES, -RES)
    # pixel bounds: col from x/RES; rows are negative (y down). Upper edge (y+step) -> row.
    col0 = int(round(x_utm / RES))
    row0 = int(round(-(y_utm + step) / RES))  # top row (north edge)
    bounds = (col0, row0, col0 + tile_px, row0 + tile_px)

    # Affine mapping pixel -> UTM for rasterization (north-up):
    # x = x_utm + col*RES ; y = (y_utm+step) - row*RES
    tf = Affine(RES, 0, x_utm, 0, -RES, y_utm + step)
    shapes = [
        (mapping(g), lab)
        for g, lab in job["burn"]
        if g.intersects(box(x_utm, y_utm, x_utm + step, y_utm + step))
    ]
    raster = np.full((tile_px, tile_px), 255, dtype=np.uint8)
    if shapes:
        # burn negatives (0) then positives (2): sort so label 2 is burned last (wins overlaps)
        shapes.sort(key=lambda s: s[1])
        rasterio.features.rasterize(
            shapes, out=raster, transform=tf, fill=255, all_touched=False
        )

    window = Window(
        storage=StorageConfig()
        .instantiate_window_storage_factory()
        .get_storage(job["ds_path"]),
        group=job["group"],
        name=f"tile_{job['grp']}_{col0}_{row0}",
        projection=projection,
        bounds=bounds,
        time_range=(
            datetime(job["year"], 1, 1, tzinfo=timezone.utc),
            datetime(job["year"], 12, 31, tzinfo=timezone.utc),
        ),
        options={"eval_group": job["grp"], "island_group": job["grp"]},
    )
    window.save()
    raster_dir = window.get_raster_dir(LABEL_LAYER, [LABEL_BAND])
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        window.projection,
        window.bounds,
        RasterArray(chw_array=raster[None]),
    )
    window.mark_layer_completed(LABEL_LAYER)
    v, c = np.unique(raster, return_counts=True)
    return {int(vi): int(ci) for vi, ci in zip(v, c)}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shp", default=str(DEFAULT_SHP))
    p.add_argument("--ds_path", default=str(DEFAULT_DATASET_PATH))
    p.add_argument("--group", default="baleares_official_eval")
    p.add_argument("--tile_px", type=int, default=512)
    p.add_argument("--year", type=int, default=2025)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--only_group",
        default=None,
        help="restrict to one of mallorca/menorca/pitiusas",
    )
    return p.parse_args()


def main() -> None:
    """Create all Baleares polygon-footprint eval tiles."""
    args = parse_args()
    ds_path = UPath(args.ds_path)
    geoms = load_group_geoms(Path(args.shp))
    jobs = []
    for grp, gd in geoms.items():
        if args.only_group and grp != args.only_group:
            continue
        origins = tile_origins(gd["footprint"], args.tile_px)
        print(
            f"{grp}: {len(origins)} tiles (footprint {gd['footprint'].area/1e6:.0f} km2)"
        )
        for x, y in origins:
            jobs.append(
                dict(
                    x=x,
                    y=y,
                    grp=grp,
                    burn=gd["burn"],
                    tile_px=args.tile_px,
                    ds_path=ds_path,
                    group=args.group,
                    year=args.year,
                )
            )
    print(f"creating {len(jobs)} eval tiles in {ds_path}/windows/{args.group} ...")
    from collections import Counter

    agg: Counter = Counter()
    with multiprocessing.Pool(args.workers) as pool:
        for counts in tqdm.tqdm(
            star_imap_unordered(pool, create_tile, jobs), total=len(jobs)
        ):
            agg.update(counts)
    pos = agg.get(2, 0)
    neg = agg.get(0, 0)
    ign = agg.get(255, 0)
    tot = pos + neg
    print(
        f"pixel labels -> positive(dense/sparse)={pos:,}  negative(non)={neg:,}  ignore={ign:,}"
    )
    if tot:
        print(f"evaluable pixels={tot:,}  positive_fraction={pos/tot:.3f}")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
