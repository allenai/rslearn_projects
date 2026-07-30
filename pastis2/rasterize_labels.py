"""Phase 4 (labels): rasterize RPG crop parcels into per-window class masks.

rslearn has no built-in vector->raster, so after windows exist we burn the parcels'
`class_id` onto each window's pixel grid (aligned to its projection + bounds, i.e. the
same grid Sentinel-2 materializes on) and write a `label` raster layer. Background/no
parcel = 0; the nodata value 255 marks pixels with no coverage if needed.

Reads directly from the normalized RPG GeoPackages (download_rpg.py output) rather than
the materialized vector layer, so it depends only on the well-understood Window
geometry API + rasterio. Adapts the FindStadiums example
(docs/examples/FindStadiums.md) for the encode/complete calls.

Run (after build_windows.py; can run before or after S2 materialize):
  python rasterize_labels.py --dataset /path/to/rslearn_ds --group rpg_2019

Scale note: this loads all parcels into memory with a spatial index. For national
scale, shard by territory/département or stream per-window from the GPKG instead.
"""

from __future__ import annotations

import argparse
import os
import pathlib
from pathlib import Path

import geopandas as gpd
from upath import UPath
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import Affine

from rslearn.dataset.dataset import Dataset
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat

HERE = Path(__file__).parent
RPG = pathlib.Path(os.environ.get("PASTIS2_RPG_DIR", HERE / "data" / "rpg"))
LABEL_LAYER = "label"
LABEL_BANDS = ["class_id"]


def load_all_parcels() -> gpd.GeoDataFrame:
    """Concatenate every territory GPKG into one GeoDataFrame (EPSG:4326) for indexing."""
    gdfs = []
    for gpkg in sorted(RPG.glob("*.gpkg")):
        g = gpd.read_file(gpkg)[["class_id", "geometry"]].to_crs(4326)
        gdfs.append(g)
        print(f"  loaded {len(g)} parcels from {gpkg.name}")
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=4326)
    gdf.sindex  # build spatial index
    return gdf


def rasterize_window(window, parcels_4326: gpd.GeoDataFrame) -> np.ndarray:
    """Return a (H, W) uint8 class-id mask for one window (0 = background)."""
    proj = window.projection
    minx, miny, maxx, maxy = window.bounds  # pixel coords
    width, height = maxx - minx, maxy - miny
    xr, yr = proj.x_resolution, proj.y_resolution  # yr is negative

    # Window pixel grid -> CRS affine: geo = pixel * resolution.
    transform = Affine(xr, 0.0, minx * xr, 0.0, yr, miny * yr)

    # Query candidate parcels via the window's geo bbox (in the window CRS -> 4326).
    from shapely.geometry import box

    bbox = box(minx * xr, maxy * yr, maxx * xr, miny * yr)
    bbox_4326 = gpd.GeoSeries([bbox], crs=proj.crs).to_crs(4326).iloc[0]
    idx = list(parcels_4326.sindex.query(bbox_4326, predicate="intersects"))
    if not idx:
        return np.zeros((height, width), dtype=np.uint8)

    sub = parcels_4326.iloc[idx].to_crs(proj.crs)
    shapes = [
        (geom, int(cid))
        for geom, cid in zip(sub.geometry, sub["class_id"])
        if cid > 0  # 0 is background; leave as fill
    ]
    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    return features.rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="rslearn dataset root")
    ap.add_argument("--group", required=True, help="window group, e.g. rpg_2019")
    args = ap.parse_args()

    print("loading parcels ...")
    parcels = load_all_parcels()

    ds = Dataset(UPath(args.dataset))
    windows = ds.load_windows(groups=[args.group])
    print(f"rasterizing labels for {len(windows)} windows ...")

    fmt = GeotiffRasterFormat()
    for i, window in enumerate(windows):
        mask = rasterize_window(window, parcels)  # (H, W)
        raster_dir = window.get_raster_dir(LABEL_LAYER, LABEL_BANDS)
        # encode_raster wants a RasterArray (C, H, W) aligned to (projection, bounds).
        fmt.encode_raster(
            raster_dir, window.projection, window.bounds,
            RasterArray(chw_array=mask[None]),
        )
        window.mark_layer_completed(LABEL_LAYER)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(windows)} done", flush=True)

    print("done")


if __name__ == "__main__":
    main()
