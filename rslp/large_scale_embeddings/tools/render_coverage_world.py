"""Render the coverage mask over a coastline basemap, in Equal Earth.

Writes data/large_scale_embeddings/coverage_world.png, the figure the README points at.
Re-run it whenever coverage_mask.tif changes; the PNG is a committed artifact and does
not regenerate itself, so it had drifted six days behind the mask once already.

Equal Earth (EPSG:8857) rather than plate carree: the figure exists to show how much of
the world is covered, and an equirectangular plot inflates the high latitudes, which is
exactly where the coverage decisions are.

Needs Natural Earth admin-1 polygons, the same download make_borders.py uses.
"""

import os
from pathlib import Path

import fiona
import numpy as np
import numpy.typing as npt
import pyproj
import rasterio
from PIL import Image, ImageDraw
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from shapely.geometry import shape

# Source grid, a coarser multiple of the mask's 1/120 degree so the downsample is exact.
SRC_RES = 1.0 / 30.0
SRC_W, SRC_H = int(360 / SRC_RES), int(180 / SRC_RES)
# Output canvas. Equal Earth is about 2.05:1, so this keeps pixels near square.
DW, DH = 5400, 2640

WATER, LAND, COVERED = 0, 1, 2
COL = {WATER: (226, 238, 243), LAND: (176, 174, 140), COVERED: (199, 92, 155)}
COAST = (40, 92, 78)
# UTM is defined from 80S to 84N. Outside that there is no zone to tile, so those bands
# are hatched rather than left looking merely uncovered.
UTM_NORTH, UTM_SOUTH = 84.0, -80.0

NE = os.environ.get(
    "NE_ADMIN1_SHP",
    str(
        Path.home()
        / "Downloads/ne_10m_admin_1_states_provinces"
        / "ne_10m_admin_1_states_provinces.shp"
    ),
)
MASK = "data/large_scale_embeddings/coverage_mask.tif"
OUT = "data/large_scale_embeddings/coverage_world.png"


def land_grid() -> npt.NDArray[np.bool_]:
    """Rasterize Natural Earth land polygons onto the source grid.

    Returns:
        a boolean array, True over land.
    """
    img = Image.new("1", (SRC_W, SRC_H), 0)
    draw = ImageDraw.Draw(img)
    with fiona.open(NE) as src:
        for feat in src:
            geom = shape(feat["geometry"])
            parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for poly in parts:
                pts = [
                    ((x + 180.0) / SRC_RES, (90.0 - y) / SRC_RES)
                    for x, y in poly.exterior.coords
                ]
                if len(pts) > 2:
                    draw.polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def covered_grid() -> npt.NDArray[np.bool_]:
    """Downsample the coverage mask onto the source grid.

    Block-any rather than block-mean, so a coastal strip one mask pixel wide survives
    instead of averaging away.

    Returns:
        a boolean array, True where covered.
    """
    with rasterio.open(MASK) as src:
        m = src.read(1) > 0
    fy, fx = m.shape[0] // SRC_H, m.shape[1] // SRC_W
    return m[: fy * SRC_H, : fx * SRC_W].reshape(SRC_H, fy, SRC_W, fx).any(axis=(1, 3))


def main() -> None:
    """Build the figure and write it to OUT."""
    cat = np.where(land_grid(), LAND, WATER).astype(np.uint8)
    cat[covered_grid()] = COVERED

    ee = pyproj.CRS.from_epsg(8857)
    fwd = pyproj.Transformer.from_crs("EPSG:4326", ee, always_xy=True)
    inv = pyproj.Transformer.from_crs(ee, "EPSG:4326", always_xy=True)
    xmax = abs(fwd.transform(180, 0)[0])
    ymax = abs(fwd.transform(0, 90)[1])
    px_x, px_y = 2 * xmax / DW, 2 * ymax / DH

    dst = np.zeros((DH, DW), np.uint8)
    reproject(
        cat,
        dst,
        src_transform=from_origin(-180, 90, SRC_RES, SRC_RES),
        src_crs="EPSG:4326",
        dst_transform=from_origin(-xmax, ymax, px_x, px_y),
        dst_crs=ee,
        resampling=Resampling.nearest,
        src_nodata=None,
        dst_nodata=None,
    )

    # pyproj's Equal Earth inverse aliases points outside the globe onto valid lon/lat
    # instead of returning inf, so membership has to be tested by round-tripping.
    gx, gy = np.meshgrid(
        -xmax + (np.arange(DW) + 0.5) * px_x, ymax - (np.arange(DH) + 0.5) * px_y
    )
    lo, la = inv.transform(gx, gy)
    x2, y2 = fwd.transform(lo, la)
    inside = (
        np.isfinite(lo)
        & np.isfinite(la)
        & (np.abs(x2 - gx) < px_x)
        & (np.abs(y2 - gy) < px_y)
    )

    rgba = np.zeros((DH, DW, 4), np.uint8)
    for v, c in COL.items():
        sel = dst == v
        rgba[sel, 0], rgba[sel, 1], rgba[sel, 2], rgba[sel, 3] = c[0], c[1], c[2], 255
    rgba[~inside, 3] = 0

    # Coastline, taken as the edge of the land/covered classes so no separate border
    # file is needed. A warped 1px line would break into dashes, hence drawing it here.
    solid = (dst != WATER) & inside
    edge = solid ^ np.pad(solid, ((0, 0), (1, 0)))[:, :-1]
    edge |= solid ^ np.pad(solid, ((1, 0), (0, 0)))[:-1, :]
    rgba[edge & inside, :3] = COAST

    beyond = inside & ((la > UTM_NORTH) | (la < UTM_SOUTH))
    hatch = beyond & (((np.arange(DH)[:, None] + np.arange(DW)[None, :]) % 14) < 2)
    rgba[hatch, :3] = (rgba[hatch, :3] * 0.55).astype(np.uint8)

    Image.fromarray(rgba, "RGBA").save(OUT, optimize=True)
    south = la[(dst == COVERED) & inside].min()
    print(f"wrote {OUT}  southernmost covered lat {south:.2f}")


if __name__ == "__main__":
    main()
