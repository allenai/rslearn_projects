"""Render the coverage mask over a coastline basemap.

Writes data/large_scale_embeddings/coverage_world.png, the figure the README points at.
Re-run it whenever coverage_mask.tif changes; the PNG is a committed artifact and does
not regenerate itself, so it had drifted six days behind the mask once already.

Needs Natural Earth admin-1 polygons, the same download make_borders.py uses.
"""

import os
from pathlib import Path

import fiona
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from shapely.geometry import shape

W, H = 5400, 2700
LAND = (176, 174, 140)
WATER = (226, 238, 243)
COVERED = (214, 92, 170)
OUTLINE = (40, 92, 78)
GRID = (255, 255, 255)
HATCH = (120, 120, 120)
# UTM is defined from 80S to 84N. Outside that there is no zone to tile, so the
# bands are hatched rather than left looking merely uncovered.
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


def to_px(lon: float, lat: float) -> tuple[float, float]:
    """Equirectangular lon/lat to pixel coordinates.

    Args:
        lon: longitude in degrees.
        lat: latitude in degrees.

    Returns:
        the (x, y) pixel position.
    """
    return ((lon + 180.0) / 360.0 * W, (90.0 - lat) / 180.0 * H)


img = Image.new("RGB", (W, H), WATER)
draw = ImageDraw.Draw(img)

# Basemap: admin-1 polygons dissolved visually by drawing them all filled.
with fiona.open(NE) as src:
    for feat in src:
        geom = shape(feat["geometry"])
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for poly in parts:
            pts = [to_px(x, y) for x, y in poly.exterior.coords]
            if len(pts) > 2:
                draw.polygon(pts, fill=LAND)

# Graticule, every 30 degrees.
for lon in range(-150, 180, 30):
    x = to_px(lon, 0)[0]
    draw.line([(x, 0), (x, H)], fill=GRID, width=1)
for lat in range(-60, 90, 30):
    y = to_px(0, lat)[1]
    draw.line([(0, y), (W, y)], fill=GRID, width=1)

# Hatch the bands outside the UTM grid, before the coverage overlay so it draws on top.
for y0, y1 in ((0, to_px(0, UTM_NORTH)[1]), (to_px(0, UTM_SOUTH)[1], H)):
    y0, y1 = int(y0), int(y1)
    for off in range(-(y1 - y0), W, 14):
        draw.line([(off, y1), (off + (y1 - y0), y0)], fill=HATCH, width=1)
    for x in range(0, W, 24):
        draw.line([(x, y0), (x + 12, y0)], fill=(60, 60, 60), width=2)
        draw.line([(x, y1 - 1), (x + 12, y1 - 1)], fill=(60, 60, 60), width=2)

base = np.asarray(img).astype(np.float32)

# Coverage overlay, downsampled by block-max so a thin coastal strip still shows.
with rasterio.open(MASK) as src:
    m = src.read(1) > 0
fy, fx = m.shape[0] // H, m.shape[1] // W
cov = m[: fy * H, : fx * W].reshape(H, fy, W, fx).any(axis=(1, 3))

# Lightened, so the basemap reads through. See 22ee752d.
alpha = 0.45
out = base.copy()
out[cov] = base[cov] * (1 - alpha) + np.array(COVERED, np.float32) * alpha

# Outline the covered region so its edge against uncovered land is legible.
edge = cov ^ np.pad(cov, ((0, 0), (1, 0)), constant_values=False)[:, :-1]
edge |= cov ^ np.pad(cov, ((1, 0), (0, 0)), constant_values=False)[:-1, :]
out[edge] = OUTLINE

Image.fromarray(out.astype(np.uint8)).save(OUT, optimize=True)
south = np.where(cov.any(axis=1))[0].max()
print(f"wrote {OUT}  southernmost covered lat {90 - south * 180 / H:.2f}")
