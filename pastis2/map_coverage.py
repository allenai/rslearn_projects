"""Make coverage maps for the PASTIS2 DROM collection:
 1) a global map highlighting the 5 overseas regions we're collecting, and
 2) one map per territory showing the window-footprint (pixel) coverage polygons.

Reads window footprints straight from the built national_ds windows (read-only).
Outputs PNGs to pastis2/maps/.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pyproj
from shapely.geometry import box
from shapely.ops import transform, unary_union

HERE = Path(__file__).parent
DS = HERE / "data" / "national_ds"
GROUP = "rpg_2019"
OUT = HERE / "maps"
OUT.mkdir(exist_ok=True)

TERR = {  # name -> (lon range, lat range) for assignment + label placement
    "Guadeloupe": ((-62.0, -60.9), (15.7, 16.6)),
    "Martinique": ((-61.3, -60.7), (14.3, 15.0)),
    "Guyane":     ((-55.0, -51.0), (2.0, 6.2)),
    "Réunion":    ((55.0, 56.0), (-21.5, -20.7)),
    "Mayotte":    ((44.9, 45.4), (-13.1, -12.5)),
}
COLORS = {"Guadeloupe": "#e6194B", "Martinique": "#f58231", "Guyane": "#3cb44b",
          "Réunion": "#4363d8", "Mayotte": "#911eb4"}

_TF: dict[int, pyproj.Transformer] = {}


def to_wgs(crs_epsg: int):
    if crs_epsg not in _TF:
        _TF[crs_epsg] = pyproj.Transformer.from_crs(crs_epsg, 4326, always_xy=True)
    return _TF[crs_epsg].transform


def assign(lon: float, lat: float) -> str | None:
    for name, ((lo, hi), (la, lb)) in TERR.items():
        if lo < lon < hi and la < lat < lb:
            return name
    return None


def load_footprints() -> dict[str, list]:
    """territory -> list of shapely polygons (window footprints in lon/lat)."""
    out: dict[str, list] = {k: [] for k in TERR}
    for wdir in glob.glob(str(DS / "windows" / GROUP / "*")):
        mp = Path(wdir) / "metadata.json"
        if not mp.exists():
            continue
        m = json.load(open(mp))
        crs = int(str(m["projection"]["crs"]).split(":")[-1])
        xr, yr = m["projection"]["x_resolution"], m["projection"]["y_resolution"]
        minx, miny, maxx, maxy = m["bounds"]
        gx0, gx1 = minx * xr, maxx * xr
        gy0, gy1 = miny * yr, maxy * yr
        poly = box(min(gx0, gx1), min(gy0, gy1), max(gx0, gx1), max(gy0, gy1))
        poly_ll = transform(to_wgs(crs), poly)
        c = poly_ll.centroid
        terr = assign(c.x, c.y)
        if terr:
            out[terr].append(poly_ll)
    return out


def global_map(fp: dict[str, list]) -> None:
    fig = plt.figure(figsize=(13, 6.5))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#eeeee4")
    ax.add_feature(cfeature.OCEAN, facecolor="#cfe8f3")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, edgecolor="#999")
    for name, ((lo, hi), (la, lb)) in TERR.items():
        clon, clat = (lo + hi) / 2, (la + lb) / 2
        n = len(fp[name])
        ax.plot(clon, clat, marker="*", markersize=16, color=COLORS[name],
                markeredgecolor="black", markeredgewidth=0.6, transform=ccrs.PlateCarree(), zorder=5)
        ax.annotate(f"{name}\n({n} windows)", xy=(clon, clat),
                    xytext=(clon + 6, clat + 6), transform=ccrs.PlateCarree(),
                    fontsize=9, fontweight="bold", color="black",
                    xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
                    arrowprops=dict(arrowstyle="-", color=COLORS[name], lw=1))
    total = sum(len(v) for v in fp.values())
    ax.set_title(f"PASTIS2 — 5 overseas French territories (DROM), {total} windows total", fontsize=13)
    fig.savefig(OUT / "global_drom.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote global_drom.png")


def territory_map(name: str, polys: list) -> None:
    if not polys:
        print(f"  {name}: no footprints"); return
    lons = [p.bounds for p in polys]
    minx = min(b[0] for b in lons); miny = min(b[1] for b in lons)
    maxx = max(b[2] for b in lons); maxy = max(b[3] for b in lons)
    mx = max(0.05, (maxx - minx) * 0.08); my = max(0.05, (maxy - miny) * 0.08)
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([minx - mx, maxx + mx, miny - my, maxy + my], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f2f2ec")
    ax.add_feature(cfeature.OCEAN, facecolor="#cfe8f3")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # coverage: filled window footprints (dissolved for a clean coverage polygon,
    # plus faint individual cells so the 1.28 km sampling is visible)
    merged = unary_union(polys)
    geoms = list(getattr(merged, "geoms", [merged]))
    for g in geoms:
        ax.add_geometries([g], ccrs.PlateCarree(), facecolor=COLORS[name],
                          edgecolor=COLORS[name], alpha=0.55, linewidth=0)
    for p in polys:
        ax.add_geometries([p], ccrs.PlateCarree(), facecolor="none",
                          edgecolor=COLORS[name], linewidth=0.3, alpha=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax.set_title(f"{name} — PASTIS2 window coverage ({len(polys)} windows, 1.28 km cells)", fontsize=12)
    safe = name.replace("é", "e")
    fig.savefig(OUT / f"coverage_{safe}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote coverage_{safe}.png ({len(polys)} windows)")


def main() -> None:
    fp = load_footprints()
    print("footprints per territory:", {k: len(v) for k, v in fp.items()})
    global_map(fp)
    for name, polys in fp.items():
        territory_map(name, polys)


if __name__ == "__main__":
    main()
