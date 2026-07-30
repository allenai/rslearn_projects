"""QA viz: per-territory sample panels straight from the materialized windows.

Read-only (does NOT touch the dataset, so it's safe to run while materialize is still
going). For each territory it assigns windows by lon/lat centroid, rasterizes the RPG
label on the fly (bbox-filtered read of that territory's GPKG -- no giant in-memory
load), picks the N windows with the most positive-class pixels, and renders a panel:
S2 true-colour + S1 (vv) + label mask, one column per window. Saves viz_qa/<terr>.png.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyogrio
import pyproj
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import box
from shapely.ops import transform as shp_transform

HERE = Path(__file__).parent
DS = HERE / "data" / "qa100_ds"
GROUP = "rpg_2019"
RPG = HERE / "data" / "rpg"
# Use the EXPANDED class map (original 24 + added tropical classes). Rasterize from the
# raw code_cultu via this map, since the GPKGs' stored class_id was baked with the
# original map and won't reflect the new classes.
CLASSMAP = os.environ.get("PASTIS2_CLASSMAP", str(HERE / "pastis_rpg_class_map_expanded.json"))
_CMAP = json.load(open(CLASSMAP))
NAMES = {int(k): v for k, v in _CMAP["class_names"].items()}
CODE2CLASS = {str(k): int(v) for k, v in _CMAP["code_to_class"].items()}
NCLS = max(NAMES) + 1
ORIG_MAX = 24  # classes 1..24 are the original positives; 25+ are the added ones
import pyogrio as _pio
_FIELDS = _pio.read_info(str(RPG / "martinique.gpkg"))["fields"]
CODECOL = next(c for c in ("code_cultu", "CODE_CULTU", "code", "CODE") if c in _FIELDS)

# S2 band order in the stored geotiff: B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12
RGB_IDX = [2, 1, 0]  # B04,B03,B02

TERR_GPKG_CRS: dict[str, int] = {
    "metropole": 2154, "guadeloupe": 5490, "martinique": 5490,
    "guyane": 2972, "reunion": 2975, "mayotte": 4471,
}


def territory_of(lon: float, lat: float) -> str:
    if -6 < lon < 10 and 41 < lat < 52:
        return "metropole"
    if -62 < lon < -60.9 and 15.7 < lat < 16.6:
        return "guadeloupe"
    if -61.3 < lon < -60.7 and 14.3 < lat < 15.0:
        return "martinique"
    if -55 < lon < -51 and 2 < lat < 6:
        return "guyane"
    if 55.0 < lon < 56.0 and -21.5 < lat < -20.7:
        return "reunion"
    if 44.9 < lon < 45.4 and -13.1 < lat < -12.5:
        return "mayotte"
    return "unknown"


def window_geo(meta: dict) -> tuple[str, tuple[float, float, float, float], float, float]:
    """Return (crs, geo_bbox(minx,miny,maxx,maxy), center_lon, center_lat)."""
    crs = meta["projection"]["crs"]
    xr, yr = meta["projection"]["x_resolution"], meta["projection"]["y_resolution"]
    minx, miny, maxx, maxy = meta["bounds"]
    gx0, gx1 = minx * xr, maxx * xr
    gy0, gy1 = miny * yr, maxy * yr
    gbb = (min(gx0, gx1), min(gy0, gy1), max(gx0, gx1), max(gy0, gy1))
    t = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = t.transform((gbb[0] + gbb[2]) / 2, (gbb[1] + gbb[3]) / 2)
    return crs, gbb, lon, lat


def rasterize_label(meta: dict, terr: str) -> np.ndarray:
    """(128,128) uint8 class mask, bbox-filtered read of the territory GPKG."""
    crs = meta["projection"]["crs"]
    xr, yr = meta["projection"]["x_resolution"], meta["projection"]["y_resolution"]
    minx, miny, maxx, maxy = meta["bounds"]
    w, h = maxx - minx, maxy - miny
    transform = Affine(xr, 0, minx * xr, 0, yr, miny * yr)
    gpkg = RPG / f"{terr}.gpkg"
    info = pyogrio.read_info(gpkg)
    gcrs = info["crs"]
    # window geo bbox -> gpkg crs bbox
    to_g = pyproj.Transformer.from_crs(crs, gcrs, always_xy=True).transform
    poly = shp_transform(to_g, box(minx * xr, maxy * yr, maxx * xr, miny * yr))
    gb = poly.bounds
    gdf = pyogrio.read_dataframe(gpkg, columns=[CODECOL], bbox=gb)
    if len(gdf) == 0:
        return np.zeros((h, w), np.uint8)
    gdf = gdf.to_crs(crs)
    shapes = [
        (geom, CODE2CLASS.get(str(code), 0))
        for geom, code in zip(gdf.geometry, gdf[CODECOL])
    ]
    shapes = [(geom, cid) for geom, cid in shapes if cid > 0]
    if not shapes:
        return np.zeros((h, w), np.uint8)
    return features.rasterize(shapes, out_shape=(h, w), transform=transform, fill=0, dtype="uint8")


def read_best_s2(wdir: Path) -> np.ndarray | None:
    """RGB (H,W,3) float in [0,1] from the least-cloudy available S2 month."""
    best, best_score = None, -1.0
    for d in sorted(glob.glob(str(wdir / "layers" / "sentinel2*"))):
        tif = glob.glob(os.path.join(d, "*", "geotiff.tif"))
        if not tif:
            continue
        with rasterio.open(tif[0]) as src:
            arr = src.read().astype(np.float32)  # (10,H,W)
        score = float((arr > 0).mean())  # prefer fewer nodata/zero pixels
        if score > best_score:
            best_score, best = score, arr
    if best is None:
        return None
    rgb = best[RGB_IDX].transpose(1, 2, 0)
    return np.clip(rgb / 3000.0, 0, 1)


def read_s1(wdir: Path) -> np.ndarray | None:
    for d in sorted(glob.glob(str(wdir / "layers" / "sentinel1*"))):
        tif = glob.glob(os.path.join(d, "*", "geotiff.tif"))
        if tif:
            with rasterio.open(tif[0]) as src:
                vv = src.read(1).astype(np.float32)
            if float((vv != 0).mean()) < 0.5:  # mostly-empty tile; skip
                continue
            # RTC vv is linear power (~0.01-1.2); show as dB with a [-25,0] stretch.
            vv_db = 10.0 * np.log10(np.clip(vv, 1e-4, None))
            return np.clip((vv_db + 25.0) / 25.0, 0, 1)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-territory", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "viz_qa"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)

    wins = sorted(glob.glob(str(DS / "windows" / GROUP / "*")))
    by_terr: dict[str, list] = {}
    for wdir in wins:
        wdir = Path(wdir)
        mp = wdir / "metadata.json"
        if not mp.exists():
            continue
        # only windows with at least one materialized S2 month
        if not glob.glob(str(wdir / "layers" / "sentinel2*" / "*" / "geotiff.tif")):
            continue
        meta = json.load(open(mp))
        _, _, lon, lat = window_geo(meta)
        terr = territory_of(lon, lat)
        by_terr.setdefault(terr, []).append((wdir, meta))

    # fixed color map over 0..24
    palette = np.vstack([
        plt.cm.tab20(np.linspace(0, 1, 20)),
        plt.cm.tab20b(np.linspace(0, 1, 20)),
        plt.cm.tab20c(np.linspace(0, 1, 20)),
    ])
    colors = np.vstack([[0.85, 0.85, 0.85, 1], palette[1:NCLS]])[:NCLS]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, NCLS + 0.5, 1), cmap.N)

    for terr, items in by_terr.items():
        if terr == "unknown":
            print(f"skip {len(items)} unknown-territory windows"); continue
        scored = []
        for wdir, meta in items:
            mask = rasterize_label(meta, terr)
            # rank by ORIGINAL positives (1..24) so the same windows are picked as
            # the pre-expansion run; the new classes (25+) still render in the mask.
            posfrac = float(((mask >= 1) & (mask <= ORIG_MAX)).mean())
            scored.append((posfrac, wdir, meta, mask))
        scored.sort(key=lambda x: -x[0])
        pick = scored[: args.per_territory]
        n = len(pick)
        fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), squeeze=False)
        present = set()
        for j, (posfrac, wdir, meta, mask) in enumerate(pick):
            rgb = read_best_s2(wdir); s1 = read_s1(wdir)
            axes[0, j].imshow(rgb if rgb is not None else np.zeros((128, 128, 3)))
            axes[0, j].set_title(f"S2 RGB\npos={posfrac*100:.0f}%", fontsize=8)
            axes[1, j].imshow(s1 if s1 is not None else np.zeros((128, 128)), cmap="gray")
            axes[1, j].set_title("S1 (vv)", fontsize=8)
            axes[2, j].imshow(mask, cmap=cmap, norm=norm, interpolation="nearest")
            axes[2, j].set_title("label mask", fontsize=8)
            present.update(int(c) for c in np.unique(mask) if c > 0)
            for r in range(3):
                axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        # legend of present classes
        from matplotlib.patches import Patch
        handles = [Patch(color=colors[c], label=f"{c} {NAMES.get(c, '?')}") for c in sorted(present)]
        if handles:
            fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), fontsize=7)
        fig.suptitle(f"PASTIS2 QA — {terr} ({len(items)} windows materialized; top {n} by positive %)", fontsize=11)
        fig.tight_layout(rect=[0, 0.06, 1, 0.97])
        p = out / f"qa_{terr}.png"
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f"saved {p}  ({len(items)} windows, showing {n})")


if __name__ == "__main__":
    main()
