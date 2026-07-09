"""Phase 1: download + normalize RPG for every French territory.

For each territory in territories.py: fetch the RPG archive (or use a local copy),
extract the parcel layer, and write a normalized GeoPackage with just the fields we
need -- geometry + CODE_CULTU + the mapped PASTIS class_id -- reprojected to nothing
(kept in native CRS; window creation handles projection).

This is a scaffold: the exact IGN/data.gouv download URLs are versioned per year, so
fill Territory.url (or Territory.local) before running. Everything downstream keys off
the normalized GPKGs this produces at data/rpg/<key>.gpkg.

Run:  python download_rpg.py --year 2019 [--territory metropole]
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import geopandas as gpd  # rslearn env has geopandas/fiona
import pyogrio

from territories import TERRITORIES, Territory

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data" / "rpg"
CLASS_MAP = json.loads((HERE / "pastis_rpg_class_map.json").read_text())["code_to_class"]


def _fetch(t: Territory, year: int) -> Path:
    """Return a local path to the territory's RPG archive/dir (download if needed)."""
    dest = RAW / str(year) / t.key
    dest.mkdir(parents=True, exist_ok=True)
    if t.local:
        return Path(t.local)
    if not t.url:
        raise SystemExit(
            f"[{t.key}] no url/local set for year {year}. Fill Territory.url from "
            f"IGN geoservices / data.gouv (RPG 2-0 PARCELLES) or set Territory.local."
        )
    archive = dest / Path(t.url).name
    if not archive.exists() or archive.stat().st_size == 0:
        import shutil
        import urllib.request

        print(f"[{t.key}] downloading {t.url}")
        # IGN geoplateforme rejects urllib's default User-Agent (HTTP 403); send a
        # browser-like UA. Stream to disk (files can be GBs).
        req = urllib.request.Request(t.url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(archive, "wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f)
    # IGN RPG archives are .7z; older data.gouv ones are .zip.
    marker = dest / ".extracted"
    if not marker.exists():
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as z:
                z.extractall(dest)
        elif archive.suffix == ".7z":
            import py7zr

            print(f"[{t.key}] extracting {archive.name}")
            with py7zr.SevenZipFile(archive) as z:
                z.extractall(dest)
        marker.touch()
    return dest


def _find_parcels_layer(path: Path) -> Path:
    """Locate the PARCELLES shapefile/gpkg under an extracted RPG delivery."""
    # Prefer the PARCELLES layer; a delivery also ships ILOTS_ANONYMES.* (no crop code),
    # so the generic *.gpkg/*.shp globs must come last.
    for pat in ("**/PARCELLES_GRAPHIQUES.shp", "**/*PARCELLES*.shp",
                "**/PARCELLES_GRAPHIQUES.gpkg", "**/*PARCELLES*.gpkg", "**/*.gpkg"):
        hits = sorted(path.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"no PARCELLES layer under {path}")


def normalize(t: Territory, year: int) -> Path:
    """Write data/rpg/<key>.gpkg with geometry + code_cultu + pastis class_id.

    Reads only the crop-code column (+ geometry) via pyogrio so the national metropole
    delivery (~9M parcels, 3.4 GB) doesn't pull every attribute into memory.
    """
    src = _find_parcels_layer(_fetch(t, year))
    print(f"[{t.key}] reading {src}")
    # RPG parcel crop code column is CODE_CULTU (older group-only products use CODE_GROUP).
    # Resolve the parcels layer explicitly (GPKG deliveries name it 'parcelle_graphique'),
    # read only the code column, then match it case-insensitively on the actual frame.
    layers = pyogrio.list_layers(src)[:, 0].tolist()
    layer = next((la for la in layers if "parcelle" in la.lower()), layers[0])
    info = pyogrio.read_info(src, layer=layer)
    fields = list(info["fields"])
    code_col = next(c for c in fields if c.upper() in ("CODE_CULTU", "CODE_GROUP"))
    gdf = gpd.read_file(src, engine="pyogrio", layer=layer, columns=[code_col])
    gdf["class_id"] = gdf[code_col].map(CLASS_MAP).fillna(0).astype("int16")
    gdf = gdf.rename(columns={code_col: "code_cultu"})
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{t.key}.gpkg"
    gdf.to_file(out, driver="GPKG")
    n_pos = int((gdf["class_id"] > 0).sum())
    print(f"[{t.key}] {len(gdf)} parcels ({n_pos} with a positive PASTIS class) -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--territory", default=None, help="one key, else all")
    args = ap.parse_args()
    todo = [t for t in TERRITORIES if args.territory in (None, t.key)]
    for t in todo:
        normalize(t, args.year)


if __name__ == "__main__":
    main()
