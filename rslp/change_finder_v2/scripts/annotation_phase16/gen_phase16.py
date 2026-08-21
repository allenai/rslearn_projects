"""Generate phase16 v2 annotation entries from external ASM datasets.

Each dataset contributes up to 100 entries. Entries follow the conventions of
annotations_phase15: 128x128 window at 10 m in the local UTM zone, group
"phase16", window_name "<dataset_slug>_<i>". Datasets with exact locations get
one positive point (pre_change = dataset date when available; post_change and
first_date_change_noticeable left unset so prepare.py skips them until fully
annotated). Datasets with only rough locations get entries with no points.

See the README in this directory for where to download each source dataset and
how the data dir must be laid out. Writes partial_<dataset>.json files into the
data dir; combine them with combine_validate.py.

Usage::

    python -m rslp.change_finder_v2.scripts.annotation_phase16.gen_phase16 \
        --data-dir /path/to/downloaded_datasets/ [dataset ...]
"""

from __future__ import annotations

import json
import random
import re
from datetime import date
from typing import TypeVar

import pyproj
import shapely
import shapely.geometry

T = TypeVar("T")

# Set from --data-dir in main().
SP = ""
WINDOW = 128
HALF = WINDOW // 2
PER_DATASET = 100

_transformers: dict[int, pyproj.Transformer] = {}


def utm_epsg(lon: float, lat: float) -> int:
    """Return the UTM EPSG code for a lon/lat point."""
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def get_transformer(epsg: int) -> pyproj.Transformer:
    """Return a cached WGS84-to-UTM transformer for the given EPSG code."""
    if epsg not in _transformers:
        _transformers[epsg] = pyproj.Transformer.from_crs(
            "EPSG:4326", f"EPSG:{epsg}", always_xy=True
        )
    return _transformers[epsg]


def shift_years(d: date, years: int) -> date:
    """Shift a date by a whole number of years, handling Feb 29."""
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + years, day=28)


def make_entry(
    slug: str,
    idx: int | str,
    lon: float,
    lat: float,
    pre_change: str | None,
    time_range: tuple[str, str] | None = None,
    with_point: bool = True,
) -> dict:
    """Build one v2 annotation entry centered on the given lon/lat."""
    epsg = utm_epsg(lon, lat)
    tf = get_transformer(epsg)
    e, n = tf.transform(lon, lat)
    cx = round(e / 10)
    cy = round(n / -10)
    if time_range is None:
        if pre_change:
            d = date.fromisoformat(pre_change)
            time_range = (
                shift_years(d, -3).isoformat(),
                shift_years(d, 3).isoformat(),
            )
        else:
            time_range = ("2019-01-01", "2025-01-01")
    point: dict[str, float | str] = {"lon": lon, "lat": lat}
    if pre_change:
        point["pre_change"] = pre_change
    return {
        "projection": {
            "crs": f"EPSG:{epsg}",
            "x_resolution": 10.0,
            "y_resolution": -10.0,
        },
        "bounds": [cx - HALF, cy - HALF, cx + HALF, cy + HALF],
        "window_name": f"{slug}_{idx}",
        "group": "phase16",
        "positive_points": [point] if with_point else [],
        "negative_points": [],
        "time_range": list(time_range),
    }


def random_point_in_polygon(
    poly: shapely.Geometry, rng: random.Random
) -> shapely.Point:
    """Return a uniformly random point inside a polygon."""
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(1000):
        p = shapely.Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(p):
            return p
    return poly.representative_point()


def sample(items: list[T], k: int, rng: random.Random) -> list[T]:
    """Return a random sample of k items, or a copy if there are fewer than k."""
    if len(items) <= k:
        return list(items)
    return rng.sample(items, k)


# ---------------------------------------------------------------------------
# 1. IPIS DRC: distinct sites whose first-ever visit falls in 2019-2024; use
#    that first in-window visit's date and location.
def gen_ipis_drc() -> list[dict]:
    """Sample IPIS DRC sites whose first visit falls in 2019-2024."""
    import pandas as pd

    rng = random.Random(16001)
    d = pd.read_csv(f"{SP}/ipis_drc.csv", low_memory=False)
    d["visit_date"] = pd.to_datetime(d["visit_date"], errors="coerce")
    d = d.dropna(subset=["visit_date", "longitude", "latitude"])
    first = d.groupby("pcode")["visit_date"].min()
    new_sites = set(first[(first >= "2019-01-01") & (first <= "2024-12-31")].index)
    rows = (
        d[d["pcode"].isin(new_sites)]
        .sort_values("visit_date")
        .groupby("pcode")
        .first()
        .reset_index()
    )
    chosen = sample(list(rows.itertuples()), PER_DATASET, rng)
    return [
        make_entry(
            "ipis_drc",
            r.pcode,
            float(r.longitude),
            float(r.latitude),
            r.visit_date.date().isoformat(),
        )
        for r in chosen
    ]


# 2. IPIS CAR: same scheme (all visits are 2019-2021).
def gen_ipis_car() -> list[dict]:
    """Sample IPIS CAR sites using the earliest in-window visit (all 2019-2021)."""
    import pandas as pd

    rng = random.Random(16002)
    d = pd.read_csv(f"{SP}/ipis_car.csv", low_memory=False)
    d["visit_date"] = pd.to_datetime(d["visit_date"], errors="coerce")
    d = d.dropna(subset=["visit_date", "longitude", "latitude"])
    d = d[(d["visit_date"] >= "2019-01-01") & (d["visit_date"] <= "2024-12-31")]
    rows = d.sort_values("visit_date").groupby("pcode").first().reset_index()
    chosen = sample(list(rows.itertuples()), PER_DATASET, rng)
    return [
        make_entry(
            "ipis_car",
            r.pcode,
            float(r.longitude),
            float(r.latitude),
            r.visit_date.date().isoformat(),
        )
        for r in chosen
    ]


# 3. IPIS Zimbabwe (Runde district, all visited Feb-Mar 2019).
def gen_ipis_zwe() -> list[dict]:
    """Sample IPIS Zimbabwe (Runde district) sites visited in Feb-Mar 2019."""
    rng = random.Random(16003)
    d = json.load(open(f"{SP}/zwe_wb.json"))
    fts = [
        f for f in d["features"] if f.get("geometry") and f["properties"].get("today")
    ]
    chosen = sample(fts, PER_DATASET, rng)
    out = []
    for f in chosen:
        lon, lat = f["geometry"]["coordinates"][:2]
        visit = f["properties"]["today"].rstrip("Z")
        pcode = f["properties"].get("pcode") or f["properties"].get("id")
        out.append(make_entry("ipis_zwe", pcode, float(lon), float(lat), visit))
    return out


# 4. USGS Copperbelt: ASM-flagged 1 km cells -> rough location, no points.
def gen_usgs_copperbelt() -> list[dict]:
    """Sample USGS Copperbelt ASM 1 km cells as rough locations with no points."""
    import fiona

    rng = random.Random(16004)
    tf = pyproj.Transformer.from_crs("EPSG:32735", "EPSG:4326", always_xy=True)
    cells = []
    with fiona.open(
        f"{SP}/usgs_copperbelt/Mining_Extent_Commodities_Final_reconciled_2.shp"
    ) as src:
        for ft in src:
            p = ft["properties"]
            if p.get("Mining") == 1 and p.get("Scale") == "ASM":
                geom = shapely.geometry.shape(ft["geometry"])
                c = geom.centroid
                lon, lat = tf.transform(c.x, c.y)
                cells.append((lon, lat, str(p.get("Date") or "")))
    chosen = sample(cells, PER_DATASET, rng)
    out = []
    for i, (lon, lat, date_str) in enumerate(chosen):
        m = re.match(r"(\d{2})([A-Z]{3})(\d{4})$", date_str)
        if m:
            months = {
                "JAN": 1,
                "FEB": 2,
                "MAR": 3,
                "APR": 4,
                "MAY": 5,
                "JUN": 6,
                "JUL": 7,
                "AUG": 8,
                "SEP": 9,
                "OCT": 10,
                "NOV": 11,
                "DEC": 12,
            }
            d = date(int(m.group(3)), months[m.group(2)], int(m.group(1)))
            tr = (shift_years(d, -3).isoformat(), shift_years(d, 3).isoformat())
        else:  # "2019-2023" style range
            tr = ("2018-01-01", "2026-01-01")
        out.append(
            make_entry(
                "usgs_copperbelt", i, lon, lat, None, time_range=tr, with_point=False
            )
        )
    return out


# 5. LAMES Ghana ASM polygons with imagery date >= 2019.
def gen_lames() -> list[dict]:
    """Sample LAMES Ghana ASM polygons with imagery date in 2019-2024."""
    rng = random.Random(16005)
    d = json.load(open(f"{SP}/mineseg/annotations/Ghana_ASM.geojson"))
    polys = []
    for ft in d["features"]:
        desc = ft["properties"].get("description") or ""
        m = re.search(r"date[^0-9]*(\d{4}-\d{2}-\d{2})", desc)
        a = re.search(r"asm[^0-9]*(\d)", desc)
        if not m or not a or a.group(1) != "1":
            continue
        dt = m.group(1).strip()
        if not ("2019-01-01" <= dt <= "2024-12-31"):
            continue
        geom = shapely.geometry.shape(ft["geometry"])
        if geom.is_empty:
            continue
        polys.append((geom, dt, ft["properties"].get("Name", "")))
    chosen = sample(polys, PER_DATASET, rng)
    out = []
    for geom, dt, name in chosen:
        p = random_point_in_polygon(geom, rng)
        out.append(make_entry("lames", name.lower(), p.x, p.y, dt))
    return out


# 6. Africa Mining Watch: polygons, no dates -> positive point without
#    pre_change. Split proportionally: 74 West Africa / 26 Congo Basin.
def gen_amw() -> list[dict]:
    """Sample Africa Mining Watch polygons as undated positive points."""
    rng = random.Random(16006)
    out = []
    idx = 0
    for fname, slug_suffix, count in [
        ("WestAfrica_EI_2026-06-24-dissolved.geojson", "wa", 74),
        ("CongoBasin_EI_2026-06-17-dissolved.geojson", "cb", 26),
    ]:
        d = json.load(open(f"{SP}/amw/{fname}"))
        geoms = [
            shapely.geometry.shape(f["geometry"])
            for f in d["features"]
            if f.get("geometry")
        ]
        for geom in sample(geoms, count, rng):
            p = random_point_in_polygon(geom, rng)
            out.append(
                make_entry(
                    f"amw_{slug_suffix}",
                    idx,
                    p.x,
                    p.y,
                    None,
                    time_range=("2019-01-01", "2025-01-01"),
                )
            )
            idx += 1
    return out


# 7. SmallMinesDS: prefer pixels that changed 0->1 between the 2016 and 2022
#    masks (new mining); fall back to any 2022-positive pixel.
def gen_smallminesds() -> list[dict]:
    """Sample SmallMinesDS pixels that changed from non-mining (2016) to mining (2022)."""
    import glob

    import numpy as np
    import rasterio

    rng = random.Random(16007)
    mask22 = sorted(glob.glob(f"{SP}/smallminesds/SmallMinesDS/2022/MASK/*.tif"))
    change_patches = []
    pos_only_patches = []
    for f22 in mask22:
        f16 = f22.replace("/2022/", "/2016/").replace("_2022.tif", "_2016.tif")
        with rasterio.open(f22) as src:
            m22 = src.read(1)
        if not m22.any():
            continue
        try:
            with rasterio.open(f16) as src:
                m16 = src.read(1)
            change = (m22 == 1) & (m16 == 0)
        except rasterio.errors.RasterioIOError:
            change = np.zeros_like(m22, dtype=bool)
        if change.any():
            change_patches.append((f22, "change"))
        else:
            pos_only_patches.append((f22, "pos"))
    chosen = sample(change_patches, PER_DATASET, rng)
    if len(chosen) < PER_DATASET:
        chosen += sample(pos_only_patches, PER_DATASET - len(chosen), rng)
    out = []
    to_wgs84 = {}
    for i, (f22, kind) in enumerate(chosen):
        f16 = f22.replace("/2022/", "/2016/").replace("_2022.tif", "_2016.tif")
        with rasterio.open(f22) as src:
            m22 = src.read(1)
            transform = src.transform
            crs = src.crs
        if kind == "change":
            with rasterio.open(f16) as src:
                m16 = src.read(1)
            candidates = np.argwhere((m22 == 1) & (m16 == 0))
        else:
            candidates = np.argwhere(m22 == 1)
        row, col = candidates[rng.randrange(len(candidates))]
        x, y = rasterio.transform.xy(transform, int(row), int(col))
        key = str(crs)
        if key not in to_wgs84:
            to_wgs84[key] = pyproj.Transformer.from_crs(
                crs, "EPSG:4326", always_xy=True
            )
        lon, lat = to_wgs84[key].transform(x, y)
        # MASK_GH_0001_2022.tif -> gh_0001
        patch_id = re.sub(r"^MASK_|_2022\.tif$", "", f22.split("/")[-1]).lower()
        out.append(make_entry("small_mines_ds", patch_id, lon, lat, "2022-01-01"))
    print(
        f"  smallminesds: {len(change_patches)} change patches, "
        f"{len(pos_only_patches)} positive-only patches"
    )
    return out


# 8. Pasanisi eastern DRC: mask tiles in WGS84, no dates available -> positive
#    point without pre_change.
def gen_pasanisi() -> list[dict]:
    """Sample mining pixels from Pasanisi eastern DRC mask tiles."""
    import glob

    import numpy as np
    import rasterio

    rng = random.Random(16008)
    masks = sorted(glob.glob(f"{SP}/pasanisi/dataset/ps/masks/*.tif"))
    positive = []
    for f in masks:
        with rasterio.open(f) as src:
            m = src.read(1)
        if m.any():
            positive.append(f)
    chosen = sample(positive, PER_DATASET, rng)
    out = []
    for f in chosen:
        with rasterio.open(f) as src:
            m = src.read(1)
            transform = src.transform
        candidates = np.argwhere(m == 1)
        row, col = candidates[rng.randrange(len(candidates))]
        lon, lat = rasterio.transform.xy(transform, int(row), int(col))
        mask_id = f.split("/")[-1].replace("mask_", "").replace(".tif", "")
        out.append(
            make_entry(
                "pasanisi_drc",
                mask_id,
                lon,
                lat,
                None,
                time_range=("2019-01-01", "2025-01-01"),
            )
        )
    print(f"  pasanisi: {len(positive)} positive masks of {len(masks)}")
    return out


# 9. Dethier et al. river mining districts: centroid points only -> rough
#    location, no points. Filter to Africa, active into 2019+ (all are).
def gen_dethier() -> list[dict]:
    """Sample Dethier river-mining district centroids in Africa as rough locations."""
    import pandas as pd

    rng = random.Random(16009)
    meta = pd.read_csv(f"{SP}/dethier/imports/rm_site_metadata.csv")
    ceased = pd.to_numeric(meta["Mining ceased"], errors="coerce")
    af = meta[(meta["Continent"] == "Africa") & (ceased.isna() | (ceased >= 2019))]
    keep = set(af["ID_ref"])
    kml = open(
        f"{SP}/moesm3/2022-07-11321B-river_mining_global_sites_20230501.kml"
    ).read()
    sites = []
    for pm in re.findall(r"<Placemark>(.*?)</Placemark>", kml, re.S):
        name = re.search(r"<name>(.*?)</name>", pm)
        coords = re.search(r"<coordinates>([-0-9.,]+)", pm)
        if not name or not coords or name.group(1) not in keep:
            continue
        lon, lat = (float(v) for v in coords.group(1).split(",")[:2])
        sites.append((name.group(1), lon, lat))
    chosen = sample(sites, PER_DATASET, rng)
    return [
        make_entry(
            "dethier",
            ref.lower(),
            lon,
            lat,
            None,
            time_range=("2018-01-01", "2026-01-01"),
            with_point=False,
        )
        for ref, lon, lat in chosen
    ]


# 10. Cote d'Ivoire ASM tiles (2025 imagery): random mining pixel from the
#     mask of each sampled mine-positive tile.
def gen_ivc() -> list[dict]:
    """Sample mining pixels from Cote d'Ivoire ASM tiles (2025 imagery)."""
    import numpy as np
    import pandas as pd
    import rasterio

    rng = random.Random(16010)
    d = pd.read_csv(f"{SP}/ivc_split.csv")
    d = d[d["positive_pixels"] > 0]
    chosen = sample(list(d.itertuples()), PER_DATASET, rng)
    out = []
    for r in chosen:
        path = f"/vsizip/{SP}/ivc_tiles.zip/{r.mask_path}"
        with rasterio.open(path) as src:
            m = src.read(1)
            transform = src.transform
            crs = src.crs
        candidates = np.argwhere(m > 0)
        row, col = candidates[rng.randrange(len(candidates))]
        x, y = rasterio.transform.xy(transform, int(row), int(col))
        tf = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = tf.transform(x, y)
        tile_id = r.tile_id.lower().replace("_zone_ivc", "")
        out.append(make_entry("ivc", tile_id, lon, lat, "2025-01-01"))
    return out


GENERATORS = {
    "ipis_drc": gen_ipis_drc,
    "ipis_car": gen_ipis_car,
    "ipis_zwe": gen_ipis_zwe,
    "usgs_copperbelt": gen_usgs_copperbelt,
    "lames": gen_lames,
    "amw": gen_amw,
    "smallminesds": gen_smallminesds,
    "pasanisi": gen_pasanisi,
    "dethier": gen_dethier,
    "ivc": gen_ivc,
}


def main() -> None:
    """Generate partial_<dataset>.json files for the requested datasets."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing the downloaded source datasets (see README).",
    )
    parser.add_argument("datasets", nargs="*", default=list(GENERATORS))
    args = parser.parse_args()
    global SP
    SP = args.data_dir.rstrip("/")
    names = args.datasets or list(GENERATORS)
    for name in names:
        entries = GENERATORS[name]()
        with open(f"{SP}/partial_{name}.json", "w") as f:
            json.dump(entries, f, indent=2)
        print(f"{name}: {len(entries)} entries")


if __name__ == "__main__":
    main()
