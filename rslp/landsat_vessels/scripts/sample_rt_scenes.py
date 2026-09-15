"""Sample currently-available RT (Real-Time tier) scenes over the marine ROI.

Companion to sample_scenes.py for Real-Time products. RT is a transient tier:
products processed with predicted ephemeris sit in the usgs-landsat bucket only
until USGS reprocesses them to T1/T2 (days to ~3 weeks), at which point the RT
files are deleted — and most scenes now go direct to T1 the same day. So unlike
the T1/T2 sampler there is no 2024-2026 archive to draw from; the only RT scenes
that exist are from the trailing couple of weeks. This script therefore inverts
the approach: enumerate every RT product currently live over the ROI (one
delimiter listing per candidate path/row for the current year), then fill the
same strata quotas from that pool. Re-run it in later weeks to top up the RT
sample as new RT products appear.

Path/rows already used by an earlier sample CSV are excluded so the one-scene-
per-path/row split-hygiene property holds across the union of samples.

Because RT products expire, run the detector over the output CSV promptly.

Usage:
    export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
    python -m rslp.landsat_vessels.scripts.sample_rt_scenes \
        --marine_regions /weka/dfive-default/yawenz/landsat/World_Marine_Regions.geojson \
        --exclude_csv /weka/dfive-default/yawenz/landsat/scene_sample_v1.csv \
                      /weka/dfive-default/yawenz/landsat/scene_sample_rt_v1.csv \
        --out /weka/dfive-default/yawenz/landsat/scene_sample_rt_v2.csv \
        --n_scenes 100
"""

import argparse
import concurrent.futures
import csv
import io
import json
import random
from datetime import datetime, timezone

import boto3
from upath import UPath

from rslp.landsat_vessels.scripts.sample_scenes import (
    CLOUD_STRATUM_MIN_COVER,
    STRATUM_FRACTIONS,
    build_pathrow_candidates,
    hemisphere_ice_months,
    load_land_geometries,
    load_marine_regions,
)

BUCKET = "usgs-landsat"
BUCKET_PREFIX = "collection02/level-1/standard/oli-tirs"


def list_rt_products(s3_client: "object", year: int, path: str, row: str) -> list[str]:
    """List RT product IDs under one year/path/row prefix (delimiter listing only)."""
    prefix = f"{BUCKET_PREFIX}/{year}/{path}/{row}/"
    ids = []
    paginator = s3_client.get_paginator("list_objects_v2")  # type: ignore[attr-defined]
    for page in paginator.paginate(
        Bucket=BUCKET, Prefix=prefix, Delimiter="/", RequestPayer="requester"
    ):
        for cp in page.get("CommonPrefixes", []):
            product_id = cp["Prefix"].rstrip("/").split("/")[-1]
            if product_id.endswith("_RT"):
                ids.append(product_id)
    return ids


def fetch_stac(
    s3_client: "object", year: int, path: str, row: str, product_id: str
) -> dict:
    """Fetch a product's _stac.json (cloud cover + acquisition datetime)."""
    key = f"{BUCKET_PREFIX}/{year}/{path}/{row}/{product_id}/{product_id}_stac.json"
    buf = io.BytesIO()
    s3_client.download_fileobj(  # type: ignore[attr-defined]
        BUCKET, key, buf, ExtraArgs={"RequestPayer": "requester"}
    )
    buf.seek(0)
    stac = json.load(buf)
    props = stac["properties"]
    cloud = props.get("eo:cloud_cover", props.get("landsat:cloud_cover_land", -1))
    return {
        "scene_id": product_id,
        "datetime": props["datetime"],
        "cloud_cover": cloud if cloud is not None else -1,
        "blob_path": key.rsplit("stac.json", 1)[0],
    }


def main() -> None:
    """Enumerate live RT products over the ROI and fill strata quotas."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--marine_regions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_scenes", type=int, default=100)
    parser.add_argument(
        "--exclude_csv",
        nargs="+",
        default=None,
        help="earlier sample CSV(s) whose path/rows must not be reused; when "
        "topping up, pass every prior sample (T1/T2 and RT) so the one-scene-"
        "per-path/row property holds across the union",
    )
    parser.add_argument(
        "--cache_dir",
        default="/weka/dfive-default/yawenz/landsat/scene_sampling_cache",
    )
    parser.add_argument("--year", type=int, default=None, help="default: current year")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    year = args.year or datetime.now(timezone.utc).year

    excluded_pathrows: set[tuple[str, str]] = set()
    for csv_path in args.exclude_csv or []:
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                excluded_pathrows.add((r["path"], r["row"]))
    if excluded_pathrows:
        print(
            f"excluding {len(excluded_pathrows)} path/rows from "
            f"{len(args.exclude_csv)} sample CSV(s)"
        )

    print("Loading marine regions + land, building WRS-2 candidates...")
    regions = load_marine_regions(args.marine_regions, include_lakes=False)
    land_geoms = load_land_geometries()
    candidates = build_pathrow_candidates(
        UPath(args.cache_dir) / "wrs2", regions, land_geoms
    )
    candidates = [
        c for c in candidates if (c["path"], c["row"]) not in excluded_pathrows
    ]
    print(f"  {len(candidates)} candidate path/rows after exclusions")

    # Enumerate every live RT product over the ROI (one listing per path/row).
    s3 = boto3.client("s3")

    def scan(c: dict) -> list[tuple[dict, str]]:
        return [(c, pid) for pid in list_rt_products(s3, year, c["path"], c["row"])]

    rt_hits: list[tuple[dict, str]] = []
    with concurrent.futures.ThreadPoolExecutor(args.threads) as ex:
        for i, res in enumerate(ex.map(scan, candidates), 1):
            rt_hits.extend(res)
            if i % 1000 == 0:
                print(
                    f"  scanned {i}/{len(candidates)} path/rows, {len(rt_hits)} RT so far"
                )
    print(f"RT products live over ROI: {len(rt_hits)}")

    # Pull stac metadata for each RT hit (cloud cover, timestamp, blob path).
    def enrich(hit: tuple[dict, str]) -> dict:
        c, pid = hit
        meta = fetch_stac(s3, year, c["path"], c["row"], pid)
        meta.update(
            {
                "path": c["path"],
                "row": c["row"],
                "region": c["region"],
                "lon": round(c["lon"], 4),
                "lat": round(c["lat"], 4),
                "_cand": c,
            }
        )
        return meta

    with concurrent.futures.ThreadPoolExecutor(args.threads) as ex:
        pool = list(ex.map(enrich, rt_hits))

    # Fill the same strata quotas from the live pool. A scene qualifies for a
    # stratum by the same rules as sample_scenes.py; each scene and each
    # path/row is used at most once.
    quotas = {s: round(f * args.n_scenes) for s, f in STRATUM_FRACTIONS.items()}
    print(f"quotas: {quotas}")

    def qualifies(scene: dict, stratum: str) -> bool:
        c = scene["_cand"]
        if not c[stratum]:
            return False
        if stratum == "ice":
            month = int(scene["datetime"][5:7])
            if month not in hemisphere_ice_months(c["lat"]):
                return False
        if stratum == "cloud":
            if float(scene["cloud_cover"]) < CLOUD_STRATUM_MIN_COVER:
                return False
        return True

    rng.shuffle(pool)
    selected: list[dict] = []
    used_pathrows: set[tuple[str, str]] = set()
    used_ids: set[str] = set()
    for stratum, quota in quotas.items():
        got = 0
        for scene in pool:
            if got >= quota:
                break
            key = (scene["path"], scene["row"])
            if scene["scene_id"] in used_ids or key in used_pathrows:
                continue
            if not qualifies(scene, stratum):
                continue
            scene = dict(scene, stratum=stratum)
            scene.pop("_cand")
            selected.append(scene)
            used_ids.add(scene["scene_id"])
            used_pathrows.add(key)
            got += 1
        print(f"stratum {stratum}: filled {got}/{quota}")

    fieldnames = [
        "scene_id",
        "path",
        "row",
        "datetime",
        "cloud_cover",
        "stratum",
        "region",
        "lon",
        "lat",
        "blob_path",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: s[k] for k in fieldnames} for s in selected)
    print(f"\nWrote {len(selected)} RT scenes to {args.out}")


if __name__ == "__main__":
    main()
