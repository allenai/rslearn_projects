"""Sample recent Landsat T1/T2 scenes over the Skylight marine ROI for inference.

Discovers scenes acquired in the last ``--days`` days somewhere in the marine-regions
ROI and writes a random sample of ``--n`` of them to a CSV. Because the ROI covers
~21.7k WRS-2 path/rows (all the world's oceans) and each per-path/row listing is an
S3 call, we do NOT enumerate the whole ROI: we shuffle the ROI path/rows, list a
random subset (``--pool_pathrows``) in parallel, pool every in-window scene found, and
sample ``--n`` uniformly from that pool -- a representative global sample.

Usage:
    python -m rslp.landsat_vessels.scripts.sample_recent_scenes \
        --marine_regions /weka/dfive-default/yawenz/landsat/World_Marine_Regions.geojson \
        --out /weka/dfive-default/yawenz/landsat/20260908_results/sample_scenes.csv \
        --days 28 --n 300 --pool_pathrows 1500 --workers 16
"""

import argparse
import csv
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import shapely
import shapely.geometry
import shapely.strtree
from rslearn.data_sources.aws_landsat import LandsatOliTirs
from rslearn.data_sources.wrs2 import get_wrs2_polygons
from upath import UPath


def roi_pathrows(
    marine_regions: str, wrs2_cache: UPath
) -> list[tuple[str, str, float, float]]:
    """WRS-2 (path, row) pairs whose footprint intersects the marine ROI."""
    with open(marine_regions) as f:
        coll = json.load(f)
    regions = [
        shapely.make_valid(shapely.geometry.shape(feat["geometry"]))
        for feat in coll["features"]
    ]
    tree = shapely.strtree.STRtree(regions)
    out = []
    for polygon, path, row in get_wrs2_polygons(wrs2_cache, buffer_degrees=0):
        if len(tree.query(polygon, predicate="intersects")) == 0:
            continue
        if abs(polygon.centroid.y) > 82:
            continue
        out.append(
            (path, row, round(polygon.centroid.x, 4), round(polygon.centroid.y, 4))
        )
    return out


def main() -> None:
    """Discover and sample recent ROI scenes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--marine_regions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--pool_pathrows", type=int, default=1500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cache_dir",
        default="/weka/dfive-default/yawenz/landsat/scene_sampling_cache",
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=args.days)
    cache = UPath(args.cache_dir)
    (cache / "wrs2").mkdir(parents=True, exist_ok=True)
    (cache / "products").mkdir(parents=True, exist_ok=True)

    print(f"today (UTC)={now.date()}  window >= {cutoff.date()} ({args.days}d)")
    print("Building ROI path/rows...")
    pathrows = roi_pathrows(args.marine_regions, cache / "wrs2")
    print(f"  {len(pathrows)} ROI path/rows")

    rng = random.Random(args.seed)
    rng.shuffle(pathrows)
    pool = pathrows[: args.pool_pathrows]
    print(
        f"Listing {len(pool)} random path/rows for {now.year} ({args.workers} workers)..."
    )

    ds = LandsatOliTirs(metadata_cache_dir=str(cache / "products"))

    def list_pathrow(pr: tuple) -> list[dict]:
        path, row, lon, lat = pr
        years = {cutoff.year, now.year}
        found = []
        for year in years:
            try:
                items = list(ds._read_products({(year, path, row)}))
            except Exception:  # nosec B112 - best-effort S3 listing, skip transient failures
                continue
            for it in items:
                ts = it.geometry.time_range[0]
                if ts >= cutoff:
                    found.append(
                        {
                            "scene_id": it.name,
                            "path": path,
                            "row": row,
                            "datetime": ts.isoformat(),
                            "cloud_cover": it.cloud_cover
                            if it.cloud_cover is not None
                            else -1,
                            "lon": lon,
                            "lat": lat,
                            "blob_path": it.blob_path,
                        }
                    )
        return found

    scenes: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for found in ex.map(list_pathrow, pool):
            done += 1
            for s in found:
                scenes[s["scene_id"]] = s
            if done % 100 == 0:
                print(
                    f"  listed {done}/{len(pool)} path/rows, pool={len(scenes)} scenes",
                    flush=True,
                )

    pool_scenes = list(scenes.values())
    print(
        f"discovered {len(pool_scenes)} in-window scenes across the sampled path/rows"
    )
    if len(pool_scenes) > args.n:
        pool_scenes = rng.sample(pool_scenes, args.n)
    pool_scenes.sort(key=lambda s: s["datetime"], reverse=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = [
        "scene_id",
        "path",
        "row",
        "datetime",
        "cloud_cover",
        "lon",
        "lat",
        "blob_path",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pool_scenes)
    print(f"wrote {len(pool_scenes)} scenes -> {args.out}")


if __name__ == "__main__":
    main()
