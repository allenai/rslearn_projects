"""Region-targeted sampler for a second round of recent Landsat inference.

Round 1 (sample_recent_scenes.py) draws a uniform global sample over the marine ROI.
This sampler instead over-samples specific regions where a lot of detections /
errors were observed, to check the pipeline's behavior there. Regions are lon/lat
bounding boxes (edit REGIONS below); for each we list the WRS-2 path/rows whose
footprint intersects the box, pool every scene acquired in the last --days days, and
sample up to --per_region of them. The union is written to one CSV with a ``region``
column (dedup by scene_id; a scene is labelled by the first region it falls in).

Usage:
    python -m rslp.landsat_vessels.scripts.sample_region_scenes \
        --out /weka/dfive-default/yawenz/landsat/20260908_results_round2/sample_scenes.csv \
        --days 28 --per_region 30 --workers 16
"""

import argparse
import csv
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import shapely.geometry
from rslearn.data_sources.aws_landsat import LandsatOliTirs
from rslearn.data_sources.wrs2 import get_wrs2_polygons
from upath import UPath

# (lon_min, lat_min, lon_max, lat_max) boxes read off the observed-detection maps.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "gulf_of_aden_socotra": (45.0, 10.5, 56.0, 16.0),
    "somalia_coast": (46.5, 3.5, 52.0, 11.0),
    "n_australia_arafura": (128.0, -14.5, 138.5, -8.0),
    "celebes_n_sulawesi": (120.0, -1.5, 127.5, 6.0),
    "lesser_sunda_flores": (116.5, -10.5, 124.0, -6.5),
    "se_sulawesi_banda": (120.0, -6.5, 125.5, -3.0),
    "papua_cenderawasih": (131.5, -4.5, 138.5, 0.8),
    "hawaii": (-161.0, 18.0, -154.0, 23.0),
    "e_greenland": (-45.0, 59.0, -17.0, 72.0),
    "se_canada_lakes": (-95.0, 42.5, -74.0, 50.5),
}


def main() -> None:
    """Discover and sample recent scenes per target region."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--per_region", type=int, default=30)
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
    print(f"today (UTC)={now.date()}  window >= {cutoff.date()} ({args.days}d)")

    # WRS-2 path/rows per region (intersect the region box).
    boxes = {name: shapely.geometry.box(*bb) for name, bb in REGIONS.items()}
    region_of: dict[tuple[str, str], str] = {}
    centroids: dict[tuple[str, str], tuple[float, float]] = {}
    for polygon, path, row in get_wrs2_polygons(cache / "wrs2", buffer_degrees=0):
        for name, box in boxes.items():
            if polygon.intersects(box):
                key = (path, row)
                if key not in region_of:  # first region wins
                    region_of[key] = name
                    c = polygon.centroid
                    centroids[key] = (round(c.x, 4), round(c.y, 4))
                break
    by_region: dict[str, list[tuple[str, str]]] = {}
    for key, name in region_of.items():
        by_region.setdefault(name, []).append(key)
    print("path/rows per region:")
    for name in REGIONS:
        print(f"  {name:24s} {len(by_region.get(name, []))}")

    ds = LandsatOliTirs(metadata_cache_dir=str(cache / "products"))

    def list_pathrow(key: tuple[str, str]) -> list[dict]:
        path, row = key
        lon, lat = centroids[key]
        found = []
        for year in {cutoff.year, now.year}:
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
                            "region": region_of[key],
                            "blob_path": it.blob_path,
                        }
                    )
        return found

    all_keys = list(region_of.keys())
    print(f"listing {len(all_keys)} path/rows ({args.workers} workers)...")
    region_scenes: dict[str, dict[str, dict]] = {name: {} for name in REGIONS}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for found in ex.map(list_pathrow, all_keys):
            done += 1
            for s in found:
                region_scenes[s["region"]][s["scene_id"]] = s
            if done % 50 == 0:
                print(f"  listed {done}/{len(all_keys)}", flush=True)

    rng = random.Random(args.seed)
    selected: dict[str, dict] = {}
    print("\nin-window scenes per region (pool -> sampled):")
    for name in REGIONS:
        pool = list(region_scenes[name].values())
        take = (
            pool if len(pool) <= args.per_region else rng.sample(pool, args.per_region)
        )
        for s in take:
            selected.setdefault(s["scene_id"], s)
        print(f"  {name:24s} {len(pool):4d} -> {len(take)}")

    rows = sorted(selected.values(), key=lambda s: (s["region"], s["datetime"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fields = [
        "scene_id",
        "path",
        "row",
        "datetime",
        "cloud_cover",
        "lon",
        "lat",
        "region",
        "blob_path",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} unique scenes -> {args.out}")


if __name__ == "__main__":
    main()
