"""Sample Landsat T1/T2 scenes over the Skylight marine regions for annotation.

Stratified scene sampler for building a new (negatives-focused) classifier dataset.
Path/rows are drawn from the WRS-2 grid where it intersects the marine-regions ROI,
stratified toward the environments that generate detector false positives:

  * coastal  -- scene footprint touches land (coastal clutter, islands, reefs)
  * storm    -- 40-60 deg abs latitude (whitecap belts)
  * ice      -- >= 55 deg abs latitude, non-polar-darkness months (melt ice + edge)
  * glint    -- <= 25 deg abs latitude (sun glint)
  * cloud    -- anywhere, scene cloud cover >= 40% (cloud false positives)
  * open     -- anywhere in the ROI (unbiased background)

A path/row can qualify for several strata; each is used for at most one. Because the
Landsat acquisition plan does not routinely image open ocean, each stratum draws from
a shuffled candidate list and keeps querying until its quota is filled or candidates
run out. Scene lookups go through rslearn's LandsatOliTirs data source (the same
usgs-landsat bucket the predict pipeline materializes from; it only holds T1/T2, so
Real-Time products can never be selected). Listings are cached in --cache_dir, so
re-runs are cheap.

Usage:
    python -m rslp.landsat_vessels.scripts.sample_scenes \
        --marine_regions /weka/dfive-default/yawenz/landsat/World_Marine_Regions.geojson \
        --out ./scene_sample.csv --n_scenes 200

    # Inspect the stratified path/row sample without touching AWS:
    python -m rslp.landsat_vessels.scripts.sample_scenes ... --dry_run
"""

import argparse
import csv
import json
import random
from datetime import datetime, timedelta, timezone

import shapely
import shapely.geometry
import shapely.strtree
from rslearn.data_sources.wrs2 import get_wrs2_polygons
from upath import UPath

# Fraction of the total scene budget per stratum. The mix is deliberately tilted
# toward hard-negative environments; "open" keeps a slice of unbiased background.
STRATUM_FRACTIONS = {
    "coastal": 0.20,
    "storm": 0.20,
    "ice": 0.20,
    "glint": 0.10,
    "cloud": 0.15,
    "open": 0.15,
}

# Months to use for the ice stratum: everything except the polar-darkness months
# (no usable sun elevation for Landsat at high latitude). This deliberately keeps
# the melt season (e.g. July in the Arctic) — production Skylight feedback shows
# broken melt ice is the biggest vessel false-positive source — plus the ice edge
# in spring and fall freeze-up.
ICE_MONTHS_NORTH = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
ICE_MONTHS_SOUTH = [8, 9, 10, 11, 12, 1, 2, 3, 4]

# Minimum scene cloud cover for the "cloud" stratum, which exists because clouds
# are a major source of vessel false positives in production.
CLOUD_STRATUM_MIN_COVER = 40.0

# No cloud-cover filtering: Skylight runs every scene that arrives, so matching the
# production distribution means sampling uniformly among acquisitions regardless of
# cloud (cloudy scenes are themselves a false-positive source).

# Candidate path/rows to shuffle per stratum, as a multiple of its quota. Most
# open-ocean candidates have no acquisitions, so over-generate generously.
CANDIDATE_MULTIPLE = 8


def load_marine_regions(
    path: str, include_lakes: bool
) -> list[tuple[shapely.Geometry, str]]:
    """Load the marine-regions ROI as (geometry, region name) pairs.

    Args:
        path: path to the World_Marine_Regions.geojson file.
        include_lakes: whether to keep lake/reservoir features (the file contains a
            handful of African lakes alongside the ocean/sea regions).

    Returns:
        list of (shapely geometry, region name).
    """
    with open(path) as f:
        collection = json.load(f)
    regions = []
    for feat in collection["features"]:
        featurecla = feat["properties"].get("featurecla")
        if featurecla is not None and not include_lakes:
            continue
        name = feat["properties"].get("name") or "unnamed"
        # Some polygons in this dataset are topologically invalid, which breaks
        # intersection operations later; repair them up front.
        geom = shapely.make_valid(shapely.geometry.shape(feat["geometry"]))
        regions.append((geom, name))
    return regions


def load_land_geometries() -> list[shapely.Geometry]:
    """Load Natural Earth 50m land polygons (via cartopy) for coastal detection."""
    from cartopy.io import shapereader

    shp_path = shapereader.natural_earth(
        resolution="50m", category="physical", name="land"
    )
    return [
        shapely.geometry.shape(rec.geometry)
        for rec in shapereader.Reader(shp_path).records()
    ]


def hemisphere_ice_months(lat: float) -> list[int]:
    """Months to sample for the ice stratum at this latitude."""
    return ICE_MONTHS_NORTH if lat >= 0 else ICE_MONTHS_SOUTH


def build_pathrow_candidates(
    wrs2_cache_dir: UPath,
    regions: list[tuple[shapely.Geometry, str]],
    land_geoms: list[shapely.Geometry],
) -> list[dict]:
    """Compute the candidate path/rows and their stratum memberships.

    Args:
        wrs2_cache_dir: cache directory for the WRS-2 shapefile download.
        regions: (geometry, name) pairs from load_marine_regions.
        land_geoms: land polygons for coastal detection.

    Returns:
        list of dicts with path, row, lon, lat, region, and per-stratum booleans.
    """
    # buffer_degrees=0 keeps footprints tight so strata reflect the actual scene.
    wrs2 = get_wrs2_polygons(wrs2_cache_dir, buffer_degrees=0)

    region_tree = shapely.strtree.STRtree([geom for geom, _ in regions])
    land_tree = shapely.strtree.STRtree(land_geoms)

    candidates = []
    for polygon, path, row in wrs2:
        hits = region_tree.query(polygon, predicate="intersects")
        if len(hits) == 0:
            continue
        centroid = polygon.centroid
        lat = centroid.y
        # Landsat only acquires usable ocean imagery with reasonable sun elevation;
        # beyond ~82 deg there is effectively nothing.
        if abs(lat) > 82:
            continue
        # Name the region by the largest intersection.
        best_region = max(
            (regions[i] for i in hits),
            key=lambda rg: polygon.intersection(rg[0]).area,
        )[1]
        is_coastal = len(land_tree.query(polygon, predicate="intersects")) > 0
        candidates.append(
            {
                "path": path,
                "row": row,
                "lon": centroid.x,
                "lat": lat,
                "region": best_region,
                "coastal": is_coastal,
                "storm": 40 <= abs(lat) <= 60,
                "ice": abs(lat) >= 55,
                "glint": abs(lat) <= 25,
                "cloud": True,
                "open": True,
            }
        )
    return candidates


def pick_scene(
    data_source: "object",
    candidate: dict,
    stratum: str,
    start: datetime,
    end: datetime,
    rng: random.Random,
) -> dict | None:
    """Pick one T1/T2 scene for a candidate path/row, or None if none exists.

    A random one-month window in [start, end] is chosen (restricted to shoulder-season
    months for the ice stratum); within the window a scene is picked uniformly at
    random, with no cloud filtering, to match the production input distribution.
    """
    if stratum == "ice":
        months = hemisphere_ice_months(candidate["lat"])
    else:
        months = list(range(1, 13))

    # Enumerate the (year, month) options in range with an allowed month, then try a
    # few of them in random order: a single month can easily have no acquisition.
    options = []
    cur = datetime(start.year, start.month, 1, tzinfo=timezone.utc)
    while cur <= end:
        if cur.month in months:
            options.append(cur)
        cur = (cur + timedelta(days=32)).replace(day=1)
    rng.shuffle(options)

    for month_start in options[:6]:
        month_end = (month_start + timedelta(days=32)).replace(day=1)
        items = data_source._read_products(  # type: ignore[attr-defined]
            {(month_start.year, candidate["path"], candidate["row"])}
        )
        in_window = [
            item
            for item in items
            if month_start <= item.geometry.time_range[0] < month_end
        ]
        if stratum == "cloud":
            in_window = [
                item
                for item in in_window
                if item.cloud_cover is not None
                and item.cloud_cover >= CLOUD_STRATUM_MIN_COVER
            ]
        if not in_window:
            continue

        item = rng.choice(in_window)

        return {
            "scene_id": item.name,
            "path": candidate["path"],
            "row": candidate["row"],
            "datetime": item.geometry.time_range[0].isoformat(),
            "cloud_cover": item.cloud_cover if item.cloud_cover is not None else -1,
            "stratum": stratum,
            "region": candidate["region"],
            "lon": round(candidate["lon"], 4),
            "lat": round(candidate["lat"], 4),
            "blob_path": item.blob_path,
        }
    return None


def main() -> None:
    """Run the stratified scene sampler."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--marine_regions", required=True)
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--n_scenes", type=int, default=200)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument(
        "--end",
        default="2026-05-31",
        help="end of date range; keep ~2 months before today so T1/T2 exists",
    )
    parser.add_argument(
        "--cache_dir",
        default="/weka/dfive-default/yawenz/landsat/scene_sampling_cache",
        help="cache for the WRS-2 shapefile and per-path/row scene listings",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_per_pathrow", type=int, default=1)
    parser.add_argument("--include_lakes", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="stop after stratified path/row sampling (no AWS access)",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    cache_dir = UPath(args.cache_dir)
    (cache_dir / "wrs2").mkdir(parents=True, exist_ok=True)
    (cache_dir / "products").mkdir(parents=True, exist_ok=True)

    print("Loading marine regions...")
    regions = load_marine_regions(args.marine_regions, args.include_lakes)
    print(f"  {len(regions)} region polygons")
    print("Loading land polygons (Natural Earth 50m)...")
    land_geoms = load_land_geometries()
    print("Building WRS-2 candidates (first run downloads the WRS-2 shapefile)...")
    candidates = build_pathrow_candidates(cache_dir / "wrs2", regions, land_geoms)
    print(f"  {len(candidates)} path/rows intersect the ROI")
    for stratum in STRATUM_FRACTIONS:
        n = sum(1 for c in candidates if c[stratum])
        print(f"    {stratum:8s}: {n} candidates")

    # Assign quotas and shuffle candidate lists per stratum.
    quotas = {s: round(frac * args.n_scenes) for s, frac in STRATUM_FRACTIONS.items()}
    print(f"quotas: {quotas}")

    if args.dry_run:
        print("--dry_run set; not querying AWS for scenes.")
        return

    from rslearn.data_sources.aws_landsat import LandsatOliTirs

    data_source = LandsatOliTirs(metadata_cache_dir=str(cache_dir / "products"))

    selected: list[dict] = []
    used_pathrows: dict[tuple[str, str], int] = {}
    used_scene_ids: set[str] = set()

    for stratum, quota in quotas.items():
        pool = [c for c in candidates if c[stratum]]
        rng.shuffle(pool)
        pool = pool[: quota * CANDIDATE_MULTIPLE]
        got = 0
        for candidate in pool:
            if got >= quota:
                break
            key = (candidate["path"], candidate["row"])
            if used_pathrows.get(key, 0) >= args.max_per_pathrow:
                continue
            scene = pick_scene(data_source, candidate, stratum, start, end, rng)
            if scene is None or scene["scene_id"] in used_scene_ids:
                continue
            selected.append(scene)
            used_scene_ids.add(scene["scene_id"])
            used_pathrows[key] = used_pathrows.get(key, 0) + 1
            got += 1
            print(
                f"[{stratum} {got}/{quota}] {scene['scene_id']} "
                f"cloud={scene['cloud_cover']:.0f}% region={scene['region']}"
            )
        if got < quota:
            print(f"WARNING: stratum '{stratum}' filled only {got}/{quota}")

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
        writer.writerows(selected)
    print(f"\nWrote {len(selected)} scenes to {args.out}")

    # Summary by stratum and region so silent skew is visible.
    for field in ["stratum", "region"]:
        counts: dict[str, int] = {}
        for scene in selected:
            counts[scene[field]] = counts.get(scene[field], 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])
        print(f"by {field}: {top[:12]}")


if __name__ == "__main__":
    main()
