"""Build the pool of agriculture events to run the monocrop classifier on.

Events come from a run of the forest loss driver labeling pipeline in
../2026_09_25_amazon_forest_loss_labeling (select_events.py extract with --min_area_ha 5
for 2023-01 through 2025-10, then select, studio_jobs.py with the utm model and
merge_predictions.py), which gives an events_with_new_model_predictions.geojson. Several
such files can be passed as --source NAME=path.

We keep events in [--start_month, --end_month) predicted as agriculture, drop events
whose polygon is within --min_distance_m of any labeled monocrop annotation
(export_labeled_annotations.py), and sample events uniformly at random per
(country, year), with a quota of --per_country events per country split across years
in proportion to their number of months. Each stratum has its own random generator.

Each output feature keeps the event properties and gets:
- event_uid: "<source>:<event_id>", unique across sources.
- source: the source NAME.

Output: {out_dir}/monocrop_candidates.geojson and {out_dir}/monocrop_candidates_counts.csv.

Example:
    python build_monocrop_candidates.py --out_dir /path/to/out \
        --source 2023_2025=/path/to/events_2023_2025/events_with_new_model_predictions.geojson
"""

import argparse
import csv
import json
import random
from collections import Counter, defaultdict

import pyproj
import shapely
import shapely.geometry
from shapely.ops import transform
from upath import UPath

TARGET_CATEGORY = "agriculture"
# Degrees of margin for the coarse spatial query before the exact metric distance.
# 1 km is at most ~0.0097 degrees of longitude in the study area (|lat| < 25).
QUERY_MARGIN_DEG = 0.02


def local_metric_transformer(lon: float, lat: float) -> pyproj.Transformer:
    """Get a WGS84 -> azimuthal equidistant transformer centered at (lon, lat)."""
    crs = pyproj.CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m"
    )
    return pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)


class ExclusionIndex:
    """Find geometries within a distance (meters) of a set of exclusion polygons."""

    def __init__(self, geometries: list[shapely.Geometry]) -> None:
        """Index the exclusion geometries (WGS84)."""
        self.geometries = geometries
        self.tree = shapely.STRtree(geometries)

    def is_near(self, geometry: shapely.Geometry, distance_m: float) -> bool:
        """Whether geometry is within distance_m meters of any exclusion geometry."""
        minx, miny, maxx, maxy = geometry.bounds
        query = shapely.box(
            minx - QUERY_MARGIN_DEG,
            miny - QUERY_MARGIN_DEG,
            maxx + QUERY_MARGIN_DEG,
            maxy + QUERY_MARGIN_DEG,
        )
        hits = self.tree.query(query)
        if len(hits) == 0:
            return False
        center = geometry.centroid
        project = local_metric_transformer(center.x, center.y).transform
        geometry_m = transform(project, geometry)
        for idx in hits:
            if (
                geometry_m.distance(transform(project, self.geometries[idx]))
                < distance_m
            ):
                return True
        return False


def get_months(start: str, end: str) -> list[str]:
    """Get the months (YYYY-MM) in [start, end)."""
    year, month = int(start[:4]), int(start[5:7])
    months = []
    while f"{year:04d}-{month:02d}" < end:
        months.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def main(cli_args: argparse.Namespace) -> None:
    """Write monocrop_candidates.geojson."""
    out_dir = UPath(cli_args.out_dir)
    with (out_dir / "labeled_monocrop_annotations.geojson").open() as f:
        exclusion = ExclusionIndex(
            [
                shapely.geometry.shape(feat["geometry"])
                for feat in json.load(f)["features"]
            ]
        )

    stats: Counter[tuple[str, str, str]] = Counter()
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for source_arg in cli_args.source:
        source, path = source_arg.split("=", 1)
        with UPath(path).open() as f:
            features = json.load(f)["features"]
        for feat in features:
            props = feat["properties"]
            country = props["country"]
            month = props["month"]
            if not (cli_args.start_month <= month < cli_args.end_month):
                stats[(source, country, "out_of_range")] += 1
                continue
            stats[(source, country, "in_range")] += 1
            if props["category"] != TARGET_CATEGORY:
                continue
            stats[(source, country, "agriculture")] += 1
            geometry = shapely.geometry.shape(feat["geometry"])
            if exclusion.is_near(geometry, cli_args.min_distance_m):
                stats[(source, country, "near_labeled")] += 1
                continue
            stats[(source, country, "eligible")] += 1
            props = dict(props)
            props["event_uid"] = f"{source}:{props['event_id']}"
            props["source"] = source
            strata[(country, month[:4])].append(
                {"type": "Feature", "properties": props, "geometry": feat["geometry"]}
            )

    months_per_year = Counter(
        month[:4] for month in get_months(cli_args.start_month, cli_args.end_month)
    )
    total_months = sum(months_per_year.values())
    selected = []
    for country, year in sorted(strata):
        quota = round(cli_args.per_country * months_per_year[year] / total_months)
        pool = sorted(
            strata[(country, year)], key=lambda f: f["properties"]["event_uid"]
        )
        rng = random.Random(f"{cli_args.seed}:{country}:{year}")
        sample = rng.sample(pool, min(quota, len(pool)))
        for feat in sample:
            stats[(feat["properties"]["source"], country, "sampled")] += 1
        selected.extend(sample)
    selected.sort(key=lambda f: f["properties"]["event_uid"])
    print(f"sampled {len(selected)} events")

    with (out_dir / "monocrop_candidates.geojson").open("w") as f:
        json.dump({"type": "FeatureCollection", "features": selected}, f)
    columns = [
        "out_of_range",
        "in_range",
        "agriculture",
        "near_labeled",
        "eligible",
        "sampled",
    ]
    with (out_dir / "monocrop_candidates_counts.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "country"] + columns)
        for source, country in sorted({(s, c) for s, c, _ in stats}):
            row = [stats[(source, country, col)] for col in columns]
            writer.writerow([source, country] + row)
            print(source, country, dict(zip(columns, row)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="NAME=path to an events_with_new_model_predictions.geojson",
    )
    parser.add_argument("--start_month", default="2023-01")
    parser.add_argument("--end_month", default="2025-11", help="exclusive")
    parser.add_argument("--min_distance_m", type=float, default=1000)
    parser.add_argument("--per_country", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
