"""Distance-to-coast enrichment via the Skylight public API.

Wraps Skylight's `getNearestCoastline` GraphQL query (Ai2 Lighthouse backend,
https://api.skylight.earth/graphql, no auth required) — the same dataset the
Skylight sat service uses in `filter_by_distance_to_coast`, where Landsat
detections are kept only if over water AND >= 10 m from the coastline.

Used here to enrich round-1 detection pools with `distance_to_coast_m` and
`land_cover_class` columns, so selection/annotation can (a) flag detections on
land or hugging the shoreline as near-certain false positives and (b) measure
what stronger thresholds than production's 10 m would remove.

CLI: enrich a pool CSV in place (adds the two columns):
    python -m rslp.landsat_vessels.scripts.coast_distance \
        --csv /weka/dfive-default/yawenz/landsat/round1_annotation_pool.csv
"""

import argparse
import csv
import json
import time
import urllib.error
import urllib.request

API_URL = "https://api.skylight.earth/graphql"
WATER_LAND_COVER_CLASS = "PermanentWaterBody"
LANDSAT_PRODUCTION_THRESHOLD_M = 10.0
BATCH = 250

_QUERY = """
query GetNearestCoastline($input: GetNearestCoastlineInput!) {
  getNearestCoastline(input: $input) {
    records {
      coordinate { lat lon }
      distanceToCoastMeters
      landCoverClass
    }
  }
}
"""


def lookup(
    points: list[tuple[float, float]],
    token: str | None = None,
    retries: int = 3,
) -> list[dict]:
    """Batch-lookup distance-to-coast; returns records in input order."""
    out: list[dict] = []
    for i in range(0, len(points), BATCH):
        chunk = points[i : i + BATCH]
        body = json.dumps(
            {
                "query": _QUERY,
                "variables": {
                    "input": {
                        "coordinates": [{"lat": lat, "lon": lon} for lat, lon in chunk]
                    }
                },
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        for attempt in range(retries):
            try:
                req = urllib.request.Request(
                    API_URL, data=body, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=60) as resp:  # nosec B310 - fixed https Overpass endpoint
                    payload = json.loads(resp.read().decode())
                if payload.get("errors"):
                    raise RuntimeError(f"GraphQL errors: {payload['errors']}")
                out.extend(payload["data"]["getNearestCoastline"]["records"])
                break
            except (urllib.error.URLError, RuntimeError):
                if attempt == retries - 1:
                    raise
                time.sleep(2**attempt)
    return out


def main() -> None:
    """Enrich a pool/manifest CSV with distance-to-coast columns."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))
    points = [(float(r["latitude"]), float(r["longitude"])) for r in rows]
    print(f"looking up {len(points)} points in batches of {BATCH}...")
    records = lookup(points, token=args.token)
    assert len(records) == len(rows)

    n_land = n_shore = 0
    for r, rec in zip(rows, records):
        r["distance_to_coast_m"] = rec["distanceToCoastMeters"]
        r["land_cover_class"] = rec["landCoverClass"]
        if rec["landCoverClass"] != WATER_LAND_COVER_CLASS:
            n_land += 1
        elif rec["distanceToCoastMeters"] < LANDSAT_PRODUCTION_THRESHOLD_M:
            n_shore += 1

    fieldnames = list(rows[0].keys())
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"enriched {args.csv}: {n_land} not-over-water, "
        f"{n_shore} over water but < {LANDSAT_PRODUCTION_THRESHOLD_M:g} m from coast"
    )


if __name__ == "__main__":
    main()
