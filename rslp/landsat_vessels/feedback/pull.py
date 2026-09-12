"""Stage 1 - pull in-app feedback from Skylight and reduce it to trainable rows.

This generalises the round-0 ``enrich_feedback.py`` (production only) into a single,
environment-aware step that produces exactly the CSV ``create_windows.py`` consumes:

1. read the raw in-app feedback CSV exported from the Skylight admin panel,
2. derive ``tile_id`` and, when it is a Landsat product id, ``scene_id``,
3. enrich each row with ``event_time`` / ``lat`` / ``lon`` from the Skylight GraphQL API
   (skippable with --tile-only when the export already carries coordinates),
4. keep only trusted users and GOOD/BAD labels (drop UNSURE/empty),
5. write ``feedback_<date>.csv`` with a normalised schema.

The raw export is a manual download: open the admin in-app-feedback tab for the chosen
environment (integration by default -- that is where v1.0.0 is deployed) and save the
CSV. The GraphQL auth token is grabbed from browser DevTools (Network tab -> any GraphQL
request -> Authorization header) and passed with --token.

Date and user filtering keep the batch tight: --since (with --date-field acquisition, the
default) drops detections produced before v1.0.0 shipped, and --username keeps only the
chosen reviewer(s). Both are applied *before* enrichment, so API calls are spent only on
rows that survive.

Usage:
    python -m rslp.landsat_vessels.feedback.pull \
        --input /weka/dfive-default/yawenz/landsat/feedback_20260911/in_app_feedback.csv \
        --environment integration \
        --token "Bearer eyJ..." \
        --source gcs \
        --username yawenz@allenai.org \
        --since 20260911 --date-field submission \
        --keep bad_only \
        --model-version landsat_vessels_v1.0.0 \
        --out /weka/dfive-default/yawenz/landsat/feedback_20260911/feedback_20260911.csv
"""

import argparse
import csv
import re
import sys
import time
from datetime import date, datetime, timezone

import requests

from rslp.landsat_vessels.feedback import config

# A Landsat Collection-2 Level-1 product id, e.g. LC09_L1TP_109027_20260911_20260911_02_T1.
# Group 4 is the acquisition date (YYYYMMDD); group 5 is the processing date.
PRODUCT_RE = re.compile(r"^(LC0[89])_(L1\w{2})_(\d{6})_(\d{8})_(\d{8})_02_(T1|T2|RT)$")

EVENT_QUERY = """
query GetEvent($eventId: ID!) {
  event(eventId: $eventId) {
    event_id
    start {
      time
      point {
        lat
        lon
      }
    }
  }
}
"""

# Output schema consumed by create_windows.py.
OUT_FIELDS = [
    "event_id",
    "username",
    "value",
    "label",
    "tile_id",
    "scene_id",
    "submitted",
    "event_time",
    "lat",
    "lon",
    "score",
    "model_version",
]


def _find_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    """First fieldname (case-insensitive) matching any candidate, else None."""
    lower = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    return None


def derive_tile_id(event_id: str) -> str:
    """Skylight tile id: the event id without its trailing detection index."""
    return event_id.rsplit("_", 1)[0]


def derive_scene_id(tile_id: str, event_id: str) -> str:
    """The Landsat product id for this feedback, or "" if none is embedded.

    Skylight ids embed the Landsat product id for Landsat detections; when present it
    lets create_windows pin the exact scene (and substitute a deleted RT product) rather
    than re-matching by time and space. When absent, create_windows falls back to a
    time-range match.
    """
    for candidate in (tile_id, event_id):
        match = PRODUCT_RE.search(candidate or "")
        if match:
            return match.group(0)
    return ""


def acquisition_date(scene_id: str) -> date | None:
    """The Landsat acquisition date embedded in a product id, or None."""
    match = PRODUCT_RE.search(scene_id or "")
    if not match:
        return None
    return datetime.strptime(match.group(4), "%Y%m%d").date()


def parse_date(value: str) -> date | None:
    """Parse a YYYYMMDD or ISO date/datetime into a date, or None."""
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(
                value[:10] if fmt == "%Y-%m-%d" else value[:8], fmt
            ).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def row_date(row: dict, field: str) -> date | None:
    """The date used for --since/--until filtering, per --date-field.

    - acquisition (default): the Landsat acquisition date parsed from the product id.
      This is what tells us a detection was produced by the *deployed* model, so it is
      the right filter for "only feedback on detections made after v1.0.0 shipped".
    - submission: when the user filed the feedback (raw CSV ``timestamp``).
    - event_time: the detection event time from enrichment.
    """
    if field == "acquisition":
        return acquisition_date(row.get("scene_id") or row.get("tile_id", ""))
    if field == "submission":
        return parse_date(row.get("_submitted", ""))
    if field == "event_time":
        return parse_date(row.get("event_time", ""))
    return None


def is_trusted(username: str) -> bool:
    """Whether a feedback author is on the trusted allowlist."""
    username = (username or "").strip().lower()
    if not username:
        return False
    if username in {u.lower() for u in config.TRUSTED_EXACT}:
        return True
    return any(username.endswith(domain.lower()) for domain in config.TRUSTED_DOMAINS)


def fetch_event_details(
    event_id: str, session: requests.Session, graphql_url: str
) -> dict | None:
    """event_time / lat / lon for one event via GraphQL, or None on any failure."""
    try:
        resp = session.post(
            graphql_url,
            json={"query": EVENT_QUERY, "variables": {"eventId": event_id}},
            timeout=15,
        )
        resp.raise_for_status()
        event = resp.json().get("data", {}).get("event")
        if event and event.get("start"):
            start = event["start"]
            point = start.get("point") or {}
            return {
                "event_time": start.get("time", ""),
                "lat": point.get("lat", ""),
                "lon": point.get("lon", ""),
            }
    except Exception as exc:  # noqa: BLE001 - report and continue, one bad id is not fatal
        print(f"  [error] {event_id}: {exc}")
    return None


def event_index(event_id: str) -> int | None:
    """The trailing detection index of an event id, e.g. ..._RT_2 -> 2."""
    tail = event_id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def crop_index(record: dict) -> int | None:
    """The detection's crop index, parsed from ``crop_fnames.rgb`` (``<n>_rgb.png``)."""
    rgb = (
        ((record.get("crop_fnames") or {}).get("rgb")) or record.get("crop_fname") or ""
    )
    base = rgb.rsplit("/", 1)[-1]
    head = base.split("_", 1)[0]
    return int(head) if head.isdigit() else None


def _gsutil_cat(url: str) -> str | None:
    """Return the contents of a GCS object via gsutil, or None if it is absent."""
    import subprocess  # nosec B404 - fixed gsutil argv, no shell

    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed gsutil argv, no shell
            ["gsutil", "cat", url], capture_output=True, text=True, timeout=120
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [gsutil error] {url}: {exc}")
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def enrich_from_gcs(rows: list[dict], detections_base: str) -> tuple[int, set[str]]:
    """Fill lat/lon/event_time/score from the sat-service detection JSONs on GCS.

    Groups rows by scene, reads each scene's JSON once, and matches each feedback
    ``event_id``'s trailing index to the detection whose crop index equals it. Returns
    (rows enriched, model versions seen). This is the offline, authoritative alternative
    to the GraphQL ``event`` query -- the same records Skylight served.
    """
    import json
    from collections import defaultdict

    by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        scene = row.get("scene_id")
        if scene:
            by_scene[scene].append(row)
    print(f"enriching {len(rows)} rows from {len(by_scene)} scene JSONs on GCS...")

    enriched = 0
    versions: set[str] = set()
    missing_scenes: list[str] = []
    for i, (scene, grp) in enumerate(sorted(by_scene.items()), 1):
        if i == 1 or i % 25 == 0:
            print(f"  {i}/{len(by_scene)} scenes")
        match = PRODUCT_RE.search(scene)
        if not match:
            continue
        acq = match.group(4)  # YYYYMMDD
        # Try the acquisition-date folder first, then the processing-date folder.
        contents = None
        for date_str in (acq, match.group(5)):
            url = f"{detections_base}/{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}/{scene}.json"
            contents = _gsutil_cat(url)
            if contents is not None:
                break
        if contents is None:
            missing_scenes.append(scene)
            continue
        try:
            data = json.loads(contents)
        except json.JSONDecodeError:
            missing_scenes.append(scene)
            continue
        if data.get("rslearn_model_version"):
            versions.add(str(data["rslearn_model_version"]))
        by_idx: dict[int, dict] = {}
        for pos, rec in enumerate(data.get("detections", [])):
            idx = crop_index(rec)
            by_idx[idx if idx is not None else pos] = rec
        acq_iso = f"{acq[:4]}-{acq[4:6]}-{acq[6:8]}T00:00:00+00:00"
        for row in grp:
            idx = event_index(row["event_id"])
            rec = by_idx.get(idx) if idx is not None else None
            if rec is None:
                continue
            row["lat"] = rec.get("latitude", "")
            row["lon"] = rec.get("longitude", "")
            row["event_time"] = rec.get("ts") or acq_iso
            row["score"] = rec.get("score", "")
            row["model_version"] = data.get("rslearn_model_version", "")
            enriched += 1
    if missing_scenes:
        print(
            f"  {len(missing_scenes)} scene JSONs not found on GCS "
            f"(e.g. {missing_scenes[:2]})"
        )
    if versions:
        print(f"  model version(s) in detections: {sorted(versions)}")
    return enriched, versions


def main() -> None:
    """Pull, enrich, filter and normalise Skylight in-app feedback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="raw in_app_feedback.csv exported from Skylight"
    )
    parser.add_argument("--out", required=True, help="normalised output CSV path")
    parser.add_argument(
        "--environment",
        choices=sorted(config.SKYLIGHT_ENVIRONMENTS),
        default=config.DEFAULT_ENVIRONMENT,
        help="which Skylight deployment the feedback came from",
    )
    parser.add_argument(
        "--graphql-url",
        default=None,
        help="override the GraphQL endpoint (default: the environment's)",
    )
    parser.add_argument("--token", help='GraphQL auth header, e.g. "Bearer eyJ..."')
    parser.add_argument(
        "--cookie", help="GraphQL session cookie (alternative to token)"
    )
    parser.add_argument(
        "--tile-only",
        action="store_true",
        help="skip enrichment (input already has event_time/lat/lon)",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "gcs", "api", "tile"],
        default="auto",
        help="where to get lat/lon: gcs = sat-service detection JSONs (authoritative, "
        "offline); api = Skylight GraphQL (needs --token, IP-restricted); tile = already "
        "in the input. auto: tile if coords present, else api if --token, else gcs.",
    )
    parser.add_argument(
        "--gcs-detections",
        default=None,
        help="base gs:// path of the sat-service detections (default: the environment's)",
    )
    parser.add_argument(
        "--keep",
        choices=["good_bad", "bad_only", "good_only"],
        default="good_bad",
        help="which feedback values to keep (bad_only = false positives, the FP-focus)",
    )
    parser.add_argument(
        "--no-trust-filter",
        action="store_true",
        help="keep feedback from every user, not only the trusted allowlist",
    )
    parser.add_argument(
        "--username",
        action="append",
        default=None,
        metavar="EMAIL",
        help="keep only these exact users (repeatable); overrides the trust allowlist. "
        "e.g. --username yawenz@allenai.org",
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYYMMDD",
        help="keep feedback on/after this date. Use the v1.0.0 deploy date to exclude "
        "detections the old model produced (e.g. --since 20260911).",
    )
    parser.add_argument(
        "--until",
        default=None,
        metavar="YYYYMMDD",
        help="keep feedback on/before this date",
    )
    parser.add_argument(
        "--date-field",
        choices=["acquisition", "submission", "event_time"],
        default="acquisition",
        help="which date --since/--until apply to (default: acquisition = when the "
        "detection was produced, i.e. by which deployed model)",
    )
    parser.add_argument(
        "--model-version",
        action="append",
        default=None,
        metavar="VERSION",
        help="keep only detections produced by these model version(s) (repeatable; "
        "substring match on the detection JSON's rslearn_model_version). Only the model "
        "under evaluation should feed its own feedback loop, "
        "e.g. --model-version landsat_vessels_v1.0.0. Requires --source gcs (the "
        "version comes from the detection JSON).",
    )
    args = parser.parse_args()

    model_versions = (
        [v.strip() for v in args.model_version] if args.model_version else None
    )

    since = parse_date(args.since) if args.since else None
    until = parse_date(args.until) if args.until else None
    if args.since and since is None:
        sys.exit(f"could not parse --since {args.since!r} (use YYYYMMDD)")
    if args.until and until is None:
        sys.exit(f"could not parse --until {args.until!r} (use YYYYMMDD)")
    username_allowlist = (
        {u.strip().lower() for u in args.username} if args.username else None
    )

    env = config.SKYLIGHT_ENVIRONMENTS[args.environment]
    graphql_url = args.graphql_url or env["graphql_url"]

    with open(args.input, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    print(f"read {len(rows)} rows from {args.input}")
    print(f"environment: {args.environment} (admin: {env['admin_feedback_url']})")

    event_col = _find_column(fieldnames, ["event_id", "eventid", "event"])
    value_col = _find_column(fieldnames, ["value", "label", "rating"])
    user_col = _find_column(fieldnames, ["username", "user", "email", "user_email"])
    # Submission time (when the feedback was filed) is distinct from the detection's
    # event_time (added by enrichment); keep them apart so date filtering is unambiguous.
    submitted_col = _find_column(
        fieldnames, ["timestamp", "created_at", "submitted_at"]
    )
    time_col = _find_column(fieldnames, ["event_time"])
    lat_col = _find_column(fieldnames, ["lat", "latitude"])
    lon_col = _find_column(fieldnames, ["lon", "lng", "longitude"])
    if not event_col or not value_col:
        sys.exit(f"input must have an event id and a value column; found {fieldnames}")

    # tile_id / scene_id / submission time are all derivable without any API call.
    for row in rows:
        event_id = (row.get(event_col) or "").strip()
        row["event_id"] = event_id
        row["tile_id"] = derive_tile_id(event_id)
        row["scene_id"] = derive_scene_id(row["tile_id"], event_id)
        row["_submitted"] = row.get(submitted_col, "") if submitted_col else ""

    # --- Cheap pre-filters (no API): value, user, and date. Only survivors get enriched,
    # so the coordinate lookups are spent only on rows we will actually keep. ---
    keep_values = {
        "good_bad": {"GOOD", "BAD"},
        "bad_only": {"BAD"},
        "good_only": {"GOOD"},
    }[args.keep]

    dropped = {"value": 0, "user": 0, "date": 0, "no_coords": 0}
    prefiltered: list[dict] = []
    for row in rows:
        value = (row.get(value_col) or "").strip().upper()
        username = (row.get(user_col, "") if user_col else "").strip()
        if value not in keep_values:
            dropped["value"] += 1
            continue
        if username_allowlist is not None:
            if username.lower() not in username_allowlist:
                dropped["user"] += 1
                continue
        elif not args.no_trust_filter and not is_trusted(username):
            dropped["user"] += 1
            continue
        if since or until:
            d = row_date(row, args.date_field)
            if d is None or (since and d < since) or (until and d > until):
                dropped["date"] += 1
                continue
        row["value"] = value
        row["username"] = username
        prefiltered.append(row)

    date_desc = ""
    if since or until:
        date_desc = (
            f", {args.date_field} in [{args.since or '-inf'}, {args.until or '+inf'}]"
        )
    print(
        f"pre-filter: {len(prefiltered)}/{len(rows)} rows pass value={args.keep}, "
        f"user, date (dropped value={dropped['value']}, user={dropped['user']}, "
        f"date={dropped['date']}){date_desc}"
    )

    # Resolve the enrichment source (see --source help).
    have_coords = bool(time_col and lat_col and lon_col)
    source = "tile" if args.tile_only else args.source
    if source == "auto":
        source = (
            "tile" if have_coords else ("api" if (args.token or args.cookie) else "gcs")
        )
    print(f"enrichment source: {source}")

    if source == "tile":
        if have_coords:
            print("using event_time/lat/lon already present in the export")
        for row in prefiltered:
            row["event_time"] = row.get(time_col, "") if time_col else ""
            row["lat"] = row.get(lat_col, "") if lat_col else ""
            row["lon"] = row.get(lon_col, "") if lon_col else ""
        if not have_coords:
            print("WARNING: tile source but no coord columns -> rows have no lat/lon")
    elif source == "gcs":
        base = args.gcs_detections or config.SKYLIGHT_DETECTION_BUCKETS.get(
            args.environment
        )
        if not base:
            sys.exit(
                f"no detections bucket configured for {args.environment}; "
                "pass --gcs-detections"
            )
        detections_base = f"{base}/{config.SENSOR_DETECTIONS_SUBPATH}"
        print(f"reading detections from {detections_base}")
        n_enriched, _ = enrich_from_gcs(prefiltered, detections_base)
        print(f"enriched {n_enriched}/{len(prefiltered)} rows from GCS")
    else:  # api
        if not args.token and not args.cookie:
            sys.exit("--source api needs --token or --cookie")
        session = requests.Session()
        if args.token:
            session.headers["Authorization"] = args.token
        if args.cookie:
            session.headers["Cookie"] = args.cookie
        session.headers["Content-Type"] = "application/json"

        unique_ids = sorted({row["event_id"] for row in prefiltered if row["event_id"]})
        print(f"fetching event details for {len(unique_ids)} unique events...")
        cache: dict[str, dict] = {}
        for i, eid in enumerate(unique_ids, 1):
            if i == 1 or i % 25 == 0:
                print(f"  {i}/{len(unique_ids)}")
            details = fetch_event_details(eid, session, graphql_url)
            if details:
                cache[eid] = details
            time.sleep(0.2)
        print(f"enriched {len(cache)}/{len(unique_ids)} events")
        for row in prefiltered:
            details = cache.get(row["event_id"], {})
            row["event_time"] = details.get("event_time", "")
            row["lat"] = details.get("lat", "")
            row["lon"] = details.get("lon", "")

    if model_versions and source not in ("gcs", "api"):
        print(
            f"WARNING: --model-version needs enrichment that reports model_version "
            f"(source=gcs); source is {source!r}, so no rows will match."
        )

    # Post-filter: only rows that now have coordinates (and, if requested, the right
    # model version) can become windows.
    dropped["model_version"] = 0
    kept: list[dict] = []
    for row in prefiltered:
        if not str(row.get("lat", "")).strip() or not str(row.get("lon", "")).strip():
            dropped["no_coords"] += 1
            continue
        if model_versions is not None:
            mv = str(row.get("model_version", ""))
            if not any(want in mv for want in model_versions):
                dropped["model_version"] += 1
                continue
        kept.append(
            {
                "event_id": row["event_id"],
                "username": row["username"],
                "value": row["value"],
                "label": config.FEEDBACK_VALUE_TO_LABEL[row["value"]],
                "tile_id": row["tile_id"],
                "scene_id": row["scene_id"],
                "submitted": row.get("_submitted", ""),
                "event_time": row.get("event_time", ""),
                "lat": row.get("lat", ""),
                "lon": row.get("lon", ""),
                "score": row.get("score", ""),
                "model_version": row.get("model_version", ""),
            }
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(kept)

    n_bad = sum(1 for r in kept if r["value"] == "BAD")
    n_good = sum(1 for r in kept if r["value"] == "GOOD")
    n_scene = sum(1 for r in kept if r["scene_id"])
    print(
        f"\nwrote {len(kept)} rows to {args.out} "
        f"({n_good} GOOD/correct, {n_bad} BAD/incorrect; {n_scene} with a pinned scene_id)"
    )
    print(f"dropped: {dropped}")
    print(f"generated at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
