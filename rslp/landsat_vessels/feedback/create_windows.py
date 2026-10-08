"""Stage 2 - turn a normalised feedback CSV into rslearn windows.

Each feedback row becomes one window in a dated group (e.g. ``feedback_20260911``) of the
classifier dataset, carrying a ``label`` layer (``incorrect`` for a BAD / false positive,
``correct`` for a GOOD). Training then reads the group exactly like the existing
``feedback_20260325`` and ``round1_20260803`` groups.

Two ways a window's imagery is located, chosen per row:

- **pinned** (``scene_id`` present): the exact Landsat product is looked up in the AWS
  data source and written into the window, substituting the definitive T1/T2 product when
  the reported RT product has since been deleted from ``s3://usgs-landsat`` (same logic as
  ``scripts/create_round1_windows.py``). This is exact and reproducible.
- **matched** (no ``scene_id``): only a time range is written, and ``rslearn dataset
  prepare`` matches the scene by time and space later. Less precise, used only when the
  feedback carries no product id.

After windows are written, materialise the group so the ``landsat`` layer is populated::

    # pinned windows already know their item, so prepare is skipped for them:
    rslearn dataset prepare     --root <root> --group <group> --workers 32   # matched rows only
    rslearn dataset materialize --root <root> --group <group> --workers 32

``--materialize`` runs those for you.

Usage:
    python -m rslp.landsat_vessels.feedback.create_windows \
        --csv /weka/dfive-default/yawenz/landsat/feedback_20260911/feedback_20260911.csv \
        --group feedback_20260911 \
        --materialize
"""

import argparse
import csv as _csv
import hashlib
import re
import subprocess  # nosec B404 - shells out only to the rslearn dataset CLI
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import WindowLayerData
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_crs
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.landsat_vessels.feedback import config

PRODUCT_RE = re.compile(r"^(LC0[89])_(L1\w{2})_(\d{6})_(\d{8})_(\d{8})_02_(T1|T2|RT)$")
# Substitution preference when the exact product a detection ran on is missing: try RT
# first (production runs on the RT tier, so RT is the imagery the deployed model actually
# saw), then T1, then T2 of the same acquisition. This is the opposite of
# scripts/create_round1_windows.py, which preferred T1/T2 for long-term reproducibility;
# feedback windows instead want to reproduce what production saw.
TIER_RANK = {"RT": 0, "T1": 1, "T2": 2}


def parse_product(product_id: str) -> tuple[str, str, str, str, str, str]:
    """Split a Landsat product id into its six components."""
    match = PRODUCT_RE.match(product_id)
    if not match:
        raise ValueError(f"cannot parse product id {product_id}")
    return match.groups()  # type: ignore[return-value]


def resolve_items(
    make_data_source: Callable[[], Any], scene_ids: set[str], workers: int
) -> tuple[dict[str, Any], dict[str, str]]:
    """One AWS data-source item per scene, substituting deleted RT products.

    Returns (item by scene id, replacement product id by scene id). Mirrors
    ``scripts/create_round1_windows.resolve_items``: listings run on a thread pool with a
    per-thread data source (boto3 clients are not thread-safe); the on-disk metadata cache
    is shared so a second run costs no S3 calls.
    """
    by_pathrow: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for scene_id in scene_ids:
        _, _, pathrow, acquired, _, _ = parse_product(scene_id)
        by_pathrow[(int(acquired[:4]), pathrow[:3], pathrow[3:])].add(scene_id)

    local = threading.local()

    def list_products(key: tuple[int, str, str]) -> dict[str, Any]:
        """List AWS products for one (year, path, row) key, by product id."""
        if not hasattr(local, "data_source"):
            local.data_source = make_data_source()
        return {item.name: item for item in local.data_source._read_products({key})}

    keys = sorted(by_pathrow)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        listings = list(
            tqdm.tqdm(
                pool.map(list_products, keys),
                total=len(keys),
                desc="listing products (year/path/row)",
            )
        )

    items: dict[str, Any] = {}
    substitutions: dict[str, str] = {}
    for key, available in zip(keys, listings):
        for scene_id in sorted(by_pathrow[key]):
            if scene_id in available:
                items[scene_id] = available[scene_id]
                continue
            platform, _, pathrow, acquired, _, _ = parse_product(scene_id)
            candidates = []
            for name in available:
                c_platform, _, c_pathrow, c_acquired, processed, tier = parse_product(
                    name
                )
                if (c_platform, c_pathrow, c_acquired) == (platform, pathrow, acquired):
                    candidates.append((TIER_RANK[tier], -int(processed), name))
            if not candidates:
                continue
            replacement = sorted(candidates)[0][2]
            items[scene_id] = available[replacement]
            substitutions[scene_id] = replacement
    return items, substitutions


def window_bounds(
    lon: float, lat: float, window_size: int
) -> tuple[Projection, tuple[int, int, int, int]]:
    """Projection and pixel bounds of a window centred on a detection."""
    projection = Projection(
        get_utm_ups_crs(lon, lat), config.WINDOW_RESOLUTION, -config.WINDOW_RESOLUTION
    )
    geometry = STGeometry(
        WGS84_PROJECTION, shapely.Point(lon, lat), None
    ).to_projection(projection)
    col, row = int(geometry.shp.x), int(geometry.shp.y)
    half = window_size // 2
    return projection, (col - half, row - half, col + half, row + half)


def assign_split(event_id: str, val_fraction: float) -> str:
    """Deterministically send a fraction of events to val, the rest to train.

    Feedback windows are hard negatives meant to enter *training*; evaluation stays on
    the frozen round1 test set and the feedback_20260325 val set. A small val holdout is
    available (val_fraction) but off by default.
    """
    if val_fraction <= 0:
        return "train"
    bucket = int(hashlib.sha256(event_id.encode()).hexdigest()[:8], 16) % 1000
    return "val" if bucket < val_fraction * 1000 else "train"


def parse_event_time(value: str) -> datetime:
    """Parse an ISO event time, assuming UTC when no tz is given."""
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def main() -> None:
    """Create feedback windows and (optionally) materialise them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", required=True, help="normalised feedback CSV (pull.py)"
    )
    default_group = f"feedback_{datetime.now(timezone.utc):%Y%m%d}"
    parser.add_argument("--group", default=default_group, help="window group name")
    parser.add_argument(
        "--dataset-root", default=str(config.DATASET_ROOT), help="classifier dataset"
    )
    parser.add_argument("--window-size", type=int, default=config.WINDOW_SIZE)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="fraction of events held out to the val split (default 0 = all train)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only first N rows")
    parser.add_argument(
        "--workers", type=int, default=32, help="threads for S3 listings / materialize"
    )
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="run rslearn dataset prepare/materialize on the group after writing windows",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="resolve items and print the plan, write nothing",
    )
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        rows = list(_csv.DictReader(f))
    if args.limit:
        rows = rows[: args.limit]
    rows = [
        r
        for r in rows
        if str(r.get("lat", "")).strip() and str(r.get("lon", "")).strip()
    ]
    pinned = [r for r in rows if r.get("scene_id")]
    matched = [r for r in rows if not r.get("scene_id")]
    print(
        f"{len(rows)} rows: {len(pinned)} pinned (scene_id), {len(matched)} matched "
        f"(time-range) -> group {args.group}"
    )

    dataset = Dataset(UPath(args.dataset_root))
    layer_config = dataset.layers[config.LANDSAT_LAYER]

    items: dict[str, Any] = {}
    substitutions: dict[str, str] = {}
    if pinned:
        items, substitutions = resolve_items(
            lambda: layer_config.instantiate_data_source(dataset.path),
            {r["scene_id"] for r in pinned},
            args.workers,
        )
        unresolved = {r["scene_id"] for r in pinned} - set(items)
        print(
            f"resolved {len(items)} scenes "
            f"({len(substitutions)} RT->definitive), {len(unresolved)} unresolved"
        )
        for scene_id in sorted(unresolved):
            print(f"  UNRESOLVED {scene_id}")

    vector_format = GeojsonVectorFormat()
    index: list[dict] = []
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    written = 0

    for row in tqdm.tqdm(rows, desc="creating windows"):
        event_id = row["event_id"]
        lon, lat = float(row["lon"]), float(row["lat"])
        label = row.get("label") or config.FEEDBACK_VALUE_TO_LABEL.get(
            (row.get("value") or "").upper(), "incorrect"
        )
        split = assign_split(event_id, args.val_fraction)
        counts[split][label] += 1

        projection, bounds = window_bounds(lon, lat, args.window_size)
        try:
            event_time = parse_event_time(row["event_time"])
        except (ValueError, KeyError):
            print(
                f"  skipping {event_id}: unparseable event_time {row.get('event_time')!r}"
            )
            continue
        time_range = (
            event_time - timedelta(minutes=config.TIME_BUFFER_MINUTES),
            event_time + timedelta(minutes=config.TIME_BUFFER_MINUTES),
        )

        scene_id = row.get("scene_id") or ""
        item = items.get(scene_id) if scene_id else None
        if scene_id and item is None:
            # pinned but unresolved (product gone, no substitute): skip rather than
            # silently fall back to a fuzzy time match.
            continue

        record = {
            "window": event_id,
            "group": args.group,
            "split": split,
            "label": label,
            "value": row.get("value", ""),
            "username": row.get("username", ""),
            "scene_id": scene_id,
            "substituted_product": substitutions.get(scene_id),
            "item_name": getattr(item, "name", None),
            "lon": lon,
            "lat": lat,
            "ts": event_time.isoformat(),
            "crs": str(projection.crs),
            "bounds": list(bounds),
        }
        index.append(record)
        if args.dry_run:
            continue

        window = Window(
            storage=dataset.storage,
            group=args.group,
            name=event_id,
            projection=projection,
            bounds=bounds,
            time_range=time_range,
            options={"split": split, "weight": 1},
            data_factory=dataset.window_data_storage_factory,
        )
        window.save()
        if item is not None:
            window.save_layer_datas(
                {
                    config.LANDSAT_LAYER: WindowLayerData(
                        layer_name=config.LANDSAT_LAYER,
                        serialized_item_groups=[[item.serialize()]],
                        materialized=False,
                    )
                }
            )

        properties = {
            "label": label,
            "feedback_value": row.get("value", ""),
            "annotator": row.get("username", "unknown"),
            "source": "skylight_in_app_feedback",
            "model_version": config.DEPLOYED_MODEL_VERSION,
            "annotation_round": args.group,
            "event_id": event_id,
            "lon": lon,
            "lat": lat,
            "ts": event_time.isoformat(),
            "labeled_at": datetime.now(timezone.utc).isoformat(),
        }
        feature = Feature(window.get_geometry(), properties)
        with window.data.open_layer_writer(config.LABEL_LAYER) as writer:
            writer.write_vector(vector_format, [feature])
        window.mark_layer_completed(config.LABEL_LAYER)
        written += 1

    print(
        f"\nsplit x label: {{ {', '.join(f'{s}: {dict(c)}' for s, c in sorted(counts.items()))} }}"
    )

    if args.dry_run:
        print("dry run: nothing written")
        return

    index_path = UPath(args.csv).parent / f"{args.group}_index.json"
    import json

    with index_path.open("w") as f:
        json.dump(index, f)
    print(f"wrote {written} windows to {dataset.path / 'windows' / args.group}")
    print(f"wrote provenance index to {index_path}")

    if args.materialize:
        _materialize(args.dataset_root, args.group, bool(matched), args.workers)
    else:
        print("\nnext: materialise the imagery:")
        if matched:
            print(
                f"  rslearn dataset prepare     --root {args.dataset_root} "
                f"--group {args.group} --workers {args.workers}"
            )
        print(
            f"  rslearn dataset materialize --root {args.dataset_root} "
            f"--group {args.group} --workers {args.workers}"
        )


def _materialize(root: str, group: str, run_prepare: bool, workers: int) -> None:
    """Run the rslearn dataset CLI to populate the landsat layer for the group."""
    steps = []
    if run_prepare:
        # Only matched (unpinned) windows need prepare; pinned windows already carry
        # their item. prepare is safe on the whole group -- it no-ops windows that
        # already have their layer data.
        steps.append(["prepare"])
    steps.append(["materialize"])
    for step in steps:
        cmd = [
            "rslearn",
            "dataset",
            step[0],
            "--root",
            root,
            "--group",
            group,
            "--workers",
            str(workers),
        ]
        print(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd, check=True)  # nosec B603 - cmd built internally, no shell


if __name__ == "__main__":
    main()
