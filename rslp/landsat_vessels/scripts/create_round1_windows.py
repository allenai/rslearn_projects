"""Create rslearn windows for the round-1 annotation pool.

One window per detection in ``round1_annotation_pool_v2.geojson``, on the same 15 m
grid the classifier trains on (see ``windows/feedback_20260325``, 64 px @ 15 m) but
512 px across instead of 64. The wider window serves three purposes at once:

- the annotation crop view is the centre 128 x 128 px (1.92 km), pan-sharpened RGB;
- the annotation zoom-out view is the full 512 px extent (7.68 km);
- training reads the same window with a deterministic centre ``Crop`` back to 64 px
  (offset 224) or 128 px (offset 192), so no imagery has to be re-acquired to change
  the training window size.

Items are written directly rather than via ``rslearn dataset prepare``: each pool row
names the exact scene its detection came from, and prepare would instead re-match by
time/space and could pick an overlapping neighbour scene. For the 500 RT rows, whose
products USGS has since deleted from s3://usgs-landsat, we fall back to the definitive
T1/T2 product of the same acquisition (same platform, path/row and acquisition date) —
the same substitution ``spectral/extract_spectra.py`` makes, recorded per window in the
index so it is never silently invisible.

Usage:
    export AWS_ACCESS_KEY_ID=$(beaker secret read AWS_ACCESS_KEY_ID -w ai2/earth-systems)
    export AWS_SECRET_ACCESS_KEY=$(beaker secret read AWS_SECRET_ACCESS_KEY -w ai2/earth-systems)
    python create_round1_windows.py [--limit N] [--group NAME] [--dry_run]
"""

import argparse
import hashlib
import json
import re
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any

import shapely
import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import WindowLayerData
from rslearn.utils.geometry import Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_crs
from upath import UPath

POOL_PATH = UPath(
    "/weka/dfive-default/yawenz/landsat/round1_annotation_pool_v2.geojson"
)
DATASET_ROOT = UPath(
    "/weka/dfive-default/rslearn-eai/datasets/landsat_vessel_detection/classifier/dataset_20250624"
)
INDEX_DIR = UPath("/weka/dfive-default/yawenz/landsat/annotation_round1")

GROUP = "round1_20260803"
LAYER_NAME = "landsat"

# 512 px at 15 m/pixel = 7.68 km across. Centre 128 px = the annotation crop view;
# centre 64 px = the current training window.
WINDOW_SIZE = 512
WINDOW_RESOLUTION = 15
CROP_VIEW_SIZE = 128
TRAIN_VIEW_SIZE = 64

# The pool's timestamp is the scene acquisition time; items are pinned by name so this
# range only has to be wide enough that a later re-prepare would rediscover the scene.
TIME_BUFFER_MINUTES = 20

# Scene-level split quotas, applied within each (stratum, tier) cell by detection
# count. Splitting on scene keeps every detection from one acquisition in one split,
# so no location or lighting condition leaks between train and test.
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10

PRODUCT_RE = re.compile(r"^(LC0[89])_(L1\w{2})_(\d{6})_(\d{8})_(\d{8})_02_(T1|T2|RT)$")
TIER_RANK = {"T1": 0, "T2": 1, "RT": 2}


def parse_product(product_id: str) -> tuple[str, str, str, str, str, str]:
    """Split a Landsat product id into (platform, level, pathrow, acquired, processed, tier)."""
    match = PRODUCT_RE.match(product_id)
    if not match:
        raise ValueError(f"cannot parse product id {product_id}")
    return match.groups()  # type: ignore[return-value]


def stable_hash(value: str) -> int:
    """A hash that is stable across processes and releases (unlike hash())."""
    return int(hashlib.sha256(value.encode()).hexdigest()[:16], 16)


def assign_scene_splits(rows: list[dict]) -> dict[str, str]:
    """Assign every scene to train/val/test, stratified by (stratum, tier).

    Within each cell, scenes are ordered by a stable hash of the scene id and filled
    into test until the cell's test detection quota is met, then val, then train. Quotas
    are on detection counts rather than scene counts because detections per scene vary
    by an order of magnitude (flood scenes vs quiet ones).
    """
    cells: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        cells[(row["stratum"], row["tier"])][row["scene_id"]] += 1

    splits: dict[str, str] = {}
    for cell, scene_counts in sorted(cells.items()):
        total = sum(scene_counts.values())
        quotas = {
            "test": TEST_FRACTION * total,
            "val": VAL_FRACTION * total,
        }
        filled = {"test": 0, "val": 0}
        for scene_id in sorted(scene_counts, key=lambda s: stable_hash(s + str(cell))):
            count = scene_counts[scene_id]
            for split in ("test", "val"):
                # Fill up to but never past the quota, so overshoot never comes out of
                # train. The filled == 0 case is the exception: a cell whose scenes are
                # all larger than its quota must still contribute one, or that stratum
                # would be missing from the split entirely.
                if filled[split] + count <= quotas[split] or filled[split] == 0:
                    splits[scene_id] = split
                    filled[split] += count
                    break
            else:
                splits[scene_id] = "train"
    return splits


def resolve_items(
    make_data_source: Callable[[], Any], scene_ids: set[str], workers: int
) -> tuple[dict[str, object], dict[str, str]]:
    """Look up one data source item per scene, substituting for deleted RT products.

    Returns (item by scene id, substituted product id by scene id). Scenes whose
    product is gone and which have no same-acquisition replacement are absent from the
    first dict, and are reported by the caller rather than silently dropped.

    One product listing per (year, path, row) takes several seconds against S3, so the
    listings run on a thread pool. boto3 clients are not thread-safe, so each worker
    thread builds its own data source; the on-disk metadata cache is shared and written
    atomically, so a second run of this script costs no S3 calls at all.
    """
    by_pathrow: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    for scene_id in scene_ids:
        _, _, pathrow, acquired, _, _ = parse_product(scene_id)
        by_pathrow[(int(acquired[:4]), pathrow[:3], pathrow[3:])].add(scene_id)

    local = threading.local()

    def list_products(key: tuple[int, str, str]) -> dict[str, Any]:
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

    items: dict[str, object] = {}
    substitutions: dict[str, str] = {}
    for key, available in zip(keys, listings):
        for scene_id in sorted(by_pathrow[key]):
            if scene_id in available:
                items[scene_id] = available[scene_id]
                continue
            # Product is gone (RT superseded). Prefer T1 over T2 over RT, and the most
            # recent processing date, among products for the same acquisition.
            platform, _, pathrow, acquired, _, _ = parse_product(scene_id)
            candidates = []
            for name in available:
                cand_platform, _, cand_pathrow, cand_acquired, processed, tier = (
                    parse_product(name)
                )
                if (cand_platform, cand_pathrow, cand_acquired) == (
                    platform,
                    pathrow,
                    acquired,
                ):
                    candidates.append((TIER_RANK[tier], -int(processed), name))
            if not candidates:
                continue
            replacement = sorted(candidates)[0][2]
            items[scene_id] = available[replacement]
            substitutions[scene_id] = replacement
    return items, substitutions


def window_bounds(
    lon: float, lat: float
) -> tuple[Projection, tuple[int, ...], float, float]:
    """Projection and pixel bounds of the window centred on a detection."""
    projection = Projection(
        get_utm_ups_crs(lon, lat), WINDOW_RESOLUTION, -WINDOW_RESOLUTION
    )
    geometry = STGeometry(
        WGS84_PROJECTION, shapely.Point(lon, lat), None
    ).to_projection(projection)
    # Round to whole pixels so the detection sits at the centre pixel of the window,
    # matching how the detector's own crop windows are cut.
    col = int(geometry.shp.x)
    row = int(geometry.shp.y)
    half = WINDOW_SIZE // 2
    bounds = (col - half, row - half, col + half, row + half)
    return projection, bounds, geometry.shp.x, geometry.shp.y


def main() -> None:
    """Create rslearn windows for the round-1 annotation group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default=GROUP, help="window group name")
    parser.add_argument("--limit", type=int, default=None, help="only first N rows")
    parser.add_argument(
        "--workers", type=int, default=8, help="threads for S3 product listings"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="resolve items and splits, print the summary, write nothing",
    )
    args = parser.parse_args()

    with POOL_PATH.open() as f:
        features = json.load(f)["features"]
    rows = [feature["properties"] for feature in features]
    if args.limit:
        rows = rows[: args.limit]
    print(f"pool: {len(rows)} detections, {len({r['scene_id'] for r in rows})} scenes")

    dataset = Dataset(DATASET_ROOT)
    layer_config = dataset.layers[LAYER_NAME]

    items, substitutions = resolve_items(
        lambda: layer_config.instantiate_data_source(dataset.path),
        {r["scene_id"] for r in rows},
        args.workers,
    )
    missing = {r["scene_id"] for r in rows} - set(items)
    print(
        f"resolved {len(items)} scenes "
        f"({len(substitutions)} substituted RT -> definitive), {len(missing)} unresolved"
    )
    for scene_id in sorted(missing):
        print(f"  UNRESOLVED {scene_id}")

    splits = assign_scene_splits(rows)
    split_counts: dict[str, int] = defaultdict(int)
    index: list[dict] = []

    for row in tqdm.tqdm(rows, desc="creating windows"):
        scene_id = row["scene_id"]
        if scene_id not in items:
            continue
        item = items[scene_id]
        lon, lat = float(row["longitude"]), float(row["latitude"])
        projection, bounds, centre_x, centre_y = window_bounds(lon, lat)
        ts = datetime.fromisoformat(row["ts"])
        time_range = (
            ts - timedelta(minutes=TIME_BUFFER_MINUTES),
            ts + timedelta(minutes=TIME_BUFFER_MINUTES),
        )
        split = splits[scene_id]
        split_counts[split] += 1

        record = {
            "window": row["detection_id"],
            "group": args.group,
            "split": split,
            "crs": str(projection.crs),
            "bounds": list(bounds),
            "resolution": WINDOW_RESOLUTION,
            "crop_view_size": CROP_VIEW_SIZE,
            "train_view_size": TRAIN_VIEW_SIZE,
            "centre_offset_px": [centre_x - bounds[0], centre_y - bounds[1]],
            "item_name": item.name,  # type: ignore[attr-defined]
            "item_blob_path": item.blob_path,  # type: ignore[attr-defined]
            "substituted_product": substitutions.get(scene_id),
            **row,
        }
        index.append(record)

        if args.dry_run:
            continue

        window = Window(
            storage=dataset.storage,
            group=args.group,
            name=row["detection_id"],
            projection=projection,
            bounds=bounds,
            time_range=time_range,
            options={
                "split": split,
                "weight": 1,
                "stratum": row["stratum"],
                "slice": row["slice"],
                "tier": row["tier"],
            },
            data_factory=dataset.window_data_storage_factory,
        )
        window.save()
        window.save_layer_datas(
            {
                LAYER_NAME: WindowLayerData(
                    layer_name=LAYER_NAME,
                    serialized_item_groups=[[item.serialize()]],  # type: ignore[attr-defined]
                    materialized=False,
                )
            }
        )

    print("\nsplit (detections):", dict(sorted(split_counts.items())))
    scene_splits: dict[str, int] = defaultdict(int)
    for scene_id in {r["scene_id"] for r in rows}:
        scene_splits[splits[scene_id]] += 1
    print("split (scenes):    ", dict(sorted(scene_splits.items())))

    if args.dry_run:
        print("\ndry run: no windows written")
        return

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_path = INDEX_DIR / f"{args.group}_index.json"
    with index_path.open("w") as f:
        json.dump(index, f)
    print(f"\nwrote {len(index)} windows to {DATASET_ROOT / 'windows' / args.group}")
    print(f"wrote index to {index_path}")


if __name__ == "__main__":
    main()
