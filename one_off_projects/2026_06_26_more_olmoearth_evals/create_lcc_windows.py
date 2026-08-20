"""Create rslearn windows + binary labels for the LCC olmoearth_evals tasks.

This builds a small classification dataset from one or more change_finder_v2 v2
annotation JSONs (the same format used by ``rslp.change_finder_v2.lcc_model_seg``).

Two transition variants are supported, selected with ``--transition``:
- ``deforestation``: a point is positive iff it is a fully-annotated positive point
  whose ``pre_category == "tree"`` (tree -> anything). All other positive points and
  all negative points become negatives.
- ``urban_expansion``: a point is positive iff it is a fully-annotated positive point
  whose ``post_category == "urban/built-up"`` (anything -> urban). All other positive
  points and all negative points become negatives.

Each annotated point (positive or negative) becomes its own 64x64 window centered on
its lon/lat, with a single vector ``label`` layer carrying a ``category`` property
("positive" or "negative") for the stock ``ClassificationTask``.

The window timestamp drives the OlmoEarth Datasets Sentinel-2 source layers in
``config.json``: the window ``time_range`` is set to the full post-change calendar
year, so ``post_sentinel2`` (time_offset 0) covers that year and ``pre_sentinel2``
(time_offset -1095d) covers the year three years earlier. The post-change calendar
year is the year *after* the change date:
- positive points: year after ``post_change``.
- negative points: year after the midpoint of the entry ``time_range``.
Points whose pre year would be before 2017 are skipped.

Idempotent: windows that already exist are skipped, so re-running after adding new
annotations only creates the new windows.

After running this script, use ``rslearn dataset prepare`` / ``materialize`` to
download imagery (needs OlmoEarth Datasets creds: OEDATASETS_API_URL,
DATASETS_API_TOKEN).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from typing import Any

import shapely
from rslearn.dataset import Dataset, Window
from rslearn.utils import Feature
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

WINDOW_SIZE = 64
LABEL_LAYER = "label"

# Earliest pre year we allow; pre year = post year - 3.
MIN_PRE_YEAR = 2017

# Maximum number of negative points to keep per annotation entry (all positive
# points are always kept). If an entry has more, a deterministic random subset is
# selected.
MAX_NEGATIVES_PER_ENTRY = 4

TREE_CATEGORY = "tree"
URBAN_CATEGORY = "urban/built-up"

# Fields that must all be present for a positive point to be usable.
POSITIVE_REQUIRED_FIELDS = (
    "post_change",
    "pre_category",
    "post_category",
)


def _parse_date(s: str) -> datetime:
    """Parse ISO date string to UTC datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _positive_is_complete(pt: dict[str, Any]) -> bool:
    """Whether a positive point has the fields we need to use it."""
    return all(pt.get(field) for field in POSITIVE_REQUIRED_FIELDS)


def _is_positive_for_transition(pt: dict[str, Any], transition: str) -> bool:
    """Whether a complete positive point counts as positive for the transition."""
    if transition == "deforestation":
        return pt["pre_category"] == TREE_CATEGORY
    elif transition == "urban_expansion":
        return pt["post_category"] == URBAN_CATEGORY
    else:
        raise ValueError(f"unknown transition {transition}")


def _post_year_from_date(change_time: datetime) -> int:
    """Post-change calendar year is the year after the change date."""
    return change_time.year + 1


def _make_window(
    *,
    dataset: Dataset,
    ds_upath: UPath,
    group: str,
    window_name: str,
    projection: Projection,
    lon: float,
    lat: float,
    post_year: int,
    category: str,
) -> bool:
    """Create one window + label for a single point. Returns True if created."""
    window_root = Window.get_window_root(ds_upath, group, window_name)
    if (window_root / "metadata.json").exists():
        return False

    # Center a WINDOW_SIZE x WINDOW_SIZE window on the point.
    st = STGeometry(
        WGS84_PROJECTION, shapely.Point(lon, lat), time_range=None
    ).to_projection(projection)
    center_col = math.floor(st.shp.x)
    center_row = math.floor(st.shp.y)
    half = WINDOW_SIZE // 2
    bounds = (
        center_col - half,
        center_row - half,
        center_col + half,
        center_row + half,
    )

    time_range = (
        datetime(post_year, 1, 1, tzinfo=timezone.utc),
        datetime(post_year + 1, 1, 1, tzinfo=timezone.utc),
    )

    split_hash = hashlib.sha256(f"{group}/{window_name}".encode()).hexdigest()
    split = "val" if split_hash[0] in "01" else "train"

    window = Window(
        storage=dataset.storage,
        group=group,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        options=dict(split=split, category=category),
        data_factory=dataset.window_data_storage_factory,
    )
    window.save()

    feature = Feature(window.get_geometry(), {"category": category})
    with window.data.open_layer_writer(LABEL_LAYER) as writer:
        writer.write_vector(GeojsonVectorFormat(), [feature])
    window.mark_layer_completed(LABEL_LAYER)
    return True


def create_windows(
    *,
    v2_json_paths: list[str],
    ds_path: str,
    transition: str,
    max_per_class: int | None = None,
) -> None:
    """Create LCC classification windows + labels from v2 annotation JSONs.

    Args:
        v2_json_paths: Paths to the v2 annotation JSONs.
        ds_path: Path to the output rslearn dataset (config.json must exist).
        transition: "deforestation" or "urban_expansion".
        max_per_class: optional cap on the number of windows per class.
    """
    if transition not in ("deforestation", "urban_expansion"):
        raise ValueError(f"unknown transition {transition}")

    entries: list[dict[str, Any]] = []
    for v2_json_path in v2_json_paths:
        with open(v2_json_path) as f:
            entries.extend(json.load(f))

    ds_upath = UPath(ds_path)
    dataset = Dataset(ds_upath)

    counts = {"positive": 0, "negative": 0}
    created = 0
    skipped_existing = 0
    skipped_incomplete = 0
    skipped_old = 0
    skipped_capped = 0
    seen: set[tuple[str, str]] = set()

    def _emit(
        *,
        group: str,
        window_name: str,
        projection: Projection,
        lon: float,
        lat: float,
        post_year: int,
        category: str,
    ) -> None:
        nonlocal created, skipped_existing, skipped_old, skipped_capped
        if post_year - 3 < MIN_PRE_YEAR:
            skipped_old += 1
            return
        if max_per_class is not None and counts[category] >= max_per_class:
            skipped_capped += 1
            return
        key = (group, window_name)
        if key in seen:
            return
        seen.add(key)
        if _make_window(
            dataset=dataset,
            ds_upath=ds_upath,
            group=group,
            window_name=window_name,
            projection=projection,
            lon=lon,
            lat=lat,
            post_year=post_year,
            category=category,
        ):
            created += 1
            counts[category] += 1
            if created % 50 == 0:
                print(f"  Created {created} windows...")
        else:
            skipped_existing += 1

    for entry in entries:
        projection = Projection.deserialize(entry["projection"])
        group = entry["group"]
        base_name = entry["window_name"]

        # Negative-year reference: midpoint of the entry time_range.
        tr = entry["time_range"]
        t_start = _parse_date(tr[0])
        t_end = _parse_date(tr[1])
        neg_post_year = _post_year_from_date(t_start + (t_end - t_start) / 2)

        for idx, pt in enumerate(entry.get("positive_points", [])):
            if not _positive_is_complete(pt):
                skipped_incomplete += 1
                continue
            is_pos = _is_positive_for_transition(pt, transition)
            category = "positive" if is_pos else "negative"
            post_year = _post_year_from_date(_parse_date(pt["post_change"]))
            _emit(
                group=group,
                window_name=f"{base_name}_pos_{idx}",
                projection=projection,
                lon=pt["lon"],
                lat=pt["lat"],
                post_year=post_year,
                category=category,
            )

        # Keep all positive points but at most MAX_NEGATIVES_PER_ENTRY negative
        # points. If there are more, select a deterministic random subset (seeded by
        # the entry so re-runs are idempotent), preserving original indices for
        # stable window names.
        negative_points = list(enumerate(entry.get("negative_points", [])))
        if len(negative_points) > MAX_NEGATIVES_PER_ENTRY:
            rng = random.Random(f"{group}/{base_name}")
            negative_points = rng.sample(negative_points, MAX_NEGATIVES_PER_ENTRY)
        for idx, pt in negative_points:
            _emit(
                group=group,
                window_name=f"{base_name}_neg_{idx}",
                projection=projection,
                lon=pt["lon"],
                lat=pt["lat"],
                post_year=neg_post_year,
                category="negative",
            )

    print(
        f"Created {created} windows "
        f"(positive={counts['positive']}, negative={counts['negative']}); "
        f"skipped {skipped_existing} existing + {skipped_incomplete} incomplete "
        f"+ {skipped_old} too-old + {skipped_capped} capped"
    )


def main() -> None:
    """Create LCC classification windows + labels from v2 annotation JSONs."""
    parser = argparse.ArgumentParser(
        description="Create LCC classification windows + labels from v2 annotations."
    )
    parser.add_argument(
        "--v2-json-paths",
        nargs="+",
        required=True,
        help="Path(s) to v2 annotation JSONs.",
    )
    parser.add_argument(
        "--ds-path",
        required=True,
        help="Path to the rslearn dataset (config.json must exist).",
    )
    parser.add_argument(
        "--transition",
        required=True,
        choices=["deforestation", "urban_expansion"],
        help="Which positive transition to label.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional cap on number of windows per class.",
    )
    args = parser.parse_args()

    create_windows(
        v2_json_paths=args.v2_json_paths,
        ds_path=args.ds_path,
        transition=args.transition,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
