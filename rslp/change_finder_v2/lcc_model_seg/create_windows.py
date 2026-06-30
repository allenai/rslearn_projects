"""Create rslearn windows + segmentation labels for the LCC seg model.

This script takes one or more v2 annotation JSONs (the same format used by the
dual-pass LCC model, see ``rslp.change_finder_v2.lcc_model.prepare``) and creates
an rslearn dataset with:
- One 128x128 window per annotation entry, with a single-instant ``time_range``.
- A ``label`` layer: a single-band uint8 raster with per-pixel transition class
  IDs at the annotated points (0 elsewhere, which the SegmentationTask masks out).

Unlike the dual-pass LCC model, there is no imagery querying here. The imagery is
fetched entirely by the standard ``rslearn dataset prepare`` / ``materialize``
pipeline, driven by the OlmoEarth Datasets source layers in ``config.json`` (which
derive the recent + historical Sentinel-2 stacks from the window timestamp).

The window timestamp T is chosen so the model observes the change inside its
8-month "gap":
- positive entries: T = post_change of the reference positive point.
- negative-only entries: T = midpoint of the entry's time_range.

The window is centered on the reference point so a random 64x64 training crop of
the 128x128 window reliably contains it.

Idempotent: windows that already exist are skipped, so re-running after adding new
annotations only creates the new windows.

After running this script, use ``rslearn dataset prepare`` / ``materialize`` to
download imagery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import shapely
from rslearn.config import LayerConfig
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.raster_array import RasterArray
from upath import UPath

WINDOW_SIZE = 128

# Maximum allowed span between pre_change and post_change for a usable positive
# point. This matches the 8-month gap in the dataset config (using 30-day months),
# so the change always falls inside the observed gap.
MAX_CHANGE_SPAN = timedelta(days=240)

NODATA = 0
NO_CHANGE = 1

# Segmentation classes: 0=nodata (masked), 1=no_change, 2..10 = transitions.
CLASS_NAMES = [
    "nodata",
    "no_change",
    "deforestation",
    "urban_expansion",
    "construction_mining",
    "from_water",
    "to_water",
    "urban_erosion",
    "new_crop_field",
    "wetland_loss",
    "forest_regrowth",
]

# Map (pre_category, post_category) -> transition class ID, derived from the
# requested transition groupings.
TRANSITION_MAP: dict[tuple[str, str], int] = {
    # deforestation
    ("tree", "grassland"): 2,
    ("tree", "crops"): 2,
    ("tree", "bare"): 2,
    ("tree", "urban/built-up"): 2,
    ("shrub", "grassland"): 2,
    # urban expansion
    ("crops", "urban/built-up"): 3,
    ("bare", "urban/built-up"): 3,
    ("grassland", "urban/built-up"): 3,
    ("water", "urban/built-up"): 3,
    ("urban/built-up", "urban/built-up"): 3,
    ("shrub", "urban/built-up"): 3,
    # construction/mining
    ("crops", "bare"): 4,
    ("grassland", "bare"): 4,
    # from water
    ("water", "bare"): 5,
    ("water", "grassland"): 5,
    # to water
    ("tree", "water"): 6,
    ("bare", "water"): 6,
    ("grassland", "water"): 6,
    ("crops", "water"): 6,
    # urban erosion
    ("urban/built-up", "crops"): 7,
    ("urban/built-up", "bare"): 7,
    ("urban/built-up", "grassland"): 7,
    # new crop field
    ("bare", "crops"): 8,
    ("grassland", "crops"): 8,
    ("water", "crops"): 8,
    # wetland loss
    ("wetland (herbaceous)", "water"): 9,
    # forest regrowth
    ("grassland", "tree"): 10,
}


def _parse_date(s: str) -> datetime:
    """Parse ISO date string to UTC datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _lonlat_to_pixel(
    lon: float, lat: float, projection: Projection, bounds: tuple[int, ...]
) -> tuple[int, int]:
    """Convert lon/lat to pixel coords within bounds, using floor for snapping."""
    st = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), time_range=None)
    projected = st.to_projection(projection)
    col = math.floor(projected.shp.x) - bounds[0]
    row = math.floor(projected.shp.y) - bounds[1]
    return col, row


# Fields that must all be present for a positive point to count as annotated.
POSITIVE_REQUIRED_FIELDS = (
    "pre_change",
    "post_change",
    "first_date_change_noticeable",
    "pre_category",
    "post_category",
)


def _positive_is_complete(pt: dict[str, Any]) -> bool:
    """Whether a positive point has all required annotation fields filled in."""
    return all(pt.get(field) for field in POSITIVE_REQUIRED_FIELDS)


def _point_transition_class(pt: dict[str, Any]) -> int | None:
    """Return the transition class ID for a usable positive point, else None.

    A point is usable if it is fully annotated, its pre/post span is within
    MAX_CHANGE_SPAN, and its (pre_category, post_category) pair is in TRANSITION_MAP.
    """
    if not _positive_is_complete(pt):
        return None
    pre_change = _parse_date(pt["pre_change"])
    post_change = _parse_date(pt["post_change"])
    if post_change - pre_change > MAX_CHANGE_SPAN:
        return None
    return TRANSITION_MAP.get((pt["pre_category"], pt["post_category"]))


def _write_label_layer(
    window: Window,
    layer_name: str,
    layer_config: LayerConfig,
    array_hw: np.ndarray,
) -> None:
    """Write a single-band uint8 label raster and mark layer complete."""
    band_set = layer_config.band_sets[0]
    chw = array_hw[np.newaxis, :, :].astype(np.uint8, copy=False)
    with window.data.open_layer_writer(layer_name) as writer:
        writer.write_raster(
            band_set.bands,
            band_set.instantiate_raster_format(),
            window.projection,
            window.bounds,
            RasterArray(chw_array=chw),
        )
    window.mark_layer_completed(layer_name)


def _process_entry(entry: dict[str, Any], dataset: Dataset) -> bool:
    """Create one window + label layer for an annotation entry.

    Returns True if a window was created, False if the entry was skipped because it
    has no usable positive point and no negative points.
    """
    projection = Projection.deserialize(entry["projection"])
    window_name = entry["window_name"]
    window_group = entry["group"]

    # Find the reference positive point (first usable one) to anchor timing/center.
    ref_point: dict[str, Any] | None = None
    for pt in entry.get("positive_points", []):
        if _point_transition_class(pt) is not None:
            ref_point = pt
            break

    if ref_point is not None:
        center_point = ref_point
        center_time = _parse_date(ref_point["post_change"])
    else:
        if not entry.get("negative_points"):
            return False
        center_point = entry["negative_points"][0]
        tr = entry["time_range"]
        t_start = _parse_date(tr[0])
        t_end = _parse_date(tr[1])
        center_time = t_start + (t_end - t_start) / 2

    # Center a 128x128 window on the reference/center point.
    st = STGeometry(
        WGS84_PROJECTION,
        shapely.Point(center_point["lon"], center_point["lat"]),
        time_range=None,
    )
    projected = st.to_projection(projection)
    center_col = math.floor(projected.shp.x)
    center_row = math.floor(projected.shp.y)
    half = WINDOW_SIZE // 2
    bounds = (
        center_col - half,
        center_row - half,
        center_col + half,
        center_row + half,
    )

    split_hash = hashlib.sha256(f"{window_group}/{window_name}".encode()).hexdigest()
    split = "val" if split_hash[0] in "01" else "train"

    window = Window(
        storage=dataset.storage,
        group=window_group,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=(center_time, center_time),
        options=dict(split=split),
        data_factory=dataset.window_data_storage_factory,
    )
    window.save()

    # Rasterize labels: negatives -> no_change, positives -> their transition class.
    h = bounds[3] - bounds[1]
    w = bounds[2] - bounds[0]
    label = np.zeros((h, w), dtype=np.uint8)

    for pt in entry.get("negative_points", []):
        col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
        if 0 <= col < w and 0 <= row < h:
            label[row, col] = NO_CHANGE

    for pt in entry.get("positive_points", []):
        cls = _point_transition_class(pt)
        if cls is None:
            continue
        col, row = _lonlat_to_pixel(pt["lon"], pt["lat"], projection, bounds)
        if 0 <= col < w and 0 <= row < h:
            label[row, col] = cls

    _write_label_layer(window, "label", dataset.layers["label"], label)
    return True


def create_windows(
    *,
    v2_json_paths: list[str],
    ds_path: str,
) -> None:
    """Create LCC seg windows + labels from v2 annotation JSONs.

    Idempotent: windows that already exist are skipped.

    Args:
        v2_json_paths: Paths to the v2 annotation JSONs.
        ds_path: Path to the output rslearn dataset (config.json must exist).
    """
    entries: list[dict[str, Any]] = []
    for v2_json_path in v2_json_paths:
        with open(v2_json_path) as f:
            entries.extend(json.load(f))

    ds_upath = UPath(ds_path)
    dataset = Dataset(ds_upath)

    created = 0
    skipped_existing = 0
    skipped_unusable = 0
    skipped_duplicate = 0
    skipped_incomplete = 0
    seen: set[tuple[str, str]] = set()

    for entry in entries:
        window_group = entry["group"]
        window_name = entry["window_name"]
        window_key = (window_group, window_name)
        if window_key in seen:
            skipped_duplicate += 1
            continue

        window_root = Window.get_window_root(ds_upath, window_group, window_name)
        if (window_root / "metadata.json").exists():
            skipped_existing += 1
            continue

        # Skip entries that are still mid-annotation: if there is at least one
        # positive point but any positive point is missing a required field, the
        # entry isn't finished. Negative-only entries are fine.
        positives = entry.get("positive_points", [])
        if positives and not all(_positive_is_complete(pt) for pt in positives):
            skipped_incomplete += 1
            continue

        if _process_entry(entry, dataset):
            created += 1
            seen.add(window_key)
            if created % 50 == 0:
                print(f"  Created {created} windows...")
        else:
            skipped_unusable += 1

    print(
        f"Created {created} windows, "
        f"skipped {skipped_existing} existing + {skipped_unusable} unusable "
        f"+ {skipped_incomplete} incomplete + {skipped_duplicate} duplicate inputs"
    )


def main() -> None:
    """Create LCC seg windows + labels from v2 annotation JSONs."""
    parser = argparse.ArgumentParser(
        description="Create LCC seg windows + labels from v2 annotation JSONs."
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
    args = parser.parse_args()

    create_windows(v2_json_paths=args.v2_json_paths, ds_path=args.ds_path)


if __name__ == "__main__":
    main()
