"""Select same-track historical images for a Sentinel-1 vessel detection dataset.

rslearn's own prepare step cannot express "same relative orbit as the target scene, random
pass per period", so this workflow fills in the historical layer of each window itself.

The target layer must already be prepared (its first item group holds the target scene).
For each window:

- The region is the target footprint intersected with the window.
- Candidate passes are all GRD frames (VV+VH) on the target's relative orbit, grouped
  into passes (frames less than 10 minutes apart). A pass qualifies if the union of its
  frames covers at least `min_coverage` of the region.
- Time is split into periods counted back from the window start t0:
  [t0-35d, t0-5d), [t0-65d, t0-35d), ... (with the defaults). The periods are visited most
  recent first, empty periods are skipped, and at most `max_groups` groups are kept.
- Within a period, one qualifying pass is chosen at random, with an RNG seeded by the
  window group and name, so the selection is reproducible. The group holds the frames of
  that pass needed to reach its full coverage of the region.

After this, materialize the historical layer as usual (rslearn dataset materialize).
"""

import hashlib
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import cache
from typing import Any

import shapely
import tqdm
from rslearn.config.dataset import QueryConfig, SpaceMode
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.copernicus import (
    CopernicusItem,
    Sentinel1,
    Sentinel1Polarisation,
    Sentinel1ProductType,
)
from rslearn.dataset import Dataset, Window
from rslearn.dataset.window import WindowLayerData
from rslearn.utils.geometry import STGeometry
from upath import UPath

from rslp.log_utils import get_logger

logger = get_logger(__name__)

# Frames of one pass are acquired seconds apart; passes on the same track are days apart.
PASS_GAP = timedelta(minutes=10)


@cache
def get_relative_orbit(scene_name: str) -> int:
    """Get the relative orbit number of a Sentinel-1 scene from the Copernicus API."""
    if not scene_name.endswith(".SAFE"):
        scene_name += ".SAFE"
    data_source = Sentinel1(
        product_type=Sentinel1ProductType.IW_GRDH,
        polarisation=Sentinel1Polarisation.VV_VH,
        timeout=120,
    )
    product = data_source._get_product(scene_name, expand_attributes=True)
    attributes = {a["Name"]: a["Value"] for a in product["Attributes"]}
    return int(attributes["relativeOrbitNumber"])


def search_same_track(
    geometry: STGeometry, relative_orbit: int
) -> list[CopernicusItem]:
    """Get the GRD VV+VH items on the given relative orbit that intersect the geometry."""
    data_source = Sentinel1(
        product_type=Sentinel1ProductType.IW_GRDH,
        polarisation=Sentinel1Polarisation.VV_VH,
        timeout=120,
    )
    data_source.query_filter += (
        " and Attributes/OData.CSC.IntegerAttribute/any(att:att/Name eq "
        "'relativeOrbitNumber' and att/OData.CSC.IntegerAttribute/Value eq "
        f"{relative_orbit})"
    )
    query_config = QueryConfig(space_mode=SpaceMode.INTERSECTS, max_matches=100000)
    groups = data_source.get_items([geometry], query_config)[0]
    return [item for group in groups for item in group.items]


def group_into_passes(
    items: list[CopernicusItem], exclude_scenes: set[str]
) -> list[list[CopernicusItem]]:
    """Group same-track frames into passes, sorted by time.

    Reprocessed copies of a frame (same name up to the checksum suffix) and COG variants
    are dropped; the copy kept is the first by name, so the result doesn't depend on
    the API's result order.
    """
    seen = set()
    frames = []
    for item in sorted(items, key=lambda item: item.name):
        if "_COG" in item.name or item.name in exclude_scenes:
            continue
        prefix = item.name.rsplit("_", 1)[0]
        if prefix in seen:
            continue
        seen.add(prefix)
        frames.append(item)
    frames.sort(key=lambda item: item.geometry.time_range[0])

    passes: list[list[CopernicusItem]] = []
    for item in frames:
        start = item.geometry.time_range[0]
        if passes and start - passes[-1][-1].geometry.time_range[0] <= PASS_GAP:
            passes[-1].append(item)
        else:
            passes.append([item])
    return passes


def pick_frames(
    frames: list[CopernicusItem], region: shapely.Geometry
) -> tuple[float, list[CopernicusItem]]:
    """Pick the frames of a pass needed to reach its full coverage of the region.

    Frames are added greedily by overlap with the region.

    Returns:
        (fraction of the region covered by the whole pass, chosen frames)
    """
    footprints = {item.name: item.geometry.shp.buffer(0) for item in frames}
    full = (
        shapely.union_all(list(footprints.values())).intersection(region).area
        / region.area
    )
    chosen: list[CopernicusItem] = []
    union: shapely.Geometry | None = None
    for item in sorted(
        frames, key=lambda item: -footprints[item.name].intersection(region).area
    ):
        footprint = footprints[item.name]
        if footprint.intersection(region).area <= 0:
            continue
        chosen.append(item)
        union = footprint if union is None else union.union(footprint)
        if union.intersection(region).area / region.area >= full - 1e-6:
            break
    return full, chosen


def select_for_window(
    window: Window,
    target_item: CopernicusItem,
    passes: list[list[CopernicusItem]],
    periods: list[tuple[timedelta, timedelta]],
    max_groups: int,
    min_coverage: float,
) -> tuple[list[list[CopernicusItem]], list[tuple[datetime, datetime]]]:
    """Select the historical item groups for one window.

    Args:
        window: the window.
        target_item: the target scene of the window.
        passes: candidate same-track passes.
        periods: (start offset, end offset) of each period before the window start,
            most recent first.
        max_groups: maximum number of item groups.
        min_coverage: minimum fraction of the region a pass must cover.

    Returns:
        (item groups, time range of each group's period)
    """
    window_shp = window.get_geometry().to_projection(WGS84_PROJECTION).shp
    region = target_item.geometry.shp.buffer(0).intersection(window_shp)
    if region.area <= 0:
        return [], []

    t0 = window.time_range[0]
    seed = int(
        hashlib.md5(f"{window.group}/{window.name}".encode()).hexdigest()[:16], 16
    )
    rng = random.Random(seed)
    groups: list[list[CopernicusItem]] = []
    group_time_ranges: list[tuple[datetime, datetime]] = []
    for start_offset, end_offset in periods:
        if len(groups) >= max_groups:
            break
        lo, hi = t0 - end_offset, t0 - start_offset
        options = []
        for frames in passes:
            if not all(lo <= item.geometry.time_range[0] < hi for item in frames):
                continue
            if not any(item.geometry.shp.intersects(region) for item in frames):
                continue
            coverage, chosen = pick_frames(frames, region)
            if coverage >= min_coverage:
                options.append(chosen)
        if options:
            groups.append(rng.choice(options))
            group_time_ranges.append((lo, hi))
    return groups, group_time_ranges


def get_target_item(window: Window, target_layer: str) -> CopernicusItem | None:
    """Get the target scene of a window from its prepared target layer."""
    layer_datas = window.load_layer_datas()
    if target_layer not in layer_datas:
        return None
    item_groups = layer_datas[target_layer].serialized_item_groups
    if not item_groups or not item_groups[0]:
        return None
    return CopernicusItem.deserialize(item_groups[0][0])


def make_periods(
    min_days: int, period_days: int, num_periods: int
) -> list[tuple[timedelta, timedelta]]:
    """Get the (start offset, end offset) of each period, most recent first."""
    return [
        (
            timedelta(days=min_days + i * period_days),
            timedelta(days=min_days + (i + 1) * period_days),
        )
        for i in range(num_periods)
    ]


def select_for_target(
    target_item: CopernicusItem,
    windows: list[Window],
    periods: list[tuple[timedelta, timedelta]],
    max_groups: int,
    min_coverage: float,
    exclude_scenes: set[str],
) -> dict[str, Any]:
    """Select historical item groups for all windows sharing one target scene.

    One catalog search covers all of the windows.

    Returns:
        dict from window name to (item groups, group time ranges).
    """
    relative_orbit = get_relative_orbit(target_item.name)
    window_shps = [
        w.get_geometry().to_projection(WGS84_PROJECTION).shp for w in windows
    ]
    t0s = [w.time_range[0] for w in windows]
    search_geometry = STGeometry(
        WGS84_PROJECTION,
        shapely.union_all(window_shps).envelope,
        (min(t0s) - periods[-1][1], max(t0s) - periods[0][0]),
    )
    items = search_same_track(search_geometry, relative_orbit)
    passes = group_into_passes(items, exclude_scenes)
    return {
        w.name: select_for_window(
            w, target_item, passes, periods, max_groups, min_coverage
        )
        for w in windows
    }


def prepare(
    ds_path: str,
    target_layer: str = "sentinel1",
    historical_layer: str = "sentinel1_historical",
    groups: list[str] | None = None,
    min_days: int = 5,
    period_days: int = 30,
    num_periods: int = 5,
    max_groups: int = 4,
    min_coverage: float = 0.98,
    exclude_scenes_fname: str | None = None,
    workers: int = 16,
) -> None:
    """Fill in the historical layer of each window with same-track Sentinel-1 passes.

    Args:
        ds_path: the rslearn dataset path.
        target_layer: the layer holding the target scene, already prepared.
        historical_layer: the layer to fill in. Its items are overwritten.
        groups: only process windows in these groups (default all).
        min_days: the most recent period ends this many days before the window start.
        period_days: the length of each period in days.
        num_periods: the number of periods.
        max_groups: the maximum number of historical item groups per window.
        min_coverage: the minimum fraction of (target footprint ∩ window) that a pass
            must cover.
        exclude_scenes_fname: optional JSON file with a list of scene names (.SAFE) to
            treat as unavailable.
        workers: number of target scenes to process in parallel.
    """
    dataset = Dataset(UPath(ds_path))
    windows = dataset.load_windows(groups=groups, workers=workers, show_progress=True)
    periods = make_periods(min_days, period_days, num_periods)
    exclude_scenes: set[str] = set()
    if exclude_scenes_fname is not None:
        with UPath(exclude_scenes_fname).open() as f:
            exclude_scenes = set(json.load(f))

    # Windows sharing a target scene share one catalog search.
    by_target: dict[str, list[Window]] = defaultdict(list)
    target_items: dict[str, CopernicusItem] = {}
    num_skipped = 0
    for window in windows:
        target_item = get_target_item(window, target_layer)
        if target_item is None:
            num_skipped += 1
            continue
        by_target[target_item.name].append(window)
        target_items[target_item.name] = target_item
    logger.info(
        f"{len(windows)} windows, {len(by_target)} target scenes, "
        f"{num_skipped} windows without a prepared target"
    )

    def process(target_name: str) -> int:
        target_windows = by_target[target_name]
        selections = select_for_target(
            target_items[target_name],
            target_windows,
            periods,
            max_groups,
            min_coverage,
            exclude_scenes,
        )
        for window in target_windows:
            item_groups, group_time_ranges = selections[window.name]
            layer_datas = window.load_layer_datas()
            layer_datas[historical_layer] = WindowLayerData(
                layer_name=historical_layer,
                serialized_item_groups=[
                    [item.serialize() for item in group] for group in item_groups
                ],
                group_time_ranges=list(group_time_ranges),
            )
            window.save_layer_datas(layer_datas)
        return len(target_windows)

    with ThreadPoolExecutor(workers) as executor:
        list(
            tqdm.tqdm(
                executor.map(process, list(by_target.keys())), total=len(by_target)
            )
        )
