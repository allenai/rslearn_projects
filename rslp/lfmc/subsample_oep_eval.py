"""Tag an LFMC subset with the oep_eval tag.

The subset is selected independently for train/val/test. For each split, the
sampler first applies a one-year cap per location, then selects an exact target
number of windows. State quotas mimic the overall original dataset state
distribution. Within those state quotas, LFMC target values are balanced as a
soft constraint against the original train LFMC decile distribution.

By default, this script replaces the existing tag: it computes and validates the
new subset first, then removes previous tag occurrences, applies the tag to the
new selection, and overwrites the manifest/stats files.

Usage:
    python subsample_oep_eval.py \
        --dataset_path /path/to/dataset \
        --tag oep_eval \
        --target_train 3000 \
        --target_val 1000 \
        --target_test 1000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

LocKey = tuple[str, tuple[Any, ...]]

US_STATES_SHP = (
    "/weka/dfive-default/hadriens/datasets/Misc/Us states/cb_2018_us_state_20m.shp"
)
DEFAULT_TAG = "oep_eval"
DEFAULT_TARGET_TRAIN = 3000
DEFAULT_TARGET_VAL = 1000
DEFAULT_TARGET_TEST = 1000
LFMC_NUM_BINS = 10


@dataclass(frozen=True)
class Candidate:
    """One eligible subsampling candidate window."""

    window: dict
    state: str
    lfmc_value: float
    lfmc_bin: int

    @property
    def name(self) -> str:
        """Return the rslearn window name."""
        return self.window["_name"]


def load_windows(dataset_path: str) -> list[dict]:
    """Load all window metadata from the dataset."""
    windows_dir = os.path.join(dataset_path, "windows", "spatial_split")
    names = os.listdir(windows_dir)
    print(f"  Found {len(names)} window directories, loading metadata...")
    windows = []
    for i, name in enumerate(names):
        if i % 5000 == 0 and i > 0:
            print(f"    ... loaded {i}/{len(names)}")
        meta_path = os.path.join(windows_dir, name, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        meta["_dir"] = os.path.join(windows_dir, name)
        meta["_name"] = name
        windows.append(meta)
    return windows


def default_annotation_geojson(dataset_path: str) -> str:
    """Return the default annotation GeoJSON path for an LFMC OEP dataset."""
    return os.path.abspath(
        os.path.join(
            dataset_path, "..", "..", "run_data", "annotation_features.geojson"
        )
    )


def load_lfmc_targets(annotation_geojson: str) -> dict[str, float]:
    """Load LFMC target values keyed by OEP annotation task ID."""
    with open(annotation_geojson) as f:
        geojson = json.load(f)

    targets: dict[str, float] = {}
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        task_id = properties.get("oe_annotations_task_id")
        labels = properties.get("oe_labels", {})
        value = labels.get("value")
        if not isinstance(task_id, str) or value is None:
            continue
        targets[task_id] = float(value)
    return targets


def get_window_task_id(window: dict) -> str:
    """Return the OEP annotation task ID for a window."""
    task_id = window.get("options", {}).get("source_task_id")
    if isinstance(task_id, str):
        return task_id

    name = window.get("_name", "")
    if name.startswith("task_") and "_point_" in name:
        return name[len("task_") :].rsplit("_point_", 1)[0]
    raise ValueError(f"Could not determine source task ID for window {name}")


def get_lfmc_value(window: dict, lfmc_targets: dict[str, float]) -> float:
    """Return the LFMC target value for a window."""
    task_id = get_window_task_id(window)
    try:
        return lfmc_targets[task_id]
    except KeyError as exc:
        raise KeyError(
            f"Window {window.get('_name')} references task ID {task_id}, "
            "but no LFMC target was found in annotation_geojson"
        ) from exc


def compute_location_centroid_wgs84(
    crs: str, bounds: list, x_resolution: float, y_resolution: float
) -> tuple[float, float]:
    """Convert the center of pixel bounds to WGS84 lon/lat.

    Bounds are stored in pixel coordinates; multiply by resolution to get
    real-world UTM coordinates.
    """
    cx_pixel = (bounds[0] + bounds[2]) / 2.0
    cy_pixel = (bounds[1] + bounds[3]) / 2.0
    cx_utm = cx_pixel * x_resolution
    cy_utm = cy_pixel * y_resolution
    from pyproj import Transformer

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(cx_utm, cy_utm)
    return lon, lat


def get_window_date(window: dict) -> datetime:
    """Extract the sample date from a window's time_range."""
    return datetime.fromisoformat(window["time_range"][0])


def best_one_year_subset(windows: list[dict]) -> list[dict]:
    """Find the densest 365-day sliding window and return only those samples.

    Slides across all sample dates and picks the 365-day interval containing
    the most samples.
    """
    if len(windows) <= 1:
        return windows

    sorted_windows = sorted(windows, key=get_window_date)
    dates = [get_window_date(w) for w in sorted_windows]
    year_delta = timedelta(days=365)

    best_start = 0
    best_end = 0
    best_count = 0

    for i, start_date in enumerate(dates):
        end_date = start_date + year_delta
        j = i
        while j < len(dates) and dates[j] <= end_date:
            j += 1
        count = j - i
        if count > best_count:
            best_count = count
            best_start = i
            best_end = j

    return sorted_windows[best_start:best_end]


def location_key(window: dict) -> LocKey:
    """Return the spatial location key for a window."""
    return (window["projection"]["crs"], tuple(window["bounds"]))


def group_windows_by_location(windows: list[dict]) -> dict[LocKey, list[dict]]:
    """Group windows by exact projection/bounds location."""
    locations: dict[LocKey, list[dict]] = defaultdict(list)
    for window in windows:
        locations[location_key(window)].append(window)
    return locations


def assign_locations_to_states(
    locations: dict[LocKey, list[dict]], states_gdf: Any
) -> dict[LocKey, str]:
    """Assign each location key to a US state via point-in-polygon."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Point

    loc_keys = list(locations.keys())
    points = []
    for key in loc_keys:
        crs, bounds = key[0], list(key[1])
        first_window = locations[key][0]
        x_res = first_window["projection"]["x_resolution"]
        y_res = first_window["projection"]["y_resolution"]
        lon, lat = compute_location_centroid_wgs84(crs, bounds, x_res, y_res)
        points.append(Point(lon, lat))

    points_gdf = gpd.GeoDataFrame(
        {"loc_key": loc_keys}, geometry=points, crs="EPSG:4326"
    )
    states_gdf_4326 = states_gdf.to_crs("EPSG:4326")
    joined = gpd.sjoin(points_gdf, states_gdf_4326[["NAME", "geometry"]], how="left")

    loc_to_state: dict[LocKey, str] = {}
    for _, row in joined.iterrows():
        state = row.get("NAME")
        if isinstance(state, str) and not pd.isna(state):
            loc_to_state[row["loc_key"]] = state
        else:
            loc_to_state[row["loc_key"]] = "Unknown"
    return loc_to_state


def compute_state_sample_counts(
    locations: dict[LocKey, list[dict]], loc_to_state: dict[LocKey, str]
) -> Counter[str]:
    """Count original samples per state from grouped locations."""
    counts: Counter[str] = Counter()
    for loc_key, windows in locations.items():
        counts[loc_to_state[loc_key]] += len(windows)
    return counts


def allocate_quotas(
    weights: dict[Any, int] | Counter,
    capacities: dict[Any, int] | Counter,
    target: int,
) -> dict[Any, int]:
    """Allocate exact integer quotas from weighted categories with capacity caps."""
    if target < 0:
        raise ValueError("target must be non-negative")

    categories = sorted(capacities.keys(), key=str)
    quotas = {category: 0 for category in categories}
    total_capacity = sum(max(0, int(capacities[category])) for category in categories)
    if target > total_capacity:
        raise ValueError(
            f"Cannot allocate target {target}; only {total_capacity} candidates available"
        )
    if target == 0:
        return quotas

    active = [category for category in categories if capacities[category] > 0]
    remaining_target = target

    while active:
        active_weights = {
            category: max(float(weights.get(category, 0)), 0.0) for category in active
        }
        weight_sum = sum(active_weights.values())
        if weight_sum <= 0:
            active_weights = {
                category: float(capacities[category]) for category in active
            }
            weight_sum = sum(active_weights.values())

        capped = []
        for category in active:
            share = remaining_target * active_weights[category] / weight_sum
            if share >= capacities[category]:
                quotas[category] = int(capacities[category])
                remaining_target -= quotas[category]
                capped.append(category)

        if not capped:
            break
        active = [category for category in active if category not in set(capped)]

    if not active:
        if remaining_target != 0:
            raise RuntimeError("quota allocation failed after capacity capping")
        return quotas

    active_weights = {
        category: max(float(weights.get(category, 0)), 0.0) for category in active
    }
    weight_sum = sum(active_weights.values())
    if weight_sum <= 0:
        active_weights = {category: float(capacities[category]) for category in active}
        weight_sum = sum(active_weights.values())

    raw_shares = {
        category: remaining_target * active_weights[category] / weight_sum
        for category in active
    }
    for category, share in raw_shares.items():
        quotas[category] = min(int(capacities[category]), int(math.floor(share)))

    leftover = target - sum(quotas.values())
    order = sorted(
        active,
        key=lambda category: (
            raw_shares[category] - math.floor(raw_shares[category]),
            active_weights[category],
            str(category),
        ),
        reverse=True,
    )
    while leftover > 0:
        progressed = False
        for category in order:
            if quotas[category] >= capacities[category]:
                continue
            quotas[category] += 1
            leftover -= 1
            progressed = True
            if leftover == 0:
                break
        if not progressed:
            raise RuntimeError("quota allocation could not distribute leftovers")

    return quotas


def lfmc_decile_edges(
    train_windows: list[dict], lfmc_targets: dict[str, float]
) -> list[float]:
    """Compute original train LFMC decile edges."""
    values = [get_lfmc_value(window, lfmc_targets) for window in train_windows]
    if not values:
        raise ValueError("Cannot compute LFMC deciles without train windows")
    return [float(x) for x in np.percentile(np.asarray(values), np.arange(0, 101, 10))]


def lfmc_bin(value: float, edges: list[float]) -> int:
    """Assign an LFMC value to one of ten decile bins."""
    if not np.isfinite(value):
        raise ValueError(f"LFMC value is not finite: {value}")
    return int(np.searchsorted(np.asarray(edges[1:-1]), value, side="right"))


def count_lfmc_bins(
    windows: list[dict], lfmc_targets: dict[str, float], edges: list[float]
) -> Counter[int]:
    """Count windows by LFMC decile bin."""
    counts: Counter[int] = Counter({bin_id: 0 for bin_id in range(LFMC_NUM_BINS)})
    for window in windows:
        counts[lfmc_bin(get_lfmc_value(window, lfmc_targets), edges)] += 1
    return counts


def build_candidates(
    windows: list[dict],
    loc_to_state: dict[LocKey, str],
    lfmc_targets: dict[str, float],
    edges: list[float],
) -> tuple[list[Candidate], dict[LocKey, list[dict]], dict[LocKey, list[dict]]]:
    """Build one-year-capped candidates for a split."""
    locations = group_windows_by_location(windows)
    loc_one_year: dict[LocKey, list[dict]] = {}
    candidates = []

    for loc_key, loc_windows in locations.items():
        subset = best_one_year_subset(loc_windows)
        loc_one_year[loc_key] = subset
        state = loc_to_state[loc_key]
        for window in subset:
            value = get_lfmc_value(window, lfmc_targets)
            candidates.append(
                Candidate(
                    window=window,
                    state=state,
                    lfmc_value=value,
                    lfmc_bin=lfmc_bin(value, edges),
                )
            )

    return candidates, locations, loc_one_year


def shuffled(items: list[Candidate], rng: np.random.Generator) -> list[Candidate]:
    """Return a deterministically shuffled copy of candidate items."""
    output = list(items)
    rng.shuffle(output)
    return output


def select_by_lfmc_bins(
    candidates: list[Candidate],
    target: int,
    train_bin_counts: Counter[int],
    rng: np.random.Generator,
) -> tuple[list[Candidate], dict[int, int]]:
    """Select candidates to approximate train LFMC decile proportions."""
    bin_to_candidates: dict[int, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        bin_to_candidates[candidate.lfmc_bin].append(candidate)

    capacities = Counter(
        {
            bin_id: len(bin_to_candidates.get(bin_id, []))
            for bin_id in range(LFMC_NUM_BINS)
        }
    )
    quotas = allocate_quotas(train_bin_counts, capacities, target)

    selected = []
    for bin_id in range(LFMC_NUM_BINS):
        bin_candidates = shuffled(bin_to_candidates.get(bin_id, []), rng)
        selected.extend(bin_candidates[: quotas.get(bin_id, 0)])

    return selected, {int(k): int(v) for k, v in quotas.items()}


def improve_lfmc_with_same_state_swaps(
    selected: list[Candidate],
    candidates: list[Candidate],
    desired_bin_counts: dict[int, int],
    rng: np.random.Generator,
) -> list[Candidate]:
    """Swap within states to improve global LFMC bin alignment."""
    selected_by_name = {candidate.name: candidate for candidate in selected}
    selected_counts: Counter[int] = Counter(
        candidate.lfmc_bin for candidate in selected
    )

    selected_by_state_bin: dict[tuple[str, int], list[Candidate]] = defaultdict(list)
    unselected_by_state_bin: dict[tuple[str, int], list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        key = (candidate.state, candidate.lfmc_bin)
        if candidate.name in selected_by_name:
            selected_by_state_bin[key].append(candidate)
        else:
            unselected_by_state_bin[key].append(candidate)

    for values in selected_by_state_bin.values():
        rng.shuffle(values)
    for values in unselected_by_state_bin.values():
        rng.shuffle(values)

    states = sorted({candidate.state for candidate in candidates})
    max_swaps = len(selected)
    for _ in range(max_swaps):
        surplus_bins = sorted(
            [
                bin_id
                for bin_id in range(LFMC_NUM_BINS)
                if selected_counts[bin_id] > desired_bin_counts.get(bin_id, 0)
            ],
            key=lambda bin_id: selected_counts[bin_id]
            - desired_bin_counts.get(bin_id, 0),
            reverse=True,
        )
        deficit_bins = sorted(
            [
                bin_id
                for bin_id in range(LFMC_NUM_BINS)
                if selected_counts[bin_id] < desired_bin_counts.get(bin_id, 0)
            ],
            key=lambda bin_id: desired_bin_counts.get(bin_id, 0)
            - selected_counts[bin_id],
            reverse=True,
        )
        if not surplus_bins or not deficit_bins:
            break

        changed = False
        state_order = list(states)
        rng.shuffle(state_order)
        for deficit_bin in deficit_bins:
            for surplus_bin in surplus_bins:
                for state in state_order:
                    selected_pool = selected_by_state_bin[(state, surplus_bin)]
                    unselected_pool = unselected_by_state_bin[(state, deficit_bin)]
                    if not selected_pool or not unselected_pool:
                        continue

                    removed = selected_pool.pop()
                    added = unselected_pool.pop()
                    selected_by_state_bin[(state, deficit_bin)].append(added)
                    unselected_by_state_bin[(state, surplus_bin)].append(removed)
                    selected_by_name.pop(removed.name)
                    selected_by_name[added.name] = added
                    selected_counts[surplus_bin] -= 1
                    selected_counts[deficit_bin] += 1
                    changed = True
                    break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break

    return list(selected_by_name.values())


def total_variation_distance(
    observed: dict[Any, int] | Counter, desired: dict[Any, int] | Counter
) -> float:
    """Compute total variation distance between two count distributions."""
    observed_total = sum(observed.values())
    desired_total = sum(desired.values())
    if observed_total == 0 and desired_total == 0:
        return 0.0
    if observed_total == 0 or desired_total == 0:
        return 1.0

    keys = set(observed.keys()) | set(desired.keys())
    return 0.5 * sum(
        abs(observed.get(key, 0) / observed_total - desired.get(key, 0) / desired_total)
        for key in keys
    )


def bin_stats(
    edges: list[float],
    original_train_counts: Counter[int],
    available_counts: Counter[int],
    desired_counts: dict[int, int],
    selected_counts: Counter[int],
) -> dict[str, dict[str, float | int]]:
    """Build serializable per-LFMC-bin stats."""
    stats = {}
    for bin_id in range(LFMC_NUM_BINS):
        stats[f"bin_{bin_id}"] = {
            "edge_min": float(edges[bin_id]),
            "edge_max": float(edges[bin_id + 1]),
            "original_train_samples": int(original_train_counts.get(bin_id, 0)),
            "available_candidates": int(available_counts.get(bin_id, 0)),
            "desired_samples": int(desired_counts.get(bin_id, 0)),
            "selected_samples": int(selected_counts.get(bin_id, 0)),
        }
    return stats


def state_stats(
    original_state_counts: Counter[str],
    available_counts: Counter[str],
    desired_counts: dict[str, int],
    selected_counts: Counter[str],
) -> dict[str, dict[str, int]]:
    """Build serializable per-state stats."""
    states = sorted(
        set(original_state_counts.keys())
        | set(available_counts.keys())
        | set(desired_counts.keys())
        | set(selected_counts.keys())
    )
    return {
        state: {
            "original_total_samples": int(original_state_counts.get(state, 0)),
            "available_candidates": int(available_counts.get(state, 0)),
            "desired_samples": int(desired_counts.get(state, 0)),
            "selected_samples": int(selected_counts.get(state, 0)),
        }
        for state in states
    }


def select_candidates(
    candidates: list[Candidate],
    target: int,
    original_state_counts: Counter[str],
    train_bin_counts: Counter[int],
    seed: int,
) -> tuple[list[Candidate], dict]:
    """Select exact-count candidates with state and LFMC distribution targets."""
    if target > len(candidates):
        raise ValueError(
            f"Target {target} exceeds one-year-capped candidate pool of {len(candidates)}"
        )

    rng = np.random.default_rng(seed)
    candidates_by_state: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_state[candidate.state].append(candidate)

    state_capacities = Counter(
        {
            state: len(state_candidates)
            for state, state_candidates in candidates_by_state.items()
        }
    )
    desired_state_counts = allocate_quotas(
        original_state_counts, state_capacities, target
    )

    selected: list[Candidate] = []
    state_bin_desired: dict[str, dict[int, int]] = {}
    for state in sorted(desired_state_counts):
        quota = desired_state_counts[state]
        if quota == 0:
            continue
        state_selected, state_bin_quota = select_by_lfmc_bins(
            candidates_by_state[state], quota, train_bin_counts, rng
        )
        selected.extend(state_selected)
        state_bin_desired[state] = state_bin_quota

    available_bin_counts: Counter[int] = Counter(
        {bin_id: 0 for bin_id in range(LFMC_NUM_BINS)}
    )
    for candidate in candidates:
        available_bin_counts[candidate.lfmc_bin] += 1
    desired_bin_counts = allocate_quotas(train_bin_counts, available_bin_counts, target)

    selected = improve_lfmc_with_same_state_swaps(
        selected, candidates, desired_bin_counts, rng
    )
    if len(selected) != target:
        raise RuntimeError(f"Selected {len(selected)} candidates, expected {target}")
    if len({candidate.name for candidate in selected}) != len(selected):
        raise RuntimeError("Selected duplicate windows")

    selected_state_counts = Counter(candidate.state for candidate in selected)
    selected_bin_counts = Counter(candidate.lfmc_bin for candidate in selected)

    selection_stats = {
        "state_capacities": {str(k): int(v) for k, v in state_capacities.items()},
        "desired_state_counts": {
            str(k): int(v) for k, v in desired_state_counts.items()
        },
        "selected_state_counts": {
            str(k): int(v) for k, v in selected_state_counts.items()
        },
        "available_lfmc_bin_counts": {
            str(k): int(v) for k, v in available_bin_counts.items()
        },
        "desired_lfmc_bin_counts": {
            str(k): int(v) for k, v in desired_bin_counts.items()
        },
        "selected_lfmc_bin_counts": {
            str(k): int(v) for k, v in selected_bin_counts.items()
        },
        "state_total_variation_distance": total_variation_distance(
            selected_state_counts, desired_state_counts
        ),
        "lfmc_total_variation_distance": total_variation_distance(
            selected_bin_counts, desired_bin_counts
        ),
        "state_bin_desired_counts": {
            state: {str(k): int(v) for k, v in quotas.items()}
            for state, quotas in state_bin_desired.items()
        },
    }
    return selected, selection_stats


def subsample_split(
    windows: list[dict],
    target: int,
    loc_to_state: dict[LocKey, str],
    original_state_counts: Counter[str],
    lfmc_targets: dict[str, float],
    lfmc_edges: list[float],
    train_bin_counts: Counter[int],
    seed: int,
    split_name: str,
) -> tuple[list[dict], dict]:
    """Subsample a split with exact counts, state quotas, and LFMC balancing."""
    print(f"\n--- {split_name.upper()} (target: {target}) ---")
    candidates, locations, loc_one_year = build_candidates(
        windows, loc_to_state, lfmc_targets, lfmc_edges
    )
    print(f"  Unique {split_name} locations: {len(locations)}")
    print(f"  One-year-capped {split_name} candidates: {len(candidates)}")

    selected_candidates, selection_stats = select_candidates(
        candidates, target, original_state_counts, train_bin_counts, seed
    )
    selected_windows = [candidate.window for candidate in selected_candidates]
    print(f"  Selected {len(selected_windows)} {split_name} samples")

    loc_one_year_counts = [len(windows) for windows in loc_one_year.values()]
    available_state_counts = Counter(candidate.state for candidate in candidates)
    selected_state_counts = Counter(
        candidate.state for candidate in selected_candidates
    )
    available_bin_counts = Counter(candidate.lfmc_bin for candidate in candidates)
    selected_bin_counts = Counter(
        candidate.lfmc_bin for candidate in selected_candidates
    )

    desired_state_counts = {
        state: int(count)
        for state, count in selection_stats["desired_state_counts"].items()
    }
    desired_bin_counts = {
        int(bin_id): int(count)
        for bin_id, count in selection_stats["desired_lfmc_bin_counts"].items()
    }

    split_stats = {
        "target_samples": target,
        "total_locations": len(locations),
        "candidate_locations": len(loc_one_year),
        "total_samples": len(windows),
        "candidate_samples": len(candidates),
        "selected_samples": len(selected_windows),
        "avg_samples_per_location_1yr_cap": float(np.mean(loc_one_year_counts)),
        "per_state": state_stats(
            original_state_counts,
            available_state_counts,
            desired_state_counts,
            selected_state_counts,
        ),
        "per_lfmc_decile": bin_stats(
            lfmc_edges,
            train_bin_counts,
            available_bin_counts,
            desired_bin_counts,
            selected_bin_counts,
        ),
        "state_total_variation_distance": selection_stats[
            "state_total_variation_distance"
        ],
        "lfmc_total_variation_distance": selection_stats[
            "lfmc_total_variation_distance"
        ],
        "state_bin_desired_counts": selection_stats["state_bin_desired_counts"],
    }
    return selected_windows, split_stats


def count_tagged_windows(windows: list[dict], tag: str) -> Counter[str]:
    """Count existing tagged windows by split."""
    counts: Counter[str] = Counter()
    for window in windows:
        if tag in window.get("options", {}):
            counts[window.get("options", {}).get("split", "unknown")] += 1
    return counts


def remove_tag_from_windows(windows: list[dict], tag: str) -> Counter[str]:
    """Remove a tag from all window metadata files."""
    removed: Counter[str] = Counter()
    for i, window in enumerate(windows):
        if i % 5000 == 0 and i > 0:
            print(f"    ... checked {i}/{len(windows)} windows for old tags")
        if tag not in window.get("options", {}):
            continue
        meta_path = os.path.join(window["_dir"], "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        if tag not in meta.get("options", {}):
            continue
        meta["options"].pop(tag)
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        window.get("options", {}).pop(tag, None)
        removed[meta.get("options", {}).get("split", "unknown")] += 1
    return removed


def tag_windows(windows: list[dict], tag: str) -> Counter[str]:
    """Add the given tag to the windows' metadata.json files."""
    tagged: Counter[str] = Counter()
    for i, window in enumerate(windows):
        if i % 2000 == 0 and i > 0:
            print(f"    ... tagged {i}/{len(windows)}")
        meta_path = os.path.join(window["_dir"], "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        if tag not in meta.get("options", {}):
            meta.setdefault("options", {})[tag] = ""
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            window.setdefault("options", {})[tag] = ""
        tagged[meta.get("options", {}).get("split", "unknown")] += 1
    return tagged


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tag a subset for LFMC")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--tag", type=str, default=DEFAULT_TAG, help="Tag name to apply"
    )
    parser.add_argument("--target_train", type=int, default=DEFAULT_TARGET_TRAIN)
    parser.add_argument("--target_val", type=int, default=DEFAULT_TARGET_VAL)
    parser.add_argument("--target_test", type=int, default=DEFAULT_TARGET_TEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--states_shp",
        type=str,
        default=US_STATES_SHP,
        help="Path to US states shapefile",
    )
    parser.add_argument(
        "--annotation_geojson",
        type=str,
        default=None,
        help=(
            "Path to annotation_features.geojson. Defaults to "
            "../../run_data/annotation_features.geojson relative to dataset_path."
        ),
    )
    parser.set_defaults(remove_existing_tags=True)
    parser.add_argument(
        "--remove_existing_tags",
        dest="remove_existing_tags",
        action="store_true",
        help="Remove previous occurrences of the selected tag before tagging new windows.",
    )
    parser.add_argument(
        "--keep_existing_tags",
        dest="remove_existing_tags",
        action="store_false",
        help="Fail if previous tag metadata or manifest files exist.",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print stats without tagging"
    )
    return parser.parse_args()


def main() -> None:
    """Tag a state- and LFMC-balanced subset for LFMC datasets."""
    args = parse_args()
    annotation_geojson = args.annotation_geojson or default_annotation_geojson(
        args.dataset_path
    )
    manifest_path = os.path.join(args.dataset_path, f"{args.tag}_manifest.json")
    stats_path = os.path.join(args.dataset_path, f"{args.tag}_stats.json")

    print(f"Loading windows from {args.dataset_path}...")
    all_windows = load_windows(args.dataset_path)
    print(f"  Total windows: {len(all_windows)}")

    existing_tag_counts = count_tagged_windows(all_windows, args.tag)
    existing_tag_total = sum(existing_tag_counts.values())
    manifest_exists = os.path.exists(manifest_path)
    if not args.remove_existing_tags and (manifest_exists or existing_tag_total > 0):
        raise RuntimeError(
            f"Existing tag data found for '{args.tag}' "
            f"(manifest_exists={manifest_exists}, tagged_windows={existing_tag_total}). "
            "Use the default replacement behavior or pass --remove_existing_tags."
        )

    train_windows = [
        w for w in all_windows if w.get("options", {}).get("split") == "train"
    ]
    val_windows = [w for w in all_windows if w.get("options", {}).get("split") == "val"]
    test_windows = [
        w for w in all_windows if w.get("options", {}).get("split") == "test"
    ]
    print(
        f"  Train: {len(train_windows)}, Val: {len(val_windows)}, Test: {len(test_windows)}"
    )

    print(f"Loading LFMC targets from {annotation_geojson}...")
    lfmc_targets = load_lfmc_targets(annotation_geojson)
    print(f"  Loaded {len(lfmc_targets)} LFMC targets")

    print("Loading US states shapefile...")
    import geopandas as gpd

    states_gdf = gpd.read_file(args.states_shp)

    all_locations = group_windows_by_location(all_windows)
    print(f"Assigning {len(all_locations)} locations to states...")
    loc_to_state = assign_locations_to_states(all_locations, states_gdf)
    original_state_counts = compute_state_sample_counts(all_locations, loc_to_state)

    lfmc_edges = lfmc_decile_edges(train_windows, lfmc_targets)
    train_bin_counts = count_lfmc_bins(train_windows, lfmc_targets, lfmc_edges)

    selected_train, train_stats = subsample_split(
        train_windows,
        args.target_train,
        loc_to_state,
        original_state_counts,
        lfmc_targets,
        lfmc_edges,
        train_bin_counts,
        args.seed,
        "train",
    )
    selected_val, val_stats = subsample_split(
        val_windows,
        args.target_val,
        loc_to_state,
        original_state_counts,
        lfmc_targets,
        lfmc_edges,
        train_bin_counts,
        args.seed + 1,
        "val",
    )
    selected_test, test_stats = subsample_split(
        test_windows,
        args.target_test,
        loc_to_state,
        original_state_counts,
        lfmc_targets,
        lfmc_edges,
        train_bin_counts,
        args.seed + 2,
        "test",
    )

    manifest = {
        "train": [w["_name"] for w in selected_train],
        "val": [w["_name"] for w in selected_val],
        "test": [w["_name"] for w in selected_test],
    }
    split_order = ["train", "val", "test", "unknown"]
    previous_tagged_samples = {
        split: int(existing_tag_counts.get(split, 0)) for split in split_order
    }
    zero_tagged_samples = {split: 0 for split in split_order}
    stats = {
        "tag": args.tag,
        "seed": args.seed,
        "annotation_geojson": annotation_geojson,
        "one_year_cap_days": 365,
        "remove_existing_tags": args.remove_existing_tags,
        "existing_tagged_samples": previous_tagged_samples,
        "previous_tagged_samples": previous_tagged_samples,
        "removed_tagged_samples": zero_tagged_samples.copy(),
        "total_samples": len(all_windows),
        "target_samples": {
            "train": args.target_train,
            "val": args.target_val,
            "test": args.target_test,
        },
        "tagged_samples": {
            "train": len(selected_train),
            "val": len(selected_val),
            "test": len(selected_test),
        },
        "original_overall_state_distribution": {
            state: {
                "samples": int(count),
                "proportion": float(count / len(all_windows)),
            }
            for state, count in sorted(original_state_counts.items())
        },
        "original_train_lfmc_deciles": {
            "edges": lfmc_edges,
            "counts": {str(k): int(v) for k, v in sorted(train_bin_counts.items())},
        },
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
    }

    print("\nExisting tag counts:")
    for split in split_order:
        print(f"  {split}: {existing_tag_counts.get(split, 0)}")

    if args.dry_run:
        print(
            "\n[DRY RUN] No tagging, tag removal, manifest writes, or stats writes performed."
        )
        if args.remove_existing_tags:
            print("Would remove old tag counts:")
            for split in split_order:
                print(f"  {split}: {previous_tagged_samples[split]}")
        print(f"Would write manifest to {manifest_path}")
        print(f"Would write stats to {stats_path}")
        return

    if args.remove_existing_tags:
        print(f"\nRemoving previous '{args.tag}' tags...")
        removed = remove_tag_from_windows(all_windows, args.tag)
        print(f"  Removed {sum(removed.values())} old tags")
        removed_tagged_samples = {
            split: int(removed.get(split, 0)) for split in split_order
        }
        stats["removed_tagged_samples"] = removed_tagged_samples
        for split in split_order:
            print(f"    {split}: {removed.get(split, 0)}")
    else:
        stats["removed_tagged_samples"] = zero_tagged_samples.copy()

    print(f"\nTagging selected train samples with '{args.tag}'...")
    train_tagged = tag_windows(selected_train, args.tag)
    print(f"  Tagged {train_tagged.get('train', 0)} train samples")

    print(f"Tagging selected val samples with '{args.tag}'...")
    val_tagged = tag_windows(selected_val, args.tag)
    print(f"  Tagged {val_tagged.get('val', 0)} val samples")

    print(f"Tagging selected test samples with '{args.tag}'...")
    test_tagged = tag_windows(selected_test, args.tag)
    print(f"  Tagged {test_tagged.get('test', 0)} test samples")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest to {manifest_path}")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote stats to {stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
