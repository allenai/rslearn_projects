"""Tag a spatially-balanced subset of LFMC windows with the oep_eval tag.

WARNING: This script is intended to be run exactly once per dataset. It will
refuse to run if a manifest file (oep_eval_manifest.json) from a previous run
is found.

For train: selects ~1000 windows distributed evenly across US states, operating
at the location level. Each selected location contributes at most 1 year of
samples (the densest 365-day window), maximizing geographic coverage.
For val/test: tags ALL windows.

Usage:
    python subsample_oep_eval.py --dataset_path /path/to/dataset --tag oep_eval --target_train 1000 --seed 42
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point

LocKey = tuple[str, tuple[Any, ...]]

US_STATES_SHP = (
    "/weka/dfive-default/hadriens/datasets/Misc/Us states/cb_2018_us_state_20m.shp"
)


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
        # Find how many samples fall within [start_date, start_date + 365 days]
        j = i
        while j < len(dates) and dates[j] <= end_date:
            j += 1
        count = j - i
        if count > best_count:
            best_count = count
            best_start = i
            best_end = j

    return sorted_windows[best_start:best_end]


def assign_locations_to_states(
    locations: dict[LocKey, list[dict]], states_gdf: gpd.GeoDataFrame
) -> dict[LocKey, str]:
    """Assign each location key to a US state via point-in-polygon."""
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


def select_locations_balanced(
    locations: dict[LocKey, list[dict]],
    loc_to_state: dict[LocKey, str],
    loc_one_year_count: dict[LocKey, int],
    target: int,
    seed: int,
) -> list[LocKey]:
    """Select locations with even distribution across states.

    Uses the 1-year capped sample count for each location when computing quotas,
    so that more locations can be selected for better geographic coverage.
    Returns list of selected location keys.
    """
    rng = np.random.default_rng(seed)

    state_to_locs: dict[str, list[LocKey]] = defaultdict(list)
    for loc_key, state in loc_to_state.items():
        state_to_locs[state].append(loc_key)

    states_with_data = [s for s in state_to_locs if s != "Unknown"]
    if not states_with_data:
        states_with_data = list(state_to_locs.keys())

    selected: list[LocKey] = []
    remaining_target = target

    # Iteratively allocate: states with fewer capped samples than quota get all
    settled_states = set()
    for _ in range(20):
        unsettled = [s for s in states_with_data if s not in settled_states]
        if not unsettled:
            break

        per_state_quota = remaining_target / len(unsettled) if unsettled else 0
        newly_settled = []

        for state in unsettled:
            locs = state_to_locs[state]
            total_capped = sum(loc_one_year_count[lk] for lk in locs)
            if total_capped <= per_state_quota:
                selected.extend(locs)
                remaining_target -= total_capped
                newly_settled.append(state)

        if not newly_settled:
            break
        settled_states.update(newly_settled)

    # For remaining states, select locations to fill quota
    unsettled = [s for s in states_with_data if s not in settled_states]
    if unsettled:
        per_state_quota = remaining_target / len(unsettled) if unsettled else 0
        for state in unsettled:
            locs = state_to_locs[state]
            locs_with_count = [(lk, loc_one_year_count[lk]) for lk in locs]

            # Shuffle for randomness, then sort by count (prefer dense locations)
            shuffled = list(locs_with_count)
            rng.shuffle(shuffled)
            shuffled.sort(key=lambda x: x[1], reverse=True)

            state_selected: list[LocKey] = []
            state_total = 0
            for lk, count in shuffled:
                if state_total + count > per_state_quota and state_selected:
                    break
                state_selected.append(lk)
                state_total += count

            selected.extend(state_selected)
            remaining_target -= state_total

    # Include "Unknown" state locations if any remain and we're under target
    if "Unknown" in state_to_locs and remaining_target > 0:
        unknown_locs = state_to_locs["Unknown"]
        unknown_with_count = [(lk, loc_one_year_count[lk]) for lk in unknown_locs]
        rng.shuffle(unknown_with_count)
        unknown_with_count.sort(key=lambda x: x[1], reverse=True)
        for lk, count in unknown_with_count:
            if remaining_target <= 0:
                break
            selected.append(lk)
            remaining_target -= count

    return selected


def tag_windows(windows: list[dict], tag: str) -> int:
    """Add the given tag to the windows' metadata.json files."""
    tagged = 0
    for i, w in enumerate(windows):
        if i % 2000 == 0 and i > 0:
            print(f"    ... tagged {i}/{len(windows)}")
        meta_path = os.path.join(w["_dir"], "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        if tag not in meta.get("options", {}):
            meta.setdefault("options", {})[tag] = ""
            with open(meta_path, "w") as f:
                json.dump(meta, f)
            tagged += 1
    return tagged


VAL_MAX_SAMPLES = 800
TEST_MAX_SAMPLES = 500


def subsample_split(
    windows: list[dict],
    target: int,
    states_gdf: gpd.GeoDataFrame,
    seed: int,
    split_name: str,
) -> tuple[list[dict], dict]:
    """Subsample a split using spatially-balanced selection with 1-year cap.

    Returns (selected_windows, split_stats_dict).
    """
    # Group by location
    locations: dict[LocKey, list[dict]] = defaultdict(list)
    for w in windows:
        key: LocKey = (w["projection"]["crs"], tuple(w["bounds"]))
        locations[key].append(w)
    print(f"  Unique {split_name} locations: {len(locations)}")

    # Precompute best 1-year subset for each location
    print(f"  Computing best 1-year window per {split_name} location...")
    loc_one_year: dict[LocKey, list[dict]] = {}
    loc_one_year_count: dict[LocKey, int] = {}
    for loc_key, loc_windows in locations.items():
        subset = best_one_year_subset(loc_windows)
        loc_one_year[loc_key] = subset
        loc_one_year_count[loc_key] = len(subset)
    avg_capped = np.mean(list(loc_one_year_count.values()))
    print(f"  Avg samples per location (1-year cap): {avg_capped:.1f}")

    # Assign locations to states
    print(f"  Assigning {split_name} locations to states...")
    loc_to_state = assign_locations_to_states(locations, states_gdf)

    # State distribution
    state_total_samples: dict[str, int] = defaultdict(int)
    state_total_locations: dict[str, int] = defaultdict(int)
    for loc_key, state in loc_to_state.items():
        state_total_samples[state] += len(locations[loc_key])
        state_total_locations[state] += 1

    # Select locations
    print(f"  Selecting locations for ~{target} {split_name} samples...")
    selected_locs = select_locations_balanced(
        locations, loc_to_state, loc_one_year_count, target, seed
    )

    # Collect the 1-year subset
    selected_windows = []
    for loc_key in selected_locs:
        selected_windows.extend(loc_one_year[loc_key])
    print(
        f"  Selected {len(selected_locs)} locations -> {len(selected_windows)} {split_name} samples"
    )

    # Per-state reporting
    selected_state_samples: dict[str, int] = defaultdict(int)
    selected_state_locations: dict[str, int] = defaultdict(int)
    for loc_key in selected_locs:
        state = loc_to_state[loc_key]
        selected_state_samples[state] += loc_one_year_count[loc_key]
        selected_state_locations[state] += 1
    print(f"\n  Per-state {split_name} selection:")
    for state in sorted(selected_state_samples.keys()):
        total_locs = state_total_locations[state]
        sel_locs = selected_state_locations[state]
        total_samp = state_total_samples[state]
        sel_samp = selected_state_samples[state]
        print(
            f"    {state}: {sel_locs}/{total_locs} locations, {sel_samp}/{total_samp} samples"
        )

    # Build stats for this split
    all_states = sorted(
        set(list(selected_state_samples.keys()) + list(state_total_locations.keys()))
    )
    split_stats = {
        "target_samples": target,
        "total_locations": len(locations),
        "selected_locations": len(selected_locs),
        "total_samples": len(windows),
        "selected_samples": len(selected_windows),
        "avg_samples_per_location_1yr_cap": float(avg_capped),
        "per_state": {
            state: {
                "total_locations": state_total_locations[state],
                "selected_locations": selected_state_locations.get(state, 0),
                "total_samples": state_total_samples[state],
                "selected_samples": selected_state_samples.get(state, 0),
            }
            for state in all_states
        },
    }

    return selected_windows, split_stats


def main() -> None:
    """Tag a spatially-balanced subset for LFMC datasets."""
    parser = argparse.ArgumentParser(description="Tag a subset for LFMC")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--tag", type=str, default="oep_eval", help="Tag name to apply")
    parser.add_argument("--target_train", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--states_shp",
        type=str,
        default=US_STATES_SHP,
        help="Path to US states shapefile",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print stats without tagging"
    )
    args = parser.parse_args()

    manifest_path = os.path.join(args.dataset_path, "oep_eval_manifest.json")
    if os.path.exists(manifest_path):
        raise RuntimeError(
            f"Manifest file already exists at {manifest_path}. "
            "This script is intended to be run exactly once per dataset. "
            "Remove the manifest file manually if you need to re-run."
        )

    print(f"Loading windows from {args.dataset_path}...")
    all_windows = load_windows(args.dataset_path)
    print(f"  Total windows: {len(all_windows)}")

    # Split by train/val/test
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

    # Load states shapefile
    print("Loading US states shapefile...")
    states_gdf = gpd.read_file(args.states_shp)

    # --- Train subsampling ---
    print(f"\n--- TRAIN (target: {args.target_train}) ---")
    selected_train, train_stats = subsample_split(
        train_windows, args.target_train, states_gdf, args.seed, "train"
    )

    # --- Val: subsample if over threshold, else use all ---
    if len(val_windows) > VAL_MAX_SAMPLES:
        print(
            f"\n--- VAL (target: {VAL_MAX_SAMPLES}, total {len(val_windows)} > {VAL_MAX_SAMPLES}) ---"
        )
        selected_val, val_stats = subsample_split(
            val_windows, VAL_MAX_SAMPLES, states_gdf, args.seed + 1, "val"
        )
    else:
        print(
            f"\n--- VAL: using all {len(val_windows)} samples (below {VAL_MAX_SAMPLES} threshold) ---"
        )
        selected_val = val_windows
        val_stats = {
            "total_samples": len(val_windows),
            "selected_samples": len(val_windows),
        }

    # --- Test: subsample if over threshold, else use all ---
    if len(test_windows) > TEST_MAX_SAMPLES:
        print(
            f"\n--- TEST (target: {TEST_MAX_SAMPLES}, total {len(test_windows)} > {TEST_MAX_SAMPLES}) ---"
        )
        selected_test, test_stats = subsample_split(
            test_windows, TEST_MAX_SAMPLES, states_gdf, args.seed + 2, "test"
        )
    else:
        print(
            f"\n--- TEST: using all {len(test_windows)} samples (below {TEST_MAX_SAMPLES} threshold) ---"
        )
        selected_test = test_windows
        test_stats = {
            "total_samples": len(test_windows),
            "selected_samples": len(test_windows),
        }

    if args.dry_run:
        print("\n[DRY RUN] No tagging performed.")
        return

    # Tag windows
    print(f"\nTagging selected train samples with '{args.tag}'...")
    train_tagged = tag_windows(selected_train, args.tag)
    print(f"  Tagged {train_tagged} train samples")

    print(f"Tagging selected val samples with '{args.tag}'...")
    val_tagged = tag_windows(selected_val, args.tag)
    print(f"  Tagged {val_tagged} val samples")

    print(f"Tagging selected test samples with '{args.tag}'...")
    test_tagged = tag_windows(selected_test, args.tag)
    print(f"  Tagged {test_tagged} test samples")

    # Write manifest
    manifest = {
        "train": [w["_name"] for w in selected_train],
        "val": [w["_name"] for w in selected_val],
        "test": [w["_name"] for w in selected_test],
    }
    manifest_path = os.path.join(args.dataset_path, "oep_eval_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote manifest to {manifest_path}")

    # Write stats
    stats = {
        "tag": args.tag,
        "seed": args.seed,
        "one_year_cap_days": 365,
        "val_max_samples_threshold": VAL_MAX_SAMPLES,
        "test_max_samples_threshold": TEST_MAX_SAMPLES,
        "total_samples": len(all_windows),
        "tagged_samples": {
            "train": len(selected_train),
            "val": len(selected_val),
            "test": len(selected_test),
        },
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
    }
    stats_path = os.path.join(args.dataset_path, f"{args.tag}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote stats to {stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
