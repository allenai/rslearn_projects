"""Visualize LFMC OEP eval spatial and target distributions.

Produces a 2x4 figure for one dataset:
  - Row 1: original overall map, then tagged train/val/test maps.
  - Row 2: tagged split counts, then tagged train/val/test LFMC decile bars.

The LFMC decile panels compare each tagged split against the original train
LFMC target decile distribution. The original train decile edges are computed
from all original train windows.

Usage:
    python visualize_oep_eval.py \
        --dataset_path /path/to/dataset \
        --tag oep_eval \
        --output oep_eval_map.png
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

US_STATES_SHP = (
    "/weka/dfive-default/hadriens/datasets/Misc/Us states/cb_2018_us_state_20m.shp"
)

LFMC_NUM_BINS = 10
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#2176AE", "val": "#F77F00", "test": "#06D6A0"}
REFERENCE_COLOR = "#A8ADB4"


@dataclass(frozen=True)
class WindowRecord:
    """Metadata needed to plot and bin one rslearn window."""

    name: str
    split: str
    loc_key: tuple[str, tuple[Any, ...]]
    lon: float
    lat: float
    tagged: bool
    window: dict[str, Any]


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


def get_window_task_id(window: dict[str, Any]) -> str:
    """Return the OEP annotation task ID for a window."""
    task_id = window.get("options", {}).get("source_task_id")
    if isinstance(task_id, str):
        return task_id

    name = window.get("_name", "")
    if name.startswith("task_") and "_point_" in name:
        return name[len("task_") :].rsplit("_point_", 1)[0]
    raise ValueError(f"Could not determine source task ID for window {name}")


def get_lfmc_value(window: dict[str, Any], lfmc_targets: dict[str, float]) -> float:
    """Return the LFMC target value for a window."""
    task_id = get_window_task_id(window)
    try:
        return lfmc_targets[task_id]
    except KeyError as exc:
        raise KeyError(
            f"Window {window.get('_name')} references task ID {task_id}, "
            "but no LFMC target was found in annotation_geojson"
        ) from exc


def lfmc_decile_edges(
    train_windows: list[dict[str, Any]], lfmc_targets: dict[str, float]
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
    windows: list[dict[str, Any]],
    lfmc_targets: dict[str, float],
    edges: list[float],
) -> Counter[int]:
    """Count windows by LFMC decile bin."""
    counts: Counter[int] = Counter({bin_id: 0 for bin_id in range(LFMC_NUM_BINS)})
    for window in windows:
        counts[lfmc_bin(get_lfmc_value(window, lfmc_targets), edges)] += 1
    return counts


def compute_centroid_wgs84(
    metadata: dict[str, Any], transformer_cache: dict[str, Any]
) -> tuple[float, float]:
    """Compute a window centroid in WGS84 longitude/latitude."""
    crs = metadata["projection"]["crs"]
    bounds = metadata["bounds"]
    x_res = metadata["projection"]["x_resolution"]
    y_res = metadata["projection"]["y_resolution"]

    cx = (bounds[0] + bounds[2]) / 2.0 * x_res
    cy = (bounds[1] + bounds[3]) / 2.0 * y_res
    if crs not in transformer_cache:
        from pyproj import Transformer

        transformer_cache[crs] = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return transformer_cache[crs].transform(cx, cy)


def load_window_records(dataset_path: str, tag: str) -> list[WindowRecord]:
    """Load all spatial_split windows as plotting records."""
    windows_dir = os.path.join(dataset_path, "windows", "spatial_split")
    transformer_cache: dict[str, Any] = {}
    records: list[WindowRecord] = []

    window_names = sorted(os.listdir(windows_dir))
    for i, name in enumerate(window_names):
        if i > 0 and i % 5000 == 0:
            print(f"  ... loaded {i}/{len(window_names)} metadata files")
        meta_path = os.path.join(windows_dir, name, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            metadata = json.load(f)

        crs = metadata["projection"]["crs"]
        loc_key = (crs, tuple(metadata["bounds"]))
        lon, lat = compute_centroid_wgs84(metadata, transformer_cache)
        window = dict(metadata)
        window["_name"] = name

        records.append(
            WindowRecord(
                name=name,
                split=metadata.get("options", {}).get("split", "unknown"),
                loc_key=loc_key,
                lon=lon,
                lat=lat,
                tagged=tag in metadata.get("options", {}),
                window=window,
            )
        )

    return records


def load_manifest_names(dataset_path: str, tag: str) -> set[str]:
    """Load tagged window names from the manifest, if it exists."""
    manifest_path = os.path.join(dataset_path, f"{tag}_manifest.json")
    if not os.path.exists(manifest_path):
        return set()
    with open(manifest_path) as f:
        manifest = json.load(f)

    names: set[str] = set()
    for split in SPLIT_ORDER:
        names.update(manifest.get(split, []))
    return names


def select_tagged_records(
    records: list[WindowRecord], dataset_path: str, tag: str
) -> list[WindowRecord]:
    """Return records in the tagged subset, falling back to the manifest."""
    tagged_records = [record for record in records if record.tagged]
    if tagged_records:
        print(f"  Found {len(tagged_records)} records with metadata tag '{tag}'")
        return tagged_records

    manifest_names = load_manifest_names(dataset_path, tag)
    if not manifest_names:
        return []

    print(
        f"  No metadata tags found for '{tag}'; using "
        f"{len(manifest_names)} manifest entries"
    )
    return [record for record in records if record.name in manifest_names]


def unique_locations_by_split(
    records: list[WindowRecord],
) -> dict[str, list[tuple[float, float]]]:
    """Return unique point locations keyed by split."""
    seen: dict[str, set[tuple[str, tuple[Any, ...]]]] = defaultdict(set)
    locations: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for record in records:
        if record.split not in SPLIT_ORDER:
            continue
        if record.loc_key in seen[record.split]:
            continue
        seen[record.split].add(record.loc_key)
        locations[record.split].append((record.lon, record.lat))

    return dict(locations)


def unique_locations(records: list[WindowRecord]) -> list[tuple[float, float]]:
    """Return unique point locations from records."""
    seen: set[tuple[str, tuple[Any, ...]]] = set()
    locations: list[tuple[float, float]] = []
    for record in records:
        if record.loc_key in seen:
            continue
        seen.add(record.loc_key)
        locations.append((record.lon, record.lat))
    return locations


def format_map_axis(ax: Any, states_gdf: Any, title: str) -> None:
    """Draw the shared CONUS map frame on an axis."""
    states_gdf.boundary.plot(ax=ax, linewidth=0.45, color="#717780", zorder=1)
    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(True, linewidth=0.25, alpha=0.25)


def plot_original_map(ax: Any, records: list[WindowRecord], states_gdf: Any) -> None:
    """Plot the original overall spatial distribution."""
    format_map_axis(ax, states_gdf, "Original Overall")
    locations_by_split = unique_locations_by_split(records)
    for split in ["test", "val", "train"]:
        points = locations_by_split.get(split, [])
        if not points:
            continue
        lons, lats = zip(*points)
        ax.scatter(
            lons,
            lats,
            c=SPLIT_COLORS[split],
            s=12,
            alpha=0.55,
            edgecolors="none",
            label=f"{split} ({len(points)} locs)",
            zorder=3,
        )
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)


def plot_tagged_split_map(
    ax: Any,
    records: list[WindowRecord],
    states_gdf: Any,
    split: str,
) -> None:
    """Plot one tagged split's spatial distribution."""
    split_records = [record for record in records if record.split == split]
    points = unique_locations(split_records)
    title = f"Subsampled {split.title()}"
    format_map_axis(ax, states_gdf, title)
    if not points:
        ax.text(
            0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes
        )
        return

    lons, lats = zip(*points)
    ax.scatter(
        lons,
        lats,
        c=SPLIT_COLORS[split],
        s=18,
        alpha=0.78,
        edgecolors="none",
        label=f"{len(split_records)} samples, {len(points)} locs",
        zorder=3,
    )
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)


def plot_split_counts(ax: Any, records: list[WindowRecord]) -> None:
    """Plot tagged train/val/test sample counts."""
    counts = Counter(record.split for record in records)
    values = [counts.get(split, 0) for split in SPLIT_ORDER]
    x = np.arange(len(SPLIT_ORDER))
    bars = ax.bar(
        x,
        values,
        color=[SPLIT_COLORS[split] for split in SPLIT_ORDER],
        width=0.62,
    )

    ax.set_title("Subsampled Counts", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([split.title() for split in SPLIT_ORDER])
    ax.set_ylabel("Samples")
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    top = max(values) if values else 0
    ax.set_ylim(0, max(top * 1.16, 1))
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def counts_to_proportions(counts: Counter[int]) -> np.ndarray:
    """Convert decile counts to a fixed-length proportion array."""
    total = sum(counts.values())
    if total == 0:
        return np.zeros(LFMC_NUM_BINS)
    return np.asarray(
        [counts.get(bin_id, 0) / total for bin_id in range(LFMC_NUM_BINS)]
    )


def plot_decile_bars(
    ax: Any,
    records: list[WindowRecord],
    split: str,
    lfmc_targets: dict[str, float],
    lfmc_edges: list[float],
    reference_counts: Counter[int],
) -> None:
    """Plot one tagged split's LFMC decile distribution against train reference."""
    split_windows = [record.window for record in records if record.split == split]
    selected_counts = count_lfmc_bins(split_windows, lfmc_targets, lfmc_edges)
    reference_props = counts_to_proportions(reference_counts)
    selected_props = counts_to_proportions(selected_counts)

    x = np.arange(LFMC_NUM_BINS)
    width = 0.38
    ax.bar(
        x - width / 2,
        reference_props,
        width=width,
        color=REFERENCE_COLOR,
        label="Original train",
    )
    ax.bar(
        x + width / 2,
        selected_props,
        width=width,
        color=SPLIT_COLORS[split],
        label=f"Subsampled {split}",
    )

    ax.set_title(
        f"{split.title()} LFMC Deciles (n={len(split_windows):,})",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(LFMC_NUM_BINS)])
    ax.set_xlabel("Decile")
    ax.set_ylabel("Share of samples")
    ax.set_ylim(
        0, max(float(reference_props.max()), float(selected_props.max()), 0.1) * 1.18
    )
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7, framealpha=0.85)


def load_states(states_shp: str) -> Any:
    """Load CONUS state boundaries in WGS84."""
    import geopandas as gpd

    states_gdf = gpd.read_file(states_shp)
    states_gdf = states_gdf[~states_gdf["NAME"].isin(["Alaska", "Hawaii"])]
    return states_gdf.to_crs("EPSG:4326")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize LFMC OEP eval subset")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument(
        "--woody_path",
        type=str,
        default=None,
        help="Legacy alias for --dataset_path.",
    )
    parser.add_argument(
        "--herbaceous_path",
        type=str,
        default=None,
        help="Deprecated; this 2x4 dashboard visualizes one dataset at a time.",
    )
    parser.add_argument(
        "--tag", type=str, default="oep_eval", help="Tag name to filter on"
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
    parser.add_argument("--states_shp", type=str, default=US_STATES_SHP)
    parser.add_argument("--output", type=str, default="oep_eval_map.png")
    return parser.parse_args()


def main() -> None:
    """Visualize tagged subsampled vs original LFMC distributions."""
    args = parse_args()
    dataset_path = args.dataset_path or args.woody_path
    if dataset_path is None:
        raise ValueError("Pass --dataset_path. Legacy --woody_path is also accepted.")
    if args.herbaceous_path is not None:
        print("Ignoring --herbaceous_path; this dashboard visualizes one dataset.")

    annotation_geojson = args.annotation_geojson or default_annotation_geojson(
        dataset_path
    )

    print("Loading US states shapefile...")
    states_gdf = load_states(args.states_shp)

    print(f"Loading windows from {dataset_path}...")
    records = load_window_records(dataset_path, args.tag)
    print(f"  Loaded {len(records)} windows")

    print(f"Loading tagged subset for '{args.tag}'...")
    tagged_records = select_tagged_records(records, dataset_path, args.tag)
    if not tagged_records:
        raise ValueError(
            f"No records found with tag '{args.tag}' or in {args.tag}_manifest.json"
        )

    print(f"Loading LFMC targets from {annotation_geojson}...")
    lfmc_targets = load_lfmc_targets(annotation_geojson)
    train_windows = [record.window for record in records if record.split == "train"]
    lfmc_edges = lfmc_decile_edges(train_windows, lfmc_targets)
    reference_counts = count_lfmc_bins(train_windows, lfmc_targets, lfmc_edges)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    plot_original_map(axes[0, 0], records, states_gdf)
    plot_tagged_split_map(axes[0, 1], tagged_records, states_gdf, "train")
    plot_tagged_split_map(axes[0, 2], tagged_records, states_gdf, "val")
    plot_tagged_split_map(axes[0, 3], tagged_records, states_gdf, "test")

    plot_split_counts(axes[1, 0], tagged_records)
    plot_decile_bars(
        axes[1, 1],
        tagged_records,
        "train",
        lfmc_targets,
        lfmc_edges,
        reference_counts,
    )
    plot_decile_bars(
        axes[1, 2],
        tagged_records,
        "val",
        lfmc_targets,
        lfmc_edges,
        reference_counts,
    )
    plot_decile_bars(
        axes[1, 3],
        tagged_records,
        "test",
        lfmc_targets,
        lfmc_edges,
        reference_counts,
    )

    fig.suptitle(f"{os.path.basename(dataset_path)} - {args.tag}", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
