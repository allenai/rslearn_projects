"""Visualize tagged subsampled locations vs all original locations on a US map.

Produces a 2x2 figure:
  - Row 1: Woody dataset
  - Row 2: Herbaceous dataset
  - Left column: all original locations
  - Right column: tagged subset only
Points are colored by split (train=blue, val=orange, test=green).

Usage:
    python visualize_oep_eval.py \
        --woody_path /path/to/woody/dataset \
        --herbaceous_path /path/to/herbaceous/dataset \
        --tag oep_eval \
        --output oep_eval_map.png
"""

import argparse
import json
import os
from collections import defaultdict

import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer

US_STATES_SHP = (
    "/weka/dfive-default/hadriens/datasets/Misc/Us states/cb_2018_us_state_20m.shp"
)

SPLIT_COLORS = {"train": "#2176AE", "val": "#F77F00", "test": "#06D6A0"}


def load_locations(dataset_path: str) -> dict[str, list[tuple[float, float]]]:
    """Load unique locations per split, return {split: [(lon, lat), ...]}."""
    windows_dir = os.path.join(dataset_path, "windows", "spatial_split")
    seen: dict[str, set[tuple]] = defaultdict(set)
    locations: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for name in os.listdir(windows_dir):
        meta_path = os.path.join(windows_dir, name, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        split = meta.get("options", {}).get("split", "unknown")
        crs = meta["projection"]["crs"]
        bounds = meta["bounds"]
        loc_key = (crs, tuple(bounds))

        if loc_key in seen[split]:
            continue
        seen[split].add(loc_key)

        x_res = meta["projection"]["x_resolution"]
        y_res = meta["projection"]["y_resolution"]
        cx = (bounds[0] + bounds[2]) / 2.0 * x_res
        cy = (bounds[1] + bounds[3]) / 2.0 * y_res
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(cx, cy)
        locations[split].append((lon, lat))

    return dict(locations)


def load_tagged_locations(
    dataset_path: str, tag: str
) -> dict[str, list[tuple[float, float]]]:
    """Load unique locations that have the given tag, per split."""
    windows_dir = os.path.join(dataset_path, "windows", "spatial_split")
    seen: dict[str, set[tuple]] = defaultdict(set)
    locations: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for name in os.listdir(windows_dir):
        meta_path = os.path.join(windows_dir, name, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        if tag not in meta.get("options", {}):
            continue

        split = meta.get("options", {}).get("split", "unknown")
        crs = meta["projection"]["crs"]
        bounds = meta["bounds"]
        loc_key = (crs, tuple(bounds))

        if loc_key in seen[split]:
            continue
        seen[split].add(loc_key)

        x_res = meta["projection"]["x_resolution"]
        y_res = meta["projection"]["y_resolution"]
        cx = (bounds[0] + bounds[2]) / 2.0 * x_res
        cy = (bounds[1] + bounds[3]) / 2.0 * y_res
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(cx, cy)
        locations[split].append((lon, lat))

    return dict(locations)


def plot_locations_on_ax(
    ax: plt.Axes,
    locations: dict[str, list[tuple[float, float]]],
    states_gdf: gpd.GeoDataFrame,
    title: str,
) -> None:
    """Plot location points on a US map axis."""
    states_gdf.boundary.plot(ax=ax, linewidth=0.5, color="gray")

    for split in ["test", "val", "train"]:
        if split not in locations:
            continue
        pts = locations[split]
        if not pts:
            continue
        lons, lats = zip(*pts)
        ax.scatter(
            lons,
            lats,
            c=SPLIT_COLORS[split],
            s=15,
            alpha=0.7,
            edgecolors="none",
            label=f"{split} ({len(pts)} locs)",
            zorder=3,
        )

    ax.set_xlim(-130, -65)
    ax.set_ylim(24, 50)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.8)


def main() -> None:
    """Visualize tagged subsampled vs original locations on a US map."""
    parser = argparse.ArgumentParser(description="Visualize tagged locations")
    parser.add_argument("--woody_path", type=str, required=True)
    parser.add_argument("--herbaceous_path", type=str, required=True)
    parser.add_argument(
        "--tag", type=str, default="oep_eval", help="Tag name to filter on"
    )
    parser.add_argument("--states_shp", type=str, default=US_STATES_SHP)
    parser.add_argument("--output", type=str, default="oep_eval_map.png")
    args = parser.parse_args()

    print("Loading US states shapefile...")
    states_gdf = gpd.read_file(args.states_shp)
    states_gdf = states_gdf[~states_gdf["NAME"].isin(["Alaska", "Hawaii"])]
    states_gdf = states_gdf.to_crs("EPSG:4326")

    print("Loading woody locations...")
    woody_all = load_locations(args.woody_path)
    woody_tagged = load_tagged_locations(args.woody_path, args.tag)

    print("Loading herbaceous locations...")
    herb_all = load_locations(args.herbaceous_path)
    herb_tagged = load_tagged_locations(args.herbaceous_path, args.tag)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    plot_locations_on_ax(axes[0, 0], woody_all, states_gdf, "Woody — All Locations")
    plot_locations_on_ax(
        axes[0, 1], woody_tagged, states_gdf, f"Woody — {args.tag} Subset"
    )
    plot_locations_on_ax(axes[1, 0], herb_all, states_gdf, "Herbaceous — All Locations")
    plot_locations_on_ax(
        axes[1, 1], herb_tagged, states_gdf, f"Herbaceous — {args.tag} Subset"
    )

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
