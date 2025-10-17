#!/usr/bin/env python3
"""Plot train/val point distribution from window metadata.

Assumptions:
- Directory structure: /weka/.../dataset/windows/spatial_split/task_*/metadata.json
- Each metadata.json contains:
  {
    "group": "spatial_split",
    "name": "...",
    "projection": {"crs": "EPSG:32636", "x_resolution": 10.0, "y_resolution": -10.0},
    "bounds": [xmin, ymin, xmax, ymax],
    "time_range": [..., ...],
    "options": {"split": "train", ...}
  }

Outputs:
- points.csv (x, y, split, name, group, crs)
- scatter.png (colored by split, equal aspect, gridlines)
- grid_heatmap_<split>.png (2D gridded counts per split)
"""

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_points(base_dir: Path) -> pd.DataFrame:
    """Load point data from metadata files.

    Args:
        base_dir (Path): Base directory containing task_* subfolders with metadata.json.

    Returns:
        pd.DataFrame: Dataframe containing points with metadata.
    """
    rows: list[dict] = []
    # Walk only one level (task_* folders) to be fast and predictable
    for entry in os.scandir(base_dir):
        if not entry.is_dir() or not entry.name.startswith("task_"):
            continue
        meta_path = Path(entry.path) / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {meta_path}: {e}")
            continue

        bounds = meta.get("bounds", None)
        if not bounds or len(bounds) != 4:
            print(f"[WARN] Bad bounds in {meta_path}")
            continue

        xmin, ymin, xmax, ymax = bounds
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        proj = meta.get("projection", {})
        crs = proj.get("crs", "UNKNOWN")
        split = meta.get("options", {}).get("split", "unknown")
        group = meta.get("group", "")
        name = meta.get("name", entry.name)

        rows.append(
            {
                "x": float(cx),
                "y": float(cy),
                "split": str(split),
                "name": str(name),
                "group": str(group),
                "crs": str(crs),
                "src": str(meta_path),
            }
        )

    if not rows:
        raise RuntimeError(f"No metadata points found under {base_dir}")

    df = pd.DataFrame(rows)
    return df


def snap_to_grid(df: pd.DataFrame, grid_size: float) -> pd.DataFrame:
    """Snap coordinates in the dataframe to a specified grid size.

    Args:
        df (pd.DataFrame): Dataframe containing x, y coordinates.
        grid_size (float): Size of the grid to snap points to.

    Returns:
        pd.DataFrame: A new dataframe with additional snapped grid columns.
    """
    gx = np.floor(df["x"].to_numpy() / grid_size) * grid_size
    gy = np.floor(df["y"].to_numpy() / grid_size) * grid_size
    out = df.copy()
    out["gx"] = gx
    out["gy"] = gy
    out["grid_id"] = (
        out["gx"].astype(np.int64).astype(str)
        + "_"
        + out["gy"].astype(np.int64).astype(str)
    )
    return out


def plot_scatter(df: pd.DataFrame, out_path: Path, grid_size: float | None) -> None:
    """Create a scatter plot of points colored by split.

    Args:
        df (pd.DataFrame): Dataframe containing coordinates and splits.
        out_path (Path): Path to save the scatter plot.
        grid_size (float | None): Grid size for optional overlay.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    # Plot each split as a separate layer
    for split, sub in df.groupby("split"):
        ax.scatter(
            sub["x"].to_numpy(),
            sub["y"].to_numpy(),
            s=10,
            alpha=0.7,
            label=split,
            edgecolors="none",
        )

    ax.set_aspect("equal")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    crs_set = sorted(df["crs"].unique())
    crs_note = ", ".join(crs_set)
    ax.set_title(f"Point centroids by split (CRS: {crs_note})")

    # Optional grid overlay
    if grid_size and grid_size > 0:
        xmin, xmax = df["x"].min(), df["x"].max()
        ymin, ymax = df["y"].min(), df["y"].max()
        # Expand slightly to show border lines cleanly
        pad = 0.01
        xr = xmax - xmin
        yr = ymax - ymin
        xmin -= pad * xr
        xmax += pad * xr
        ymin -= pad * yr
        ymax += pad * yr
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # Draw grid lines
        x_ticks = np.arange(
            math.floor(xmin / grid_size) * grid_size,
            math.ceil(xmax / grid_size) * grid_size + grid_size,
            grid_size,
        )
        y_ticks = np.arange(
            math.floor(ymin / grid_size) * grid_size,
            math.ceil(ymax / grid_size) * grid_size + grid_size,
            grid_size,
        )
        for xt in x_ticks:
            ax.axvline(xt, linewidth=0.5, alpha=0.2)
        for yt in y_ticks:
            ax.axhline(yt, linewidth=0.5, alpha=0.2)

        ax.text(
            0.02,
            0.98,
            f"Grid size: {grid_size:g} m",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

    ax.legend(title="Split", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_grid_heatmaps(df: pd.DataFrame, out_dir: Path, grid_size: float) -> None:
    """Generate heatmaps for each dataset split showing grid counts.

    Args:
        df (pd.DataFrame): Dataframe containing grid and split information.
        out_dir (Path): Directory to save heatmap images.
        grid_size (float): Size of the grid cells for heatmaps.
    """
    if grid_size <= 0:
        print("[INFO] Skipping grid heatmaps because grid_size <= 0")
        return

    if "gx" not in df.columns or "gy" not in df.columns:
        df = snap_to_grid(df, grid_size)

    for split, sub in df.groupby("split"):
        g = sub.groupby(["gx", "gy"]).size().reset_index(name="count")

        x_vals = np.sort(g["gx"].unique())
        y_vals = np.sort(g["gy"].unique())

        count_grid = np.zeros((len(y_vals), len(x_vals)), dtype=float)

        x_index = {v: i for i, v in enumerate(x_vals)}
        y_index = {v: i for i, v in enumerate(y_vals)}
        for _, row in g.iterrows():
            count_grid[y_index[row["gy"]], x_index[row["gx"]]] = row["count"]

        x_edges = np.append(x_vals, x_vals[-1] + grid_size)
        y_edges = np.append(y_vals, y_vals[-1] + grid_size)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        mesh = ax.pcolormesh(x_edges, y_edges, count_grid, shading="auto")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Count")
        ax.set_aspect("equal")
        ax.set_xlabel("X (meters)")
        ax.set_ylabel("Y (meters)")
        crs_set = sorted(df["crs"].unique())
        crs_note = ", ".join(crs_set)
        ax.set_title(
            f"Gridded point counts ({split}) — grid {grid_size:g} m (CRS: {crs_note})"
        )
        fig.tight_layout()
        out_path = out_dir / f"grid_heatmap_{split}.png"
        fig.savefig(out_path)
        plt.close(fig)


def main() -> None:
    """Main function to execute the script for plotting spatial splits."""
    ap = argparse.ArgumentParser(
        description="Plot train/val point distribution from metadata.json windows."
    )
    ap.add_argument(
        "--base",
        type=Path,
        default=Path(
            "/weka/dfive-default/yawenz/datasets/scratch_ft_v3/dataset/windows/spatial_split/"
        ),
        help="Base directory containing task_* subfolders with metadata.json",
    )
    ap.add_argument(
        "--grid-size",
        type=float,
        default=1280.0,
        help="Grid size in meters for visualization and heatmaps (use 0 to disable grid overlays)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("./point_split_plots"),
        help="Output directory for figures and CSV",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning: {args.base}")
    df = load_points(args.base)

    csv_path = args.out / "points.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Wrote {csv_path}")

    df_grid = snap_to_grid(df, args.grid_size) if args.grid_size > 0 else df

    scatter_path = args.out / "scatter.png"
    plot_scatter(df_grid, scatter_path, args.grid_size)
    print(f"[INFO] Wrote {scatter_path}")

    plot_grid_heatmaps(df_grid, args.out, args.grid_size)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
