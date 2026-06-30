"""Visualize the MapBiomas subsample produced by sample_expert_points.py.

Generates a 2x3 figure:
  (0,0) Geographic scatter – train only
  (0,1) Geographic scatter – val only
  (0,2) CARTA_2 histogram (overlaid train/val)
  (1,0) Class distribution bar chart
  (1,1) Year distribution bar chart
  (1,2) DECLIVIDAD distribution bar chart
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MY_ROOT = Path(os.environ.get("MY_ROOT", "."))

SPLIT_COLORS = {"train": "#1f77b4", "val": "#ff7f0e"}


def main() -> None:
    """Visualize the MapBiomas 4k subsample."""
    parser = argparse.ArgumentParser(
        description="Visualize the MapBiomas 4k subsample."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(
            MY_ROOT
            / "rslearn_projects/rslp/mapbiomas/subsampling/sample_expert_points_4k.csv"
        ),
        help="Path to the subsample CSV produced by sample_expert_points.py.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="If set, save the figure to this path instead of showing it.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("MapBiomas Subsample Overview", fontsize=16, fontweight="bold")

    # --- (0,0) Geographic scatter – train ---
    ax = axes[0, 0]
    ax.scatter(
        train["LAT"],
        train["LON"],
        s=4,
        alpha=0.4,
        color=SPLIT_COLORS["train"],
        rasterized=True,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Train ({len(train)} pts)")

    # --- (0,1) Geographic scatter – val ---
    ax = axes[0, 1]
    ax.scatter(
        val["LAT"],
        val["LON"],
        s=4,
        alpha=0.4,
        color=SPLIT_COLORS["val"],
        rasterized=True,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Val ({len(val)} pts)")

    # Share axis limits between the two geo plots
    all_lon = df["LAT"]
    all_lat = df["LON"]
    lon_pad = (all_lon.max() - all_lon.min()) * 0.03
    lat_pad = (all_lat.max() - all_lat.min()) * 0.03
    for a in axes[0, :2]:
        a.set_xlim(all_lon.min() - lon_pad, all_lon.max() + lon_pad)
        a.set_ylim(all_lat.min() - lat_pad, all_lat.max() + lat_pad)

    # --- (0,2) CARTA_2 histogram ---
    ax = axes[0, 2]
    tiles_sorted = sorted(df["CARTA_2"].unique())
    tile_to_idx = {t: i for i, t in enumerate(tiles_sorted)}
    train_idx = train["CARTA_2"].map(tile_to_idx)
    val_idx = val["CARTA_2"].map(tile_to_idx)
    bins = np.linspace(-0.5, len(tiles_sorted) - 0.5, min(50, len(tiles_sorted)) + 1)
    ax.hist(train_idx, bins=bins, alpha=0.5, label="train", color=SPLIT_COLORS["train"])
    ax.hist(val_idx, bins=bins, alpha=0.5, label="val", color=SPLIT_COLORS["val"])
    ax.set_xlabel("CARTA_2 tile (index)")
    ax.set_ylabel("Count")
    ax.set_title(f"CARTA_2 Distribution ({len(tiles_sorted)} tiles)")
    ax.legend()

    # --- (1,0) Class distribution ---
    _grouped_bar(
        axes[1, 0],
        df,
        group_col="CLASS",
        title="Class Distribution",
        xlabel="Class",
        ylabel="Count",
    )

    # --- (1,1) Year distribution ---
    _grouped_bar(
        axes[1, 1],
        df,
        group_col="YEAR",
        title="Year Distribution",
        xlabel="Year",
        ylabel="Count",
    )

    # --- (1,2) DECLIVIDAD distribution ---
    _grouped_bar(
        axes[1, 2],
        df,
        group_col="DECLIVIDAD",
        title="DECLIVIDAD Distribution",
        xlabel="DECLIVIDAD",
        ylabel="Count",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if args.out_path:
        out = Path(args.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {out}")
    else:
        plt.show()


def _grouped_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    group_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """Draw a side-by-side bar chart of train vs val counts for *group_col*."""
    ct = df.groupby([group_col, "split"]).size().unstack(fill_value=0)
    for split in ["train", "val"]:
        if split not in ct.columns:
            ct[split] = 0
    ct = ct[["train", "val"]].sort_index()

    labels = [str(v) for v in ct.index]
    x = range(len(labels))
    bar_width = 0.4

    ax.bar(
        [i - bar_width / 2 for i in x],
        ct["train"],
        width=bar_width,
        label="train",
        color=SPLIT_COLORS["train"],
    )
    ax.bar(
        [i + bar_width / 2 for i in x],
        ct["val"],
        width=bar_width,
        label="val",
        color=SPLIT_COLORS["val"],
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()


if __name__ == "__main__":
    main()
