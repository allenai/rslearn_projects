#!/usr/bin/env python3
"""Visualize spatial split windows by plotting their bounds."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle

Bounds = tuple[float, float, float, float]
RectSpec = tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate metadata.json files under a directory and plot their spatial bounds."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory that contains window subfolders with metadata.json (e.g. spatial_split).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the resulting figure (PNG, PDF, etc.).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the saved figure when using --output. Defaults to 300.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot with matplotlib's GUI in addition to saving (if --output is used).",
    )
    return parser.parse_args()


def find_metadata_files(root: Path) -> Iterable[Path]:
    """Find all metadata.json files in the given root directory."""
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    return root.rglob("metadata.json")


def parse_bounds(bounds: Iterable[float]) -> Bounds:
    """Parse and sort a set of bounds into ordered coordinates."""
    x0, y0, x1, y1 = bounds
    xmin, xmax = sorted((x0, x1))
    ymin, ymax = sorted((y0, y1))
    return xmin, ymin, xmax, ymax


def collect_rectangles(
    metadata_paths: Iterable[Path],
) -> tuple[
    defaultdict[str, list[RectSpec]],
    Bounds,
    Counter,
]:
    """Collect rectangles and compute overall bounds from metadata paths."""
    rectangles: defaultdict[str, list[RectSpec]] = defaultdict(list)
    counts: Counter = Counter()

    x_min = math.inf
    y_min = math.inf
    x_max = -math.inf
    y_max = -math.inf

    total = 0

    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as fp:
            metadata: dict = json.load(fp)

        bounds = metadata.get("bounds")
        if not bounds or len(bounds) != 4:
            raise ValueError(f"Invalid bounds in {metadata_path}: {bounds}")

        xmin, ymin, xmax, ymax = parse_bounds(bounds)

        x_min = min(x_min, xmin)
        y_min = min(y_min, ymin)
        x_max = max(x_max, xmax)
        y_max = max(y_max, ymax)

        width = xmax - xmin
        height = ymax - ymin

        split = metadata.get("options", {}).get("split", "unknown")
        rectangles[split].append((xmin, ymin, width, height))
        counts[split] += 1
        total += 1

    if total == 0:
        raise RuntimeError("No metadata.json files were found.")

    return rectangles, (x_min, y_min, x_max, y_max), counts


def choose_colors(splits: Iterable[str]) -> dict[str, str]:
    """Assign colors to each split based on a predefined palette or fallback colors."""
    palette = {
        "train": "#1f77b4",  # Blue
        "val": "#ff7f0e",  # Orange
        "validation": "#ff7f0e",
        "dev": "#ff7f0e",
        "test": "#2ca02c",  # Green
        "unknown": "#7f7f7f",  # Grey
    }

    colors: dict[str, str] = {}
    fallback_colors = iter(
        [
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#17becf",  # Teal
        ]
    )

    for split in splits:
        key = split.lower()
        if key in palette:
            colors[split] = palette[key]
        else:
            colors[split] = next(fallback_colors, "#7f7f7f")

    return colors


def plot_rectangles(
    rectangles: defaultdict[str, list[RectSpec]],
    extents: Bounds,
    counts: Counter,
) -> plt.Figure:
    """Plot rectangles using Matplotlib."""
    x_min, y_min, x_max, y_max = extents
    width = x_max - x_min
    height = y_max - y_min

    aspect_ratio = height / width if width else 1.0
    fig_width = 12
    fig_height = max(6.0, fig_width * aspect_ratio)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    colors = choose_colors(rectangles.keys())
    legend_handles: list[Patch] = []

    for split, rect_specs in rectangles.items():
        patches = [Rectangle((x, y), w, h, linewidth=0.2) for x, y, w, h in rect_specs]
        collection = PatchCollection(
            patches,
            facecolor=colors[split],
            edgecolor="black",
            linewidth=0.1,
            alpha=0.6,
        )
        ax.add_collection(collection)
        legend_handles.append(
            Patch(
                facecolor=colors[split],
                edgecolor="black",
                label=f"{split} ({counts[split]})",
            )
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title("Spatial split windows by split label")
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    ax.grid(False)

    return fig


def main() -> None:
    """Main function to execute the script."""
    args = parse_args()

    metadata_paths = list(find_metadata_files(args.root))
    rectangles, extents, counts = collect_rectangles(metadata_paths)

    total = sum(counts.values())
    x_min, y_min, x_max, y_max = extents
    print(f"Found {total} windows in {args.root}")
    print(f"x bounds: {x_min} … {x_max}")
    print(f"y bounds: {y_min} … {y_max}")
    for split, count in counts.most_common():
        print(f"{split}: {count}")

    fig = plot_rectangles(rectangles, extents, counts)

    if args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    if args.show or not args.output:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
