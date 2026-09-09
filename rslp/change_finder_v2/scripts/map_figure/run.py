"""Plot the windows in the LCC model rslearn dataset on a global map.

Each window is drawn as a point at its geographic centroid (WGS84), colored by
the annotation phase it belongs to. The on-disk window groups are collapsed into
a smaller set of annotation phases (see ``PHASES``), and the legend lists each
phase label together with the number of windows in it.

Window centroids are computed in parallel: the (potentially many) window
``metadata.json`` files are loaded with a multiprocessing pool (``--workers``,
default 64), since the per-window metadata reads dominate the runtime.

Example:
    python -m rslp.change_finder_v2.scripts.map_figure.run \
        --dataset-path /weka/.../lcc_model_dataset_20260616 \
        --output-path /tmp/lcc_dataset_map.pdf
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from rslearn.dataset import Dataset  # noqa: E402
from upath import UPath  # noqa: E402

# Default location of the lcc_model rslearn dataset.
DEFAULT_DATASET_PATH = (
    "/weka/dfive-default/rslearn-eai/datasets/change_finder/"
    "lcc_model_dataset_20260616"
)
DEFAULT_OUTPUT_PATH = "/tmp/lcc_dataset_map.pdf"

# Annotation phases, in legend order. Each phase collapses one or more on-disk
# window groups, and is drawn with a distinct color. Note the round-3 group is
# stored on disk as "annotation_phase3" (not "phase3").
PHASES: list[tuple[str, list[str], str]] = [
    (
        "Initial Annotation",
        ["initial_seed0_100k", "initial_seed1_500k", "phase6", "urban_01", "phase4"],
        "#1f77b4",
    ),
    ("Round 2", ["phase2"], "#ff7f0e"),
    ("Round 3", ["phase8"], "#2ca02c"),
    ("Round 4", ["phase5"], "#d62728"),
    ("Round 5", ["phase7"], "#9467bd"),
    ("Round 6", ["annotation_phase3"], "#8c564b"),
]


def collect_phase_points(
    dataset: Dataset, groups: list[str], workers: int
) -> list[tuple[float, float]]:
    """Return the (lon, lat) centroids of all windows in the given groups.

    Args:
        dataset: the loaded rslearn dataset.
        groups: the on-disk window group names to load.
        workers: number of multiprocessing workers for loading window metadata.

    Returns:
        a list of (longitude, latitude) centroids in WGS84 degrees.
    """
    windows = dataset.load_windows(groups=groups, workers=workers, show_progress=True)
    points = []
    for window in windows:
        centroid = window.get_geometry().to_wgs84().shp.centroid
        points.append((centroid.x, centroid.y))
    return points


def build_figure(
    phase_points: list[tuple[str, str, list[tuple[float, float]]]],
) -> plt.Figure:
    """Build the global map figure.

    Args:
        phase_points: list of (label, color, points) tuples in legend order,
            where points is a list of (lon, lat) centroids.

    Returns:
        the matplotlib Figure.
    """
    fig = plt.figure(figsize=(14, 7))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f2f2f0")
    ax.add_feature(cfeature.OCEAN, facecolor="#dfeaf2")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#888888")
    ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#bbbbbb")

    legend_handles = []
    for label, color, points in phase_points:
        if points:
            lons, lats = zip(*points)
            ax.scatter(
                lons,
                lats,
                s=10,
                color=color,
                alpha=1.0,
                edgecolors="none",
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=6,
                markerfacecolor=color,
                markeredgecolor="none",
                label=f"{label} (n={len(points)})",
            )
        )

    total = sum(len(points) for _, _, points in phase_points)
    ax.set_title(f"LCC model dataset windows by annotation phase (n={total})")
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        fontsize=9,
        title="Annotation phase",
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Path to the lcc_model rslearn dataset.",
    )
    parser.add_argument(
        "--output-path",
        default=DEFAULT_OUTPUT_PATH,
        help="Output figure path (always written as PDF).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of multiprocessing workers for loading window metadata.",
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.dataset_path))

    phase_points = []
    for label, groups, color in PHASES:
        points = collect_phase_points(dataset, groups, args.workers)
        print(f"{label}: {len(points)} windows")
        phase_points.append((label, color, points))

    fig = build_figure(phase_points)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    fig.savefig(args.output_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved {args.output_path}")


if __name__ == "__main__":
    main()
