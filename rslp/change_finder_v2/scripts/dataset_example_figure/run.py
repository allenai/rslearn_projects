"""Render an LCMonitor dataset example figure for a single annotated window.

The figure shows the pre-change and post-change Sentinel-2 mosaics side by side,
with two callout "dialog" boxes on the right:

- a "Change point" box (green) pointing to the annotated positive point, listing
  the source/destination land-cover categories and the pre-change /
  first-observable / post-change dates; and
- a "No-change point" box (red) pointing to a stable negative point, listing the
  time range over which no land-cover change occurred.

Window imagery is read from the ``sentinel2_quarterly`` layer of the lcc_model
rslearn dataset. The annotation metadata (categories, dates, negative points)
comes from the v2 annotation JSONs, keyed by ``"{group}/{window_name}"``.

For each side we pick the clearest available quarterly mosaic (lowest
nodata/cloud score) among the quarters bracketing the change, and the negative
point is chosen to be the most temporally stable in-bounds candidate that is not
too close to the positive point or the image edge.

Example:
    python -m rslp.change_finder_v2.scripts.dataset_example_figure.run \
        --dataset-path /weka/.../lcc_model_dataset_20260616 \
        --annotation-dir /weka/.../backup/20260615 \
        --window-key "urban_01/columbia_sc_EPSG:32617_48896_-377600" \
        --output-path /tmp/lcmonitor_example.pdf
"""

import argparse
import glob
import json
import math
import os
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
from matplotlib.patches import Circle, ConnectionPatch  # noqa: E402
from rslearn.dataset import Dataset  # noqa: E402
from rslearn.utils.geometry import WGS84_PROJECTION, STGeometry  # noqa: E402
from upath import UPath  # noqa: E402

# Default data locations (override on the command line as needed).
DEFAULT_DATASET_PATH = (
    "/weka/dfive-default/rslearn-eai/datasets/change_finder/"
    "lcc_model_dataset_20260616"
)
DEFAULT_ANNOTATION_DIR = (
    "/weka/dfive-default/rslearn-eai/datasets/change_finder/backup/20260615"
)
DEFAULT_WINDOW_KEY = "urban_01/columbia_sc_EPSG:32617_48896_-377600"

WINDOW_SIZE = 128
LAYER = "sentinel2_quarterly"
POSITIVE_COLOR = "#1faa1f"
NEGATIVE_COLOR = "#e11919"


def parse_date(s: str) -> datetime:
    """Parse an ISO date string into a UTC datetime."""
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def fdate(d: datetime) -> str:
    """Format a datetime as YYYY-MM-DD."""
    return d.strftime("%Y-%m-%d")


def mdate(d: datetime) -> str:
    """Format a datetime as e.g. "Feb 2021"."""
    return d.strftime("%b %Y")


def cap_cat(s: str) -> str:
    """Capitalize each space-separated word, preserving hyphens.

    e.g. "urban/built-up" -> "Urban/Built-up", "bare" -> "Bare".
    """
    return "/".join(
        " ".join(w[:1].upper() + w[1:] for w in part.split(" "))
        for part in s.split("/")
    )


def load_annotations(annotation_dir: str) -> dict[str, dict]:
    """Load v2 annotation entries keyed by "{group}/{window_name}"."""
    entries: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(annotation_dir, "annotations_*.json"))):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for entry in data:
            group = entry.get("group")
            window_name = entry.get("window_name")
            if group and window_name:
                entries[f"{group}/{window_name}"] = entry
    return entries


class WindowReader:
    """Reads RGB mosaics and computes mosaic quality for one rslearn dataset."""

    def __init__(self, dataset_path: str) -> None:
        self.dataset = Dataset(UPath(dataset_path))
        self.band_set = self.dataset.layers[LAYER].band_sets[0]
        self.raster_format = self.band_set.instantiate_raster_format()
        self.band_index = {b: i for i, b in enumerate(self.band_set.bands)}

    def read_chw(self, window, group_idx: int) -> np.ndarray:
        """Read one quarterly mosaic as a (C, H, W) float32 array."""
        arr = window.data.read_raster(
            LAYER, self.band_set.bands, self.raster_format, group_idx=group_idx
        )
        return arr.array[:, 0, :, :].astype(np.float32)

    def quality(self, chw: np.ndarray) -> float:
        """Lower is better: penalizes nodata and bright (cloudy) pixels."""
        blue = chw[self.band_index["B02"]]
        green = chw[self.band_index["B03"]]
        red = chw[self.band_index["B04"]]
        nodata = float(np.mean((blue + green + red) < 1))
        cloud = float(np.mean((blue > 2200) & (green > 2200) & (red > 2200)))
        return nodata * 2 + cloud

    def pick_clear(self, window, candidates: list[int]):
        """Return (group_idx, chw) for the clearest candidate mosaic."""
        best = None
        best_score = float("inf")
        for group_idx in candidates:
            try:
                chw = self.read_chw(window, group_idx)
            except Exception:
                continue
            score = self.quality(chw)
            if score < best_score:
                best_score = score
                best = (group_idx, chw)
        return best

    def stretch_rgb(self, chw: np.ndarray, lo_p: float = 2, hi_p: float = 98) -> np.ndarray:
        """Percentile-stretch the RGB bands of a (C, H, W) mosaic to [0, 1]."""
        rgb = np.stack(
            [
                chw[self.band_index["B04"]],
                chw[self.band_index["B03"]],
                chw[self.band_index["B02"]],
            ],
            axis=-1,
        )
        lo = np.percentile(rgb, lo_p)
        hi = np.percentile(rgb, hi_p)
        return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)


def build_figure(reader: WindowReader, entry: dict, window_key: str):
    """Build the example figure for a single annotated window."""
    positive = None
    for point in entry.get("positive_points", []):
        if all(
            point.get(k)
            for k in [
                "pre_change",
                "post_change",
                "first_date_change_noticeable",
                "pre_category",
                "post_category",
            ]
        ):
            positive = point
            break
    if positive is None:
        raise ValueError(f"no fully-annotated positive point for {window_key}")

    group, name = window_key.split("/", 1)
    window = reader.dataset.load_windows(groups=[group], names=[name])[0]
    layer_data = window.load_layer_datas().get(LAYER)
    time_ranges = layer_data.group_time_ranges or []

    pre_change = parse_date(positive["pre_change"])
    post_change = parse_date(positive["post_change"])
    first_obs = parse_date(positive["first_date_change_noticeable"])

    pre_candidates = [
        i for i, tr in enumerate(time_ranges) if tr and tr[1] <= pre_change
    ][-5:]
    post_candidates = [
        i for i, tr in enumerate(time_ranges) if tr and tr[0] >= post_change
    ][:8]
    pre_pick = reader.pick_clear(window, pre_candidates)
    post_pick = reader.pick_clear(window, post_candidates)
    if pre_pick is None or post_pick is None:
        raise ValueError(f"no clear mosaic found for {window_key}")
    pre_idx, pre_chw = pre_pick
    post_idx, post_chw = post_pick

    pre_mid = time_ranges[pre_idx][0] + (time_ranges[pre_idx][1] - time_ranges[pre_idx][0]) / 2
    post_mid = time_ranges[post_idx][0] + (time_ranges[post_idx][1] - time_ranges[post_idx][0]) / 2
    pre_rgb = reader.stretch_rgb(pre_chw)
    post_rgb = reader.stretch_rgb(post_chw)

    def lonlat_to_pixel(lon: float, lat: float) -> tuple[int, int]:
        st = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), time_range=None)
        projected = st.to_projection(window.projection)
        return (
            math.floor(projected.shp.x) - window.bounds[0],
            math.floor(projected.shp.y) - window.bounds[1],
        )

    pos_col, pos_row = lonlat_to_pixel(positive["lon"], positive["lat"])

    in_negatives: list[tuple[int, int]] = []
    for neg in entry.get("negative_points", []):
        col, row = lonlat_to_pixel(neg["lon"], neg["lat"])
        if 0 <= col < WINDOW_SIZE and 0 <= row < WINDOW_SIZE:
            in_negatives.append((col, row))
    if not in_negatives:
        raise ValueError(f"no in-bounds negative point for {window_key}")

    def local_change(col: int, row: int, k: int = 5) -> float:
        a = pre_chw[:, max(0, row - k) : row + k + 1, max(0, col - k) : col + k + 1]
        b = post_chw[:, max(0, row - k) : row + k + 1, max(0, col - k) : col + k + 1]
        return float(np.mean(np.abs(a - b)))

    def neg_score(candidate: tuple[int, int]) -> float:
        col, row = candidate
        dist_pos = math.hypot(col - pos_col, row - pos_row)
        edge = min(col, row, WINDOW_SIZE - col, WINDOW_SIZE - row)
        penalty = 0.0
        if dist_pos < 28:
            penalty += (28 - dist_pos) * 200
        if edge < 14:
            penalty += (14 - edge) * 200
        return local_change(col, row) + penalty

    neg_col, neg_row = min(in_negatives, key=neg_score)

    src_cat = positive["pre_category"]
    dst_cat = positive["post_category"]
    neg_tr = entry.get("time_range")
    neg_start = parse_date(neg_tr[0])
    neg_end = parse_date(neg_tr[1])

    # --- plot ---
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.4, 3.1), dpi=300)
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.18],
        wspace=0.05,
        left=0.008,
        right=0.992,
        top=0.90,
        bottom=0.02,
    )

    def draw_panel(ax, rgb: np.ndarray, title: str) -> None:
        ax.imshow(rgb, interpolation="lanczos", extent=[0, WINDOW_SIZE, WINDOW_SIZE, 0])
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        for (cc, rr), color in [
            ((neg_col, neg_row), NEGATIVE_COLOR),
            ((pos_col, pos_row), POSITIVE_COLOR),
        ]:
            ax.add_patch(Circle((cc, rr), 6.5, fill=False, ec="white", lw=2.8))
            ax.add_patch(Circle((cc, rr), 6.5, fill=False, ec=color, lw=1.7))

    ax_before = fig.add_subplot(gs[0, 0])
    draw_panel(ax_before, pre_rgb, f"Before  ({mdate(pre_mid)})")
    ax_after = fig.add_subplot(gs[0, 1])
    draw_panel(ax_after, post_rgb, f"After  ({mdate(post_mid)})")
    ax_dialog = fig.add_subplot(gs[0, 2])
    ax_dialog.axis("off")
    ax_dialog.set_xlim(0, 1)
    ax_dialog.set_ylim(0, 1)

    pos_body = (
        f"{cap_cat(src_cat)} $\\rightarrow$ {cap_cat(dst_cat)}\n"
        f"Pre-change:  {fdate(pre_change)}\n"
        f"First-observable:  {fdate(first_obs)}\n"
        f"Post-change:  {fdate(post_change)}"
    )
    neg_body = f"No land-cover change\nfrom {fdate(neg_start)} to {fdate(neg_end)}"

    pos_anchor = (0.10, 0.86)
    neg_anchor = (0.10, 0.30)
    ax_dialog.text(
        pos_anchor[0],
        pos_anchor[1],
        r"$\bf{Change\ point}$" + "\n" + pos_body,
        transform=ax_dialog.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        color="#111111",
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=POSITIVE_COLOR, lw=1.6),
    )
    ax_dialog.text(
        neg_anchor[0],
        neg_anchor[1],
        r"$\bf{No\text{-}change\ point}$" + "\n" + neg_body,
        transform=ax_dialog.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        color="#111111",
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=NEGATIVE_COLOR, lw=1.6),
    )

    # Arrows from the points (in the After panel) to the dialog boxes.
    con_pos = ConnectionPatch(
        xyA=(pos_col, pos_row),
        coordsA=ax_after.transData,
        xyB=(pos_anchor[0] - 0.02, pos_anchor[1] - 0.10),
        coordsB=ax_dialog.transAxes,
        arrowstyle="-|>",
        color=POSITIVE_COLOR,
        lw=1.6,
        mutation_scale=13,
        shrinkA=5,
        shrinkB=2,
        connectionstyle="arc3,rad=-0.15",
    )
    con_neg = ConnectionPatch(
        xyA=(neg_col, neg_row),
        coordsA=ax_after.transData,
        xyB=(neg_anchor[0] - 0.02, neg_anchor[1] - 0.07),
        coordsB=ax_dialog.transAxes,
        arrowstyle="-|>",
        color=NEGATIVE_COLOR,
        lw=1.6,
        mutation_scale=13,
        shrinkA=5,
        shrinkB=2,
        connectionstyle="arc3,rad=0.15",
    )
    fig.add_artist(con_pos)
    fig.add_artist(con_neg)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-path",
        default=DEFAULT_DATASET_PATH,
        help="Path to the lcc_model rslearn dataset.",
    )
    parser.add_argument(
        "--annotation-dir",
        default=DEFAULT_ANNOTATION_DIR,
        help="Directory containing the annotations_*.json v2 annotation files.",
    )
    parser.add_argument(
        "--window-key",
        default=DEFAULT_WINDOW_KEY,
        help='Window to render, as "{group}/{window_name}".',
    )
    parser.add_argument(
        "--output-path",
        default="/tmp/lcmonitor_example.pdf",
        help="Output figure path (.pdf or .png).",
    )
    args = parser.parse_args()

    annotations = load_annotations(args.annotation_dir)
    if args.window_key not in annotations:
        raise SystemExit(f"window key not found in annotations: {args.window_key}")

    reader = WindowReader(args.dataset_path)
    fig = build_figure(reader, annotations[args.window_key], args.window_key)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved {args.output_path}")


if __name__ == "__main__":
    main()
