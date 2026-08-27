"""Render "land-cover map deficiency" false-positive figures for the paper.

Each figure highlights an evaluation point where the ESA WorldCover land-cover
*segmentation* model (run per-year as a change detector) produces a high change score
but is WRONG -- the ground truth is no-change -- while our LCMonitor model correctly
predicts no change. These are typically seasonal / phenological confusions (e.g. a
crop field read as "crops" one year and "grassland" the next), which a single
land-cover map cannot disambiguate from real land-cover change.

For each candidate the figure shows a 2x2 grid:

    pre-change Sentinel-2  |  pre-change WorldCover land cover
    post-change Sentinel-2 |  post-change WorldCover land cover

so the reader can see that the imagery shows the same stable land cover across years
while WorldCover's per-year class flips.

Candidates are selected reproducibly by joining the two standardized method CSVs
(``worldcover.csv`` and ``lcc_model.csv`` written by the evaluation flow) and keeping
points where: ground truth = no-change, WorldCover predicts change (ranked by its
change score, descending), and LCMonitor predicts no change. The imagery and land-cover
rasters are read from the materialized WorldCover eval dataset
(``worldcover_rslearn_dataset``), whose windows are named ``eval_{row_index:06d}_src``
and ``eval_{row_index:06d}_dst``.

Example (run on the WEKA/Beaker node, where the dataset path is mounted):

    python -m rslp.change_finder_v2.scripts.land_cover_fp_figure.run \
        --eval-dir /weka/.../change_finder/evaluation \
        --ds-path  /weka/.../change_finder/evaluation/worldcover_rslearn_dataset \
        --output-dir /data/favyenb/vis/land_cover_fp \
        --top 10
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import shapely  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry  # noqa: E402
from rslearn.utils.raster_format import get_bandset_dirname  # noqa: E402

from rslp.change_finder_v2.evaluation.worldcover.predict_change import (  # noqa: E402
    OUTPUT_BANDS,
    OUTPUT_LAYER,
)
from rslp.change_finder_v2.lcc_model.postprocess import (  # noqa: E402
    LC_CLASS_NAMES,
    NUM_LC_CLASSES,
)

# Default data locations (override on the command line as needed).
DEFAULT_EVAL_DIR = "/weka/dfive-default/rslearn-eai/datasets/change_finder/evaluation"
DEFAULT_DS_PATH = (
    "/weka/dfive-default/rslearn-eai/datasets/change_finder/evaluation/"
    "worldcover_rslearn_dataset"
)
DEFAULT_OUTPUT_DIR = "/data/favyenb/vis/land_cover_fp"

PREDICTION_GROUP = "predict"

# Sentinel-2 band order in the materialized dataset (see config_predict.json).
S2_BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B11", "B12",
]
S2_BANDSET_DIR = get_bandset_dirname(S2_BANDS)
OUTPUT_BANDSET_DIR = get_bandset_dirname(OUTPUT_BANDS)

# RGB band indices (0-based) within S2_BANDS.
RGB_IDX = (S2_BANDS.index("B04"), S2_BANDS.index("B03"), S2_BANDS.index("B02"))

# Land-cover palette aligned to LC_CLASS_NAMES (index 0 = nodata). RGB in [0, 1].
LC_PALETTE = (
    np.array(
        [
            (255, 255, 255),  # 0  nodata
            (214, 194, 156),  # 1  bare
            (60, 60, 60),     # 2  burnt
            (245, 205, 55),   # 3  crops
            (205, 185, 120),  # 4  fallow/shifting cultivation
            (170, 215, 120),  # 5  grassland
            (190, 210, 180),  # 6  Lichen and moss
            (140, 170, 90),   # 7  shrub
            (235, 240, 250),  # 8  snow and ice
            (40, 115, 50),    # 9  tree
            (200, 85, 75),    # 10 urban/built-up
            (60, 120, 200),   # 11 water
            (95, 180, 170),   # 12 wetland (herbaceous)
        ],
        dtype=np.float32,
    )
    / 255.0
)


def _as_bool(value: str) -> bool:
    """Parse a CSV cell into a bool."""
    return str(value).strip().lower() in ("true", "1")


def _load_csv_by_row_index(path: str) -> dict[str, dict[str, str]]:
    """Load a standardized method CSV keyed by its ``row_index`` column."""
    with open(path, newline="") as f:
        return {r["row_index"]: r for r in csv.DictReader(f)}


def select_candidates(eval_dir: str, top: int) -> list[dict[str, str]]:
    """Pick the top-N WorldCover false positives that LCMonitor gets right.

    Keeps points with ground-truth no-change where WorldCover predicts change and
    LCMonitor predicts no change, ranked by WorldCover change score (descending).
    Returns the WorldCover rows (which carry lon/lat/years and WorldCover's predicted
    categories), each augmented with ``lcc_change_score``.
    """
    wc = _load_csv_by_row_index(os.path.join(eval_dir, "worldcover.csv"))
    lcc = _load_csv_by_row_index(os.path.join(eval_dir, "lcc_model.csv"))

    candidates: list[dict[str, str]] = []
    for row_index, wc_row in wc.items():
        lcc_row = lcc.get(row_index)
        if lcc_row is None:
            continue
        if not (_as_bool(wc_row["has_prediction"]) and _as_bool(lcc_row["has_prediction"])):
            continue
        if _as_bool(wc_row["has_changed"]):
            continue  # want ground-truth no-change (seasonal-variation false positive)
        if not _as_bool(wc_row["predicted_changed"]):
            continue  # WorldCover must (wrongly) say change
        if _as_bool(lcc_row["predicted_changed"]):
            continue  # LCMonitor must (correctly) say no change
        augmented = dict(wc_row)
        augmented["lcc_change_score"] = lcc_row["change_score"]
        candidates.append(augmented)

    candidates.sort(key=lambda r: -float(r["change_score"]))
    return candidates[:top]


def _window_dir(ds_path: str, row_index: int, kind: str) -> str:
    """Path to the ``_src``/``_dst`` window for an eval point."""
    return os.path.join(
        ds_path, "windows", PREDICTION_GROUP, f"eval_{row_index:06d}_{kind}"
    )


def _mosaic_rgb(window_dir: str, mosaic_dir: str) -> np.ndarray | None:
    """Read one Sentinel-2 mosaic's RGB as a (H, W, 3) float32 array, or None."""
    tif = os.path.join(window_dir, "layers", mosaic_dir, S2_BANDSET_DIR, "geotiff.tif")
    if not os.path.exists(tif):
        return None
    with rasterio.open(tif) as ds:
        arr = ds.read(indexes=[i + 1 for i in RGB_IDX]).astype(np.float32)
    return np.transpose(arr, (1, 2, 0))  # (H, W, 3)


def _mosaic_quality(rgb: np.ndarray) -> float:
    """Lower is better: penalizes nodata and bright (cloudy) pixels."""
    total = rgb.sum(axis=-1)
    nodata = float(np.mean(total < 1))
    cloud = float(np.mean(np.all(rgb > 2200, axis=-1)))
    return nodata * 2 + cloud


def _clearest_rgb(window_dir: str) -> np.ndarray | None:
    """Pick the clearest Sentinel-2 mosaic for a window and percentile-stretch it."""
    mosaic_dirs = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(window_dir, "layers", "sentinel2*"))
        if os.path.isdir(p)
    )
    best: np.ndarray | None = None
    best_score = float("inf")
    for mosaic_dir in mosaic_dirs:
        rgb = _mosaic_rgb(window_dir, mosaic_dir)
        if rgb is None:
            continue
        score = _mosaic_quality(rgb)
        if score < best_score:
            best_score = score
            best = rgb
    if best is None:
        return None
    lo, hi = np.percentile(best, 2), np.percentile(best, 98)
    return np.clip((best - lo) / max(hi - lo, 1e-6), 0, 1)


def _lc_class_map(window_dir: str) -> np.ndarray | None:
    """Read the WorldCover output raster and return a (H, W) argmax class map."""
    tif = os.path.join(window_dir, "layers", OUTPUT_LAYER, OUTPUT_BANDSET_DIR, "geotiff.tif")
    if not os.path.exists(tif):
        return None
    with rasterio.open(tif) as ds:
        arr = ds.read().astype(np.float32)
    return arr[:NUM_LC_CLASSES].argmax(axis=0)


def _point_pixel(window_dir: str, lon: float, lat: float) -> tuple[int, int] | None:
    """Convert the annotated lon/lat to a (col, row) pixel within the window."""
    metadata_path = os.path.join(window_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path) as f:
        metadata = json.load(f)
    projection = Projection.deserialize(metadata["projection"])
    bounds = metadata["bounds"]
    projected = STGeometry(
        WGS84_PROJECTION, shapely.Point(lon, lat), None
    ).to_projection(projection)
    col = math.floor(projected.shp.x) - bounds[0]
    row = math.floor(projected.shp.y) - bounds[1]
    return col, row


def _cap(name: str) -> str:
    """Capitalize a category name for display."""
    return name[:1].upper() + name[1:]


def render_candidate(
    ds_path: str, candidate: dict[str, str], output_path: str
) -> bool:
    """Render one candidate's 2x2 figure. Returns False if imagery is missing."""
    row_index = int(candidate["row_index"])
    lon, lat = float(candidate["lon"]), float(candidate["lat"])
    src_year, dst_year = candidate["src_year"], candidate["dst_year"]
    wc_score = float(candidate["change_score"])
    lcc_score = float(candidate["lcc_change_score"])
    wc_src_cat = candidate["pred_src_category"]
    wc_dst_cat = candidate["pred_dst_category"]

    src_dir = _window_dir(ds_path, row_index, "src")
    dst_dir = _window_dir(ds_path, row_index, "dst")

    pre_rgb = _clearest_rgb(src_dir)
    post_rgb = _clearest_rgb(dst_dir)
    pre_lc = _lc_class_map(src_dir)
    post_lc = _lc_class_map(dst_dir)
    if pre_rgb is None or post_rgb is None or pre_lc is None or post_lc is None:
        print(f"[row {row_index}] missing imagery/output; skipping")
        return False

    pre_px = _point_pixel(src_dir, lon, lat)
    post_px = _point_pixel(dst_dir, lon, lat)

    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.95), dpi=300)

    def show_rgb(ax, rgb, px, title) -> None:
        ax.imshow(rgb, interpolation="lanczos")
        _decorate(ax, px, title)

    def show_lc(ax, lc, px, title) -> None:
        ax.imshow(LC_PALETTE[lc], interpolation="nearest")
        _decorate(ax, px, title)

    def _decorate(ax, px, title) -> None:
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        if px is not None:
            ax.add_patch(
                plt.Circle(px, 3.2, fill=False, ec="white", lw=1.8)
            )
            ax.add_patch(
                plt.Circle(px, 3.2, fill=False, ec="#e11919", lw=1.0)
            )

    show_rgb(axes[0], pre_rgb, pre_px, title=f"Sentinel-2 ({src_year})")
    show_rgb(axes[1], post_rgb, post_px, title=f"Sentinel-2 ({dst_year})")
    show_lc(axes[2], pre_lc, pre_px, title=f"WorldCover ({src_year})")
    show_lc(axes[3], post_lc, post_px, title=f"WorldCover ({dst_year})")

    # Legend: only the land-cover classes that appear in either mask.
    present = sorted(set(np.unique(pre_lc)).union(np.unique(post_lc)))
    handles = [
        Patch(facecolor=LC_PALETTE[c], edgecolor="#888888", linewidth=0.4,
              label=_cap(LC_CLASS_NAMES[c]))
        for c in present
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=6,
        frameon=False,
        handlelength=1.0,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.subplots_adjust(
        left=0.01, right=0.99, top=0.86, bottom=0.16, wspace=0.05
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    metadata = {
        "row_index": row_index,
        "lon": lon,
        "lat": lat,
        "src_year": src_year,
        "dst_year": dst_year,
        "worldcover_change_score": wc_score,
        "lcmonitor_change_score": lcc_score,
        "worldcover_pred_src_category": wc_src_cat,
        "worldcover_pred_dst_category": wc_dst_cat,
    }
    metadata_path = os.path.splitext(output_path)[0] + ".json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[row {row_index}] saved {output_path} and {metadata_path}")
    return True


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-dir",
        default=DEFAULT_EVAL_DIR,
        help="Directory with worldcover.csv and lcc_model.csv.",
    )
    parser.add_argument(
        "--ds-path",
        default=DEFAULT_DS_PATH,
        help="Materialized WorldCover eval dataset (with src/dst windows).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write the per-candidate PDFs.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top candidates to render. Default: 10.",
    )
    parser.add_argument(
        "--row-indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit row_index list to render (overrides automatic selection).",
    )
    args = parser.parse_args()

    if args.row_indices:
        wc = _load_csv_by_row_index(os.path.join(args.eval_dir, "worldcover.csv"))
        lcc = _load_csv_by_row_index(os.path.join(args.eval_dir, "lcc_model.csv"))
        candidates = []
        for ri in args.row_indices:
            row = dict(wc[str(ri)])
            row["lcc_change_score"] = lcc[str(ri)]["change_score"]
            candidates.append(row)
    else:
        candidates = select_candidates(args.eval_dir, args.top)

    print(f"Rendering {len(candidates)} candidate figure(s) to {args.output_dir}")
    rendered = 0
    for rank, candidate in enumerate(candidates):
        output_path = os.path.join(
            args.output_dir, f"fp_{rank:02d}_row{int(candidate['row_index'])}.pdf"
        )
        if render_candidate(args.ds_path, candidate, output_path):
            rendered += 1
    print(f"Done: {rendered}/{len(candidates)} figures rendered.")


if __name__ == "__main__":
    main()
