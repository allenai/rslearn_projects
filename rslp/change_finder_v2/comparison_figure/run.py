"""Render LCMonitor-vs-embeddings disagreement example figures (change_finder_v2).

Selects evaluation points where the OlmoEarth embeddings are confident there is *no*
change (cosine change score below the embeddings' ~90%-recall operating threshold) but
the LCMonitor (LCC model) is confident there *is* change (binary change score above the
LCMonitor's ~90%-precision operating threshold), and renders one before/after figure row
per point.

Two categories are produced:

- ``cat1_lcmonitor_wrong``: ground truth is *no change*, so LCMonitor is wrong (false
  positive) and the embeddings are right.
- ``cat2_lcmonitor_right``: ground truth *is change*, so LCMonitor is right (true
  positive) and the embeddings missed it (false negative).

Imagery is read directly from the already-materialized OlmoEarth embeddings evaluation
dataset (``.../evaluation/olmoearth_embeddings``), which holds up to 12 monthly
Sentinel-2 mosaics for each point's ``src_year`` and ``dst_year`` window. No imagery API
access is needed. For each year we pick the clearest (least cloudy / least nodata)
monthly mosaic.

Each row is a compact two-panel PDF: the "before" (src_year) and "after" (dst_year)
RGB images side by side, with the evaluation point circled and a small location callout
overlaid on the imagery, and the metadata (ground truth, LCMonitor verdict, embeddings
verdict) printed underneath so two rows can be placed side by side in the paper.

Example::

    python -m rslp.change_finder_v2.comparison_figure.run \
        --eval-dir /weka/dfive-default/rslearn-eai/datasets/change_finder/evaluation \
        --output-dir /weka/.../lcc_vs_embedding_figure_rows \
        --per-category 10
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from datetime import datetime
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import (  # noqa: E402
    AnchoredOffsetbox,
    TextArea,
    VPacker,
)
import rasterio  # noqa: E402
from rasterio.warp import transform as warp_transform  # noqa: E402

# 12-band Sentinel-2 order written by the dataset config.
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
         "B8A", "B09", "B11", "B12"]
BAND_DIR = "_".join(BANDS)
# RGB band indices into the 12-band array.
RGB_IDX = [BANDS.index("B04"), BANDS.index("B03"), BANDS.index("B02")]
PREDICTION_GROUP = "predict"


def as_bool(v: Any) -> bool:
    """Parse a CSV cell into a bool."""
    return str(v).strip().lower() in ("true", "1")


def load_all(path: str) -> list[dict[str, str]]:
    """Load all standardized-CSV rows, in file order (position == window index)."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _point_key(row: dict[str, str]) -> tuple[float, float]:
    """Stable (lon, lat) key for matching the same point across the two CSVs."""
    return (round(float(row["lon"]), 5), round(float(row["lat"]), 5))


def recall_threshold(rows: list[dict[str, str]], target_recall: float) -> float:
    """Largest change-score threshold whose recall is still >= target_recall."""
    pos = [float(r["change_score"]) for r in rows if as_bool(r["has_changed"])]
    n = len(pos)
    best = 0.0
    for t in sorted({float(r["change_score"]) for r in rows}):
        recall = sum(1 for s in pos if s >= t) / n
        if recall >= target_recall:
            best = t
    return best


def precision_threshold(rows: list[dict[str, str]], target_precision: float) -> float:
    """Smallest change-score threshold whose precision is >= target_precision."""
    for t in sorted({float(r["change_score"]) for r in rows}):
        flagged = [r for r in rows if float(r["change_score"]) >= t]
        if not flagged:
            continue
        tp = sum(1 for r in flagged if as_bool(r["has_changed"]))
        if tp / len(flagged) >= target_precision:
            return t
    return 1.0


def select_categories(
    lcc_csv: str, emb_csv: str, target_recall: float, target_precision: float
) -> tuple[float, float, list[dict], list[dict]]:
    """Find the disagreement points, split into the two ground-truth categories.

    The two CSVs index *different* point orderings (and slightly different point
    counts), so points are matched across models by their (lon, lat) location rather
    than by ``row_index``. The embeddings CSV's file position is the index used to name
    the imagery windows (``eval_{idx:06d}_{src,dst}``).
    """
    lcc_all = load_all(lcc_csv)
    emb_all = load_all(emb_csv)
    lcc_pred = [r for r in lcc_all if as_bool(r.get("has_prediction"))]
    emb_pred = [r for r in emb_all if as_bool(r.get("has_prediction"))]
    emb_thr = recall_threshold(emb_pred, target_recall)
    lcc_thr = precision_threshold(lcc_pred, target_precision)

    # Map each point's (lon, lat) to its embeddings CSV index + row.
    emb_by_key: dict[tuple[float, float], tuple[int, dict[str, str]]] = {}
    for emb_idx, row in enumerate(emb_all):
        emb_by_key[_point_key(row)] = (emb_idx, row)

    cat1: list[dict] = []  # ground truth no-change -> LCMonitor wrong
    cat2: list[dict] = []  # ground truth change   -> LCMonitor right
    for lcc in lcc_all:
        if not as_bool(lcc.get("has_prediction")):
            continue
        match = emb_by_key.get(_point_key(lcc))
        if match is None:
            continue
        emb_idx, emb = match
        if not as_bool(emb.get("has_prediction")):
            continue
        lcc_score = float(lcc["change_score"])
        emb_score = float(emb["change_score"])
        if not (lcc_score >= lcc_thr and emb_score < emb_thr):
            continue
        rec = {
            "emb_index": emb_idx,
            "lon": float(lcc["lon"]),
            "lat": float(lcc["lat"]),
            "src_year": int(emb["src_year"]),
            "dst_year": int(emb["dst_year"]),
            "has_changed": as_bool(lcc["has_changed"]),
            "src_category": lcc["src_category"],
            "dst_category": lcc["dst_category"],
            "lcc_score": lcc_score,
            "emb_score": emb_score,
            "pred_src_category": lcc["pred_src_category"],
            "pred_dst_category": lcc["pred_dst_category"],
        }
        (cat2 if rec["has_changed"] else cat1).append(rec)

    # Most confident disagreements first (highest LCMonitor score, lowest emb score).
    cat1.sort(key=lambda r: (-r["lcc_score"], r["emb_score"]))
    cat2.sort(key=lambda r: (-r["lcc_score"], r["emb_score"]))
    return emb_thr, lcc_thr, cat1, cat2


def _mosaic_mid_date(window_dir: str, layer_name: str) -> datetime | None:
    """Return the mid-date of a materialized mosaic from the window's items.json."""
    items_path = os.path.join(window_dir, "items.json")
    if not os.path.exists(items_path):
        return None
    try:
        items = json.load(open(items_path))
    except Exception:  # noqa: BLE001
        return None
    # layer "sentinel2" -> group 0, "sentinel2.N" -> group N.
    group_idx = 0 if "." not in layer_name else int(layer_name.split(".")[1])
    groups = None
    if isinstance(items, list):
        for entry in items:
            if isinstance(entry, dict) and entry.get("layer_name") == "sentinel2":
                groups = entry.get("serialized_item_groups")
                break
    elif isinstance(items, dict):
        groups = items.get("sentinel2")
    if not isinstance(groups, list) or group_idx >= len(groups):
        return None
    group = groups[group_idx]
    if not group:
        return None
    tr = group[0].get("geometry", {}).get("time_range")
    if not tr:
        return None
    try:
        t0 = datetime.fromisoformat(tr[0])
        t1 = datetime.fromisoformat(tr[1])
    except Exception:  # noqa: BLE001
        return None
    return t0 + (t1 - t0) / 2


def _quality_score(rgb_u16: np.ndarray) -> float:
    """Lower is a cleaner mosaic.

    Penalizes nodata, bright clouds, and atmospheric haze (a high blue-band floor
    with washed-out, low-contrast scenes), so we avoid picking hazy mosaics that
    happen to be cloud-free.
    """
    nodata = np.all(rgb_u16 <= 1, axis=0)
    nodata_frac = float(np.mean(nodata))
    visible = np.mean(rgb_u16, axis=0)
    cloud_frac = float(np.mean(visible >= 3000))
    valid = ~nodata
    if not valid.any():
        return 1e9
    # Haze proxy: the dark-pixel floor of the blue band (reflectance units). Clear
    # scenes have a low blue floor (~0.02-0.06); hazy scenes lift it well above that.
    blue = rgb_u16[2][valid].astype(np.float32) / 10000.0
    haze = float(np.percentile(blue, 5))
    return 2.0 * cloud_frac + nodata_frac + 3.0 * haze


def pick_clearest_mosaic(
    window_dir: str,
) -> tuple[np.ndarray, datetime | None, str] | None:
    """Return (RGB uint16 [3,H,W], mid-date, tif-path) for the cleanest mosaic."""
    pattern = os.path.join(window_dir, "layers", "sentinel2*", BAND_DIR, "geotiff.tif")
    best: tuple[float, np.ndarray, datetime | None, str] | None = None
    for tif in sorted(glob.glob(pattern)):
        layer_name = os.path.basename(os.path.dirname(os.path.dirname(tif)))
        try:
            with rasterio.open(tif) as src:
                arr = src.read()
        except Exception:  # noqa: BLE001
            continue
        rgb = arr[RGB_IDX]
        score = _quality_score(rgb)
        if best is None or score < best[0]:
            best = (score, rgb, _mosaic_mid_date(window_dir, layer_name), tif)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _point_pixel(tif_path: str, lon: float, lat: float) -> tuple[float, float] | None:
    """Convert a lon/lat to (col, row) pixel coordinates within a GeoTIFF."""
    try:
        with rasterio.open(tif_path) as ds:
            xs, ys = warp_transform("EPSG:4326", ds.crs, [lon], [lat])
            col, row = ~ds.transform * (xs[0], ys[0])
    except Exception:  # noqa: BLE001
        return None
    return (col, row)


def to_display(rgb_before: np.ndarray, rgb_after: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stretch a before/after RGB pair with one shared, color-preserving contrast.

    A single low/high cut is computed jointly across both images and all three
    channels (not per-channel), so colors stay natural (no decorrelation speckle)
    and the before/after pair is directly comparable.
    """
    valid = np.concatenate([
        rgb_before[rgb_before > 1].ravel(),
        rgb_after[rgb_after > 1].ravel(),
    ]).astype(np.float32)
    if valid.size:
        lo, hi = np.percentile(valid, [2, 98])
    else:
        lo, hi = 0.0, 3000.0
    if hi <= lo:
        hi = lo + 1.0

    def stretch(rgb: np.ndarray) -> np.ndarray:
        chan = (rgb.astype(np.float32) - lo) / (hi - lo)
        return np.transpose(np.clip(chan, 0, 1), (1, 2, 0))

    return stretch(rgb_before), stretch(rgb_after)


def cap_cat(s: str) -> str:
    """Capitalize a land-cover category label for display."""
    return "/".join(
        " ".join(w[:1].upper() + w[1:] for w in part.split(" "))
        for part in s.split("/")
    )


def _ground_truth_label(rec: dict) -> str:
    """One-line ground-truth label for the metadata caption."""
    if not rec["has_changed"]:
        return "No change"
    src, dst = cap_cat(rec["src_category"]), cap_cat(rec["dst_category"])
    if src and dst:
        return f"Change ({src} \u2192 {dst})"
    return "Change"


def render_row(
    rec: dict,
    before: np.ndarray,
    after: np.ndarray,
    before_date: datetime | None,
    after_date: datetime | None,
    before_px: tuple[float, float] | None,
    after_px: tuple[float, float] | None,
    output_path: str,
) -> None:
    """Render and save one before/after comparison figure as a PDF.

    Two image panels sit side by side with the evaluation point circled and a
    location callout overlaid on the "before" panel; the metadata caption is
    printed underneath so two figures can be placed side by side in the paper.
    """
    h, w = before.shape[:2]
    radius = max(3.0, 0.045 * min(h, w))

    fig, axes = plt.subplots(
        1, 2, figsize=(5.0, 3.35), dpi=300,
        gridspec_kw={"wspace": 0.015},
    )

    panels = [
        (axes[0], before, before_date, rec["src_year"], "Before", before_px),
        (axes[1], after, after_date, rec["dst_year"], "After", after_px),
    ]
    for ax, img, date, year, prefix, px in panels:
        ax.imshow(img, interpolation="lanczos")
        when = date.strftime("%b %Y") if date else str(year)
        ax.set_title(f"{prefix} ({when})", fontsize=9, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        if px is not None:
            ax.add_patch(plt.Circle(px, radius, fill=False, ec="white", lw=2.2))
            ax.add_patch(plt.Circle(px, radius, fill=False, ec="#e11919", lw=1.1))

    # Location callout overlaid on the "before" panel, arrow pointing to the point.
    if before_px is not None:
        # Stop the arrowhead just outside the red circle (compute the target along
        # the text->point direction in image-pixel coordinates so it is unit-safe).
        tx_frac, ty_frac = 0.04, 0.95
        text_x = -0.5 + tx_frac * w
        text_y = (h - 0.5) - ty_frac * h
        dx, dy_ = before_px[0] - text_x, before_px[1] - text_y
        dist = max((dx * dx + dy_ * dy_) ** 0.5, 1e-6)
        gap = radius + 3.0
        target = (before_px[0] - dx / dist * gap, before_px[1] - dy_ / dist * gap)
        annotation = axes[0].annotate(
            f"{rec['lat']:.4f}, {rec['lon']:.4f}",
            xy=target, xycoords="data",
            xytext=(tx_frac, ty_frac), textcoords="axes fraction",
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black",
                      lw=0.8, alpha=0.9),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6,
                            shrinkA=2.0, shrinkB=0.0),
        )
        annotation.arrow_patch.set_path_effects(
            [pe.withStroke(linewidth=2.6, foreground="black")]
        )

    # Metadata caption underneath: lines left-aligned to each other, block centered.
    meta_lines = [
        f"Ground truth: {_ground_truth_label(rec)}",
        f"LCMonitor: Change ({rec['lcc_score']:.2f})",
        f"Embeddings: No change ({rec['emb_score']:.2f})",
    ]
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.205, wspace=0.015)
    caption = VPacker(
        children=[TextArea(line, textprops=dict(fontsize=9)) for line in meta_lines],
        align="left", pad=0, sep=2.0,
    )
    fig.add_artist(AnchoredOffsetbox(
        loc="upper center", child=caption, pad=0, frameon=False,
        bbox_to_anchor=(0.5, 0.19), bbox_transform=fig.transFigure,
    ))

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_category(
    records: list[dict],
    embeddings_ds: str,
    out_dir: str,
    limit: int,
) -> list[dict]:
    """Render up to ``limit`` rows for one category; return the manifest entries."""
    os.makedirs(out_dir, exist_ok=True)
    manifest: list[dict] = []
    rendered = 0
    for rec in records:
        if rendered >= limit:
            break
        idx = rec["emb_index"]
        src_dir = os.path.join(
            embeddings_ds, "windows", PREDICTION_GROUP, f"eval_{idx:06d}_src"
        )
        dst_dir = os.path.join(
            embeddings_ds, "windows", PREDICTION_GROUP, f"eval_{idx:06d}_dst"
        )
        before = pick_clearest_mosaic(src_dir)
        after = pick_clearest_mosaic(dst_dir)
        if before is None or after is None:
            print(f"[idx {idx}] missing imagery; skipping", flush=True)
            continue
        before_disp, after_disp = to_display(before[0], after[0])
        before_px = _point_pixel(before[2], rec["lon"], rec["lat"])
        after_px = _point_pixel(after[2], rec["lon"], rec["lat"])
        out_path = os.path.join(out_dir, f"row_{idx:06d}.pdf")
        render_row(
            rec, before_disp, after_disp, before[1], after[1],
            before_px, after_px, out_path,
        )
        manifest.append({
            "file": os.path.basename(out_path),
            "emb_index": idx,
            "lon": rec["lon"],
            "lat": rec["lat"],
            "src_year": rec["src_year"],
            "dst_year": rec["dst_year"],
            "has_changed": rec["has_changed"],
            "src_category": rec["src_category"],
            "dst_category": rec["dst_category"],
            "lcc_score": rec["lcc_score"],
            "emb_score": rec["emb_score"],
        })
        rendered += 1
        print(f"[idx {idx}] wrote {out_path}", flush=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    default_eval = (
        "/weka/dfive-default/rslearn-eai/datasets/change_finder/evaluation"
    )
    parser.add_argument("--eval-dir", default=default_eval,
                        help="Evaluation directory holding the standardized CSVs.")
    parser.add_argument("--lcc-csv", default=None,
                        help="LCMonitor CSV (default: <eval-dir>/lcc_model.csv).")
    parser.add_argument("--emb-csv", default=None,
                        help="Embeddings CSV (default: "
                             "<eval-dir>/olmoearth_embeddings_cosine.csv).")
    parser.add_argument("--embeddings-ds", default=None,
                        help="OlmoEarth embeddings dataset with materialized imagery "
                             "(default: <eval-dir>/olmoearth_embeddings).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write the two category subfolders into.")
    parser.add_argument("--per-category", type=int, default=10,
                        help="Max rows to render per category.")
    parser.add_argument("--target-recall", type=float, default=0.90,
                        help="Embeddings operating recall (no-change threshold).")
    parser.add_argument("--target-precision", type=float, default=0.90,
                        help="LCMonitor operating precision (change threshold).")
    args = parser.parse_args()

    lcc_csv = args.lcc_csv or os.path.join(args.eval_dir, "lcc_model.csv")
    emb_csv = args.emb_csv or os.path.join(
        args.eval_dir, "olmoearth_embeddings_cosine.csv"
    )
    embeddings_ds = args.embeddings_ds or os.path.join(
        args.eval_dir, "olmoearth_embeddings"
    )

    emb_thr, lcc_thr, cat1, cat2 = select_categories(
        lcc_csv, emb_csv, args.target_recall, args.target_precision
    )
    print(f"Embeddings no-change threshold (recall {args.target_recall}): "
          f"{emb_thr:.4f}", flush=True)
    print(f"LCMonitor change threshold (precision {args.target_precision}): "
          f"{lcc_thr:.4f}", flush=True)
    print(f"Category 1 (LCMonitor wrong, no real change): {len(cat1)} candidates",
          flush=True)
    print(f"Category 2 (LCMonitor right, real change): {len(cat2)} candidates",
          flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    m1 = render_category(
        cat1, embeddings_ds,
        os.path.join(args.output_dir, "cat1_lcmonitor_wrong"), args.per_category,
    )
    m2 = render_category(
        cat2, embeddings_ds,
        os.path.join(args.output_dir, "cat2_lcmonitor_right"), args.per_category,
    )
    print(f"\nRendered {len(m1)} rows to cat1_lcmonitor_wrong and "
          f"{len(m2)} rows to cat2_lcmonitor_right under {args.output_dir}",
          flush=True)


if __name__ == "__main__":
    main()
