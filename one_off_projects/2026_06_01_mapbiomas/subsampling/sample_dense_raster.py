"""Sample dense raster windows around expert-point locations.

For each point in the subsample CSV produced by sample_expert_points.py,
reads a search-size patch from the matching year's MapBiomas raster, samples
n-candidates random window-size sub-windows, scores each by weighted
minority-class pixel count, and keeps the best one.  Outputs an updated
subsample CSV whose LON/LAT columns reflect the selected window centre,
plus per-class/per-split statistics.

Setting --search-size equal to --window-size with --n-candidates 1 reproduces
a simple centred-window extraction (no optimisation).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

MY_ROOT = Path(os.environ.get("MY_ROOT", "."))

DEFAULT_CSV = (
    MY_ROOT / "rslearn_projects/rslp/mapbiomas/subsampling/sample_expert_points_4k.csv"
)
DEFAULT_RASTER_DIR = MY_ROOT / "datasets/mapbiomas/data"
DEFAULT_OUT_DIR = MY_ROOT / "rslearn_projects/rslp/mapbiomas/subsampling"
DEFAULT_LEGEND = (
    MY_ROOT / "datasets/mapbiomas/metadata/Codigos-da-legenda-colecao-10.csv"
)
DEFAULT_HIERARCHY = MY_ROOT / "datasets/mapbiomas/metadata/hierarchy.csv"

MINORITY_CLASSES: set[int] = {
    6,
    41,
    5,
    29,
    46,
    48,
    40,
    47,
    25,
    50,
    49,
    35,
    32,
    23,
    30,
    62,
}


# ---------------------------------------------------------------------------
# Legend / summary utilities
# ---------------------------------------------------------------------------


def load_legend(path: Path) -> dict[int, str]:
    """Return {class_id: english_description} from the MapBiomas legend CSV."""
    legend = pd.read_csv(path, sep="\t")
    legend["Description"] = legend["Description"].str.strip()
    return dict(zip(legend["Class_ID"].astype(int), legend["Description"]))


def load_hierarchy(path: Path) -> dict[int, dict]:
    """Parse the hierarchy CSV and return per-class hierarchy info.

    Returns ``{class_id: {"leaf_level": int,
                          "parent_class_id": int | None,
                          "parent_leaf_level": int | None,
                          "parent_class_desc": str | None}}``.
    """
    h = pd.read_csv(path)
    result: dict[int, dict] = {}
    for cid, grp in h.groupby("Class_ID"):
        cid = int(cid)
        leaf_level = int(grp["Leaf_Level"].iloc[0])

        if leaf_level > 1:
            parent_row = grp[grp["Hierarchy_Level"] == leaf_level - 1].iloc[0]
            parent_class_id = int(parent_row["Level_Class_ID"])
            parent_leaf_level = leaf_level - 1
            parent_class_desc = str(parent_row["Level_Description"])
        else:
            parent_class_id = None
            parent_leaf_level = None
            parent_class_desc = None

        result[cid] = {
            "leaf_level": leaf_level,
            "parent_class_id": parent_class_id,
            "parent_leaf_level": parent_leaf_level,
            "parent_class_desc": parent_class_desc,
        }
    return result


def build_summary(
    per_window: pd.DataFrame,
    legend: dict[int, str],
    window_size: int,
    hierarchy: dict[int, dict] | None = None,
) -> pd.DataFrame:
    """Aggregate per-window counts into a global summary with train/val breakdown."""
    class_cols = [c for c in per_window.columns if c.startswith("class_")]
    class_ids = [int(c.split("_")[1]) for c in class_cols]

    pixels_per_window = window_size * window_size

    rows: list[dict] = []
    for cid, col in zip(class_ids, class_cols):
        vals_all = per_window[col].values
        total_all = int(vals_all.sum())

        rec: dict = {
            "class_id": cid,
            "class_name": legend.get(cid, "unknown"),
            "leaf_level": None,
            "parent_class_id": None,
            "parent_leaf_level": None,
            "parent_class_desc": None,
            "total_pixels": total_all,
            "frac_of_all": total_all / (len(per_window) * pixels_per_window),
            "mean_per_window": float(vals_all.mean()),
            "std_per_window": float(vals_all.std()),
        }

        if hierarchy and cid in hierarchy:
            hi = hierarchy[cid]
            rec["leaf_level"] = hi["leaf_level"]
            rec["parent_class_id"] = hi["parent_class_id"]
            rec["parent_leaf_level"] = hi["parent_leaf_level"]
            rec["parent_class_desc"] = hi["parent_class_desc"]

        for split in ("train", "val"):
            mask = per_window["split"] == split
            vals = per_window.loc[mask, col].values
            n_windows = int(mask.sum())
            total = int(vals.sum())
            rec[f"{split}_total_pixels"] = total
            rec[f"{split}_frac"] = (
                total / (n_windows * pixels_per_window) if n_windows else 0.0
            )
            rec[f"{split}_mean_per_window"] = float(vals.mean()) if n_windows else 0.0
            rec[f"{split}_std_per_window"] = float(vals.std()) if n_windows else 0.0

        rows.append(rec)

    summary = (
        pd.DataFrame(rows)
        .sort_values("total_pixels", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def print_summary(
    summary: pd.DataFrame, per_window: pd.DataFrame, window_size: int
) -> None:
    """Print a human-readable summary to stdout."""
    pixels_per_window = window_size * window_size
    n_total = len(per_window)
    n_train = (per_window["split"] == "train").sum()
    n_val = (per_window["split"] == "val").sum()

    print("\n" + "=" * 140)
    print("WINDOW CLASS STATISTICS SUMMARY")
    print(
        f"  Window size: {window_size}x{window_size} ({pixels_per_window} pixels/window)"
    )
    print(f"  Samples: {n_total} total ({n_train} train, {n_val} val)")
    print(f"  Total pixels across all windows: {n_total * pixels_per_window:,}")
    print("=" * 140)

    has_hierarchy = (
        "leaf_level" in summary.columns and summary["leaf_level"].notna().any()
    )

    if has_hierarchy:
        fmt = (
            "{:<6s}  {:<35s}  {:<5s}  {:<6s}  {:<25s}"
            "  {:>12s}  {:>8s}  {:>12s}  {:>8s}  {:>12s}  {:>8s}"
        )
        print(
            fmt.format(
                "ID",
                "Class",
                "Lvl",
                "ParID",
                "Parent",
                "All pixels",
                "All %",
                "Train px",
                "Train %",
                "Val px",
                "Val %",
            )
        )
    else:
        fmt = "{:<6s}  {:<35s}  {:>12s}  {:>8s}  {:>12s}  {:>8s}  {:>12s}  {:>8s}"
        print(
            fmt.format(
                "ID",
                "Class",
                "All pixels",
                "All %",
                "Train px",
                "Train %",
                "Val px",
                "Val %",
            )
        )
    print("-" * 140)

    for _, r in summary.iterrows():
        if has_hierarchy:
            lvl = str(int(r["leaf_level"])) if pd.notna(r["leaf_level"]) else ""
            par_id = (
                str(int(r["parent_class_id"]))
                if pd.notna(r["parent_class_id"])
                else "-"
            )
            par_desc = (
                str(r["parent_class_desc"])[:25]
                if pd.notna(r["parent_class_desc"])
                else "-"
            )
            print(
                fmt.format(
                    str(int(r["class_id"])),
                    r["class_name"][:35],
                    lvl,
                    par_id,
                    par_desc,
                    f"{int(r['total_pixels']):,}",
                    f"{r['frac_of_all']:.3%}",
                    f"{int(r['train_total_pixels']):,}",
                    f"{r['train_frac']:.3%}",
                    f"{int(r['val_total_pixels']):,}",
                    f"{r['val_frac']:.3%}",
                )
            )
        else:
            print(
                (
                    "{:<6s}  {:<35s}  {:>12s}  {:>8s}  {:>12s}  {:>8s}  {:>12s}  {:>8s}"
                ).format(
                    str(int(r["class_id"])),
                    r["class_name"][:35],
                    f"{int(r['total_pixels']):,}",
                    f"{r['frac_of_all']:.3%}",
                    f"{int(r['train_total_pixels']):,}",
                    f"{r['train_frac']:.3%}",
                    f"{int(r['val_total_pixels']):,}",
                    f"{r['val_frac']:.3%}",
                )
            )
    print("=" * 140 + "\n")


# Weights = 1 / observed_fraction from the baseline window_class_stats run.
MINORITY_WEIGHTS: dict[int, float] = {
    6: 1.0 / 0.01899,
    41: 1.0 / 0.01792,
    5: 1.0 / 0.00772,
    29: 1.0 / 0.00634,
    46: 1.0 / 0.00583,
    48: 1.0 / 0.00474,
    40: 1.0 / 0.00246,
    47: 1.0 / 0.00235,
    25: 1.0 / 0.00183,
    50: 1.0 / 0.00081,
    49: 1.0 / 0.00066,
    35: 1.0 / 0.00059,
    32: 1.0 / 0.00045,
    23: 1.0 / 0.00041,
    30: 1.0 / 0.00034,
    62: 1.0 / 0.00033,
}

# Pre-build a uint8-indexed lookup table for fast vectorised scoring.
_WEIGHT_LUT = np.zeros(256, dtype=np.float64)
for _cls, _w in MINORITY_WEIGHTS.items():
    _WEIGHT_LUT[_cls] = _w


def _score_subwindow(patch: np.ndarray, row_off: int, col_off: int, size: int) -> float:
    """Sum minority weights for all pixels in the sub-window."""
    sub = patch[row_off : row_off + size, col_off : col_off + size]
    return float(_WEIGHT_LUT[sub].sum())


def optimize_windows(
    df: pd.DataFrame,
    raster_dir: Path,
    window_size: int,
    search_size: int,
    n_candidates: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find the best 32x32 sub-window per sample and return updated CSV + per-window counts.

    Returns:
    -------
    optimized_df : pd.DataFrame
        Same schema as the input subsample CSV but with LON/LAT shifted to the
        selected window centre.
    per_window : pd.DataFrame
        Per-window class counts (same format as window_class_stats output).
    """
    half_search = search_size // 2
    half_win = window_size // 2
    max_offset = (
        search_size - window_size
    )  # valid sub-window origin range [0, max_offset]

    updated_rows: list[dict] = []
    count_records: list[dict] = []

    for year, group in df.groupby("YEAR"):
        raster_path = raster_dir / f"brazil_coverage_{year}.tif"
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster not found: {raster_path}")

        print(f"  Year {year}: {len(group)} samples from {raster_path.name}")

        with rasterio.open(raster_path) as src:
            for _, row in group.iterrows():
                actual_lon = row["LAT"]
                actual_lat = row["LON"]
                r, c = src.index(actual_lon, actual_lat)

                big_win = Window(
                    col_off=c - half_search,
                    row_off=r - half_search,
                    width=search_size,
                    height=search_size,
                )
                patch = src.read(1, window=big_win, boundless=True, fill_value=0)

                offsets = np.column_stack(
                    [
                        rng.integers(0, max_offset + 1, size=n_candidates),
                        rng.integers(0, max_offset + 1, size=n_candidates),
                    ]
                )

                best_score = -1.0
                best_idx = 0
                for i in range(n_candidates):
                    ro, co = int(offsets[i, 0]), int(offsets[i, 1])
                    score = _score_subwindow(patch, ro, co, window_size)
                    if score > best_score:
                        best_score = score
                        best_idx = i

                best_ro = int(offsets[best_idx, 0])
                best_co = int(offsets[best_idx, 1])

                # Raster row/col of the selected sub-window centre
                new_r = (r - half_search) + best_ro + half_win
                new_c = (c - half_search) + best_co + half_win
                new_lon, new_lat = src.xy(new_r, new_c)

                # Build updated CSV row (LON/LAT columns are swapped in source)
                updated_rows.append(
                    {
                        "TARGETID": int(row["TARGETID"]),
                        "LON": new_lat,
                        "LAT": new_lon,
                        "YEAR": int(row["YEAR"]),
                        "CLASS": int(row["CLASS"]),
                        "BORDA": int(row["BORDA"]),
                        "COUNT": int(row["COUNT"]),
                        "CARTA_2": row["CARTA_2"],
                        "DECLIVIDAD": row["DECLIVIDAD"],
                        "split": row["split"],
                    }
                )

                # Per-window class counts for the selected sub-window
                sub = patch[
                    best_ro : best_ro + window_size, best_co : best_co + window_size
                ]
                codes, counts = np.unique(sub, return_counts=True)
                rec: dict = {
                    "TARGETID": int(row["TARGETID"]),
                    "YEAR": int(row["YEAR"]),
                    "center_class": int(row["CLASS"]),
                    "split": row["split"],
                }
                for code, cnt in zip(codes.tolist(), counts.tolist()):
                    rec[f"class_{code}"] = cnt
                count_records.append(rec)

    optimized_df = pd.DataFrame(updated_rows)
    out_cols = [
        "TARGETID",
        "LON",
        "LAT",
        "YEAR",
        "CLASS",
        "BORDA",
        "COUNT",
        "CARTA_2",
        "DECLIVIDAD",
        "split",
    ]
    optimized_df = (
        optimized_df[out_cols]
        .sort_values(
            ["split", "CLASS", "YEAR"],
        )
        .reset_index(drop=True)
    )

    per_window = pd.DataFrame(count_records).fillna(0)
    class_cols = sorted(
        [c for c in per_window.columns if c.startswith("class_")],
        key=lambda c: int(c.split("_")[1]),
    )
    meta_cols = ["TARGETID", "YEAR", "center_class", "split"]
    per_window = per_window[meta_cols + class_cols]
    for col in class_cols:
        per_window[col] = per_window[col].astype(int)

    return optimized_df, per_window


def main() -> None:
    """Select minority-boosting 32x32 windows for each subsample point."""
    parser = argparse.ArgumentParser(
        description="Select minority-boosting 32x32 windows for each subsample point.",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(DEFAULT_CSV),
        help="Path to the subsample CSV (output of sample_expert_points.py).",
    )
    parser.add_argument(
        "--raster-dir",
        type=str,
        default=str(DEFAULT_RASTER_DIR),
        help="Directory containing brazil_coverage_{year}.tif rasters.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=16,
        help="Side length of the target window in pixels (default: 16).",
    )
    parser.add_argument(
        "--search-size",
        type=int,
        default=512,
        help="Side length of the search neighbourhood in pixels (default: 512).",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=64,
        help="Number of random sub-window candidates to evaluate (default: 64).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--legend-path",
        type=str,
        default=str(DEFAULT_LEGEND),
        help="Path to the MapBiomas legend CSV (tab-separated).",
    )
    parser.add_argument(
        "--hierarchy-path",
        type=str,
        default=str(DEFAULT_HIERARCHY),
        help="Path to the MapBiomas hierarchy CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    raster_dir = Path(args.raster_dir)
    out_dir = Path(args.out_dir)
    legend_path = Path(args.legend_path)
    hierarchy_path = Path(args.hierarchy_path)
    window_size = args.window_size
    search_size = args.search_size

    legend = load_legend(legend_path)
    hierarchy = load_hierarchy(hierarchy_path) if hierarchy_path.exists() else None
    minority_names = {
        cid: legend.get(cid, str(cid)) for cid in sorted(MINORITY_CLASSES)
    }

    print("=" * 70)
    print("MINORITY-BOOSTING WINDOW SELECTION")
    print("=" * 70)
    print(
        f"  Search window:   {search_size}x{search_size} pixels centred on each point"
    )
    print(f"  Target window:   {window_size}x{window_size} pixels")
    print(f"  Candidates:      {args.n_candidates} random sub-windows per point")
    print(f"  Seed:            {args.seed}")
    print(f"  Minority classes ({len(MINORITY_CLASSES)}):")
    for cid in sorted(MINORITY_CLASSES):
        print(
            f"    {cid:>3d}  {minority_names[cid]:<35s}  weight={MINORITY_WEIGHTS[cid]:,.1f}"
        )
    print("\n  Scoring: for each candidate sub-window, sum the weight of every")
    print("  minority-class pixel.  Keep the sub-window with the highest score.")
    print("=" * 70)

    print(f"\nLoading subsample CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(
        f"  {len(df)} samples ({(df['split'] == 'train').sum()} train, "
        f"{(df['split'] == 'val').sum()} val)"
    )

    rng = np.random.default_rng(args.seed)

    print(
        f"\nSearching {search_size}x{search_size} neighbourhood, "
        f"{args.n_candidates} candidates per point …"
    )
    optimized_df, per_window = optimize_windows(
        df,
        raster_dir,
        window_size,
        search_size,
        args.n_candidates,
        rng,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / "sample_dense_raster_4k.csv"
    optimized_df.to_csv(csv_out, index=False)
    print(f"\nWrote optimized subsample: {csv_out}  ({len(optimized_df)} rows)")

    summary = build_summary(per_window, legend, window_size, hierarchy)
    summary_path = out_dir / "sample_dense_raster_window_stats_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary stats:      {summary_path}  ({len(summary)} rows)")

    print_summary(summary, per_window, window_size)


if __name__ == "__main__":
    main()
