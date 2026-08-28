"""Sanity-check the created rslearn label windows against reference data.

Validates that:
  A) Every window maps to a row in the subsample CSV.
  B) TARGETID, CLASS (and LON/LAT for sparse) match the 85k validation shapefile.
  C) (dense only) The 48x48 rslearn raster perfectly matches the 16x16 MapBiomas
     raster patch upscaled 3x.
  D) Produces a 5x5 visualization grid of randomly sampled windows.

Usage::

    # Sparse expert labels
    python -m rslp.mapbiomas.sanity_check --mode sparse --ds-name mapbiomas_3k

    # Dense raster labels
    python -m rslp.mapbiomas.sanity_check --mode dense --ds-name mapbiomas_3k
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.windows
from pyproj import Transformer

MY_ROOT = Path(os.environ.get("MY_ROOT", "."))

DEFAULT_DS_NAME = "mapbiomas_3k"
DEFAULT_RASTER_DIR = MY_ROOT / "datasets/mapbiomas/data"
DEFAULT_SHP_PATH = (
    MY_ROOT / "datasets/mapbiomas/metadata/mapbiomas_85k_points_validation.shp"
)
DEFAULT_SPARSE_CSV = (
    MY_ROOT / "rslearn_projects/rslp/mapbiomas/subsampling/sample_expert_points_4k.csv"
)
DEFAULT_DENSE_CSV = (
    MY_ROOT / "rslearn_projects/rslp/mapbiomas/subsampling/sample_dense_raster_4k.csv"
)

NODATA_VALUE = 0
UPSCALE_FACTOR = 3
LABEL_BLOCK_SIZE = 3

YEAR_RANGE = range(2016, 2023)

MAPBIOMAS_COLORS = {
    0: "#ffffff",
    1: "#1f8d49",
    3: "#006400",
    4: "#00ff00",
    5: "#687537",
    6: "#76a5af",
    9: "#02d659",
    11: "#519799",
    12: "#d6bc74",
    13: "#d89f5c",
    15: "#edde8e",
    18: "#E974ED",
    19: "#C27BA0",
    20: "#db4d4f",
    21: "#ffefc3",
    23: "#d68fe2",
    24: "#d4271e",
    25: "#9c0027",
    26: "#2532e4",
    29: "#ffaa5f",
    30: "#9065d0",
    31: "#091077",
    32: "#fc8114",
    33: "#2532e4",
    35: "#9065d0",
    36: "#e787f8",
    39: "#f5b3c8",
    40: "#c71585",
    41: "#f54ca9",
    46: "#d68fe2",
    47: "#6b9c32",
    48: "#6b9c32",
    49: "#a1d99b",
    50: "#a1d99b",
    62: "#e6ccff",
}


def _class_cmap(data: np.ndarray) -> np.ndarray:
    """Map class IDs to RGBA image for visualization."""
    from matplotlib.colors import to_rgba

    h, w = data.shape
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = 0.8  # default grey for unknown classes

    for cls_id, hex_color in MAPBIOMAS_COLORS.items():
        mask = data == cls_id
        if not mask.any():
            continue
        r, g, b, a = to_rgba(hex_color)
        rgba[mask] = [r, g, b, a]

    # Make nodata transparent-ish (white with low alpha)
    nodata_mask = data == NODATA_VALUE
    rgba[nodata_mask] = [1.0, 1.0, 1.0, 0.3]

    return rgba


# ---------------------------------------------------------------------------
# Load reference data
# ---------------------------------------------------------------------------


def load_shapefile_reference(shp_path: Path) -> pd.DataFrame:
    """Load the 85k validation shapefile and melt to long format.

    Returns DataFrame with columns: TARGETID, LON, LAT, YEAR, CLASS.
    """
    gdf = gpd.read_file(shp_path)
    static_cols = ["TARGETID", "LON", "LAT"]
    frames: list[pd.DataFrame] = []
    for year in YEAR_RANGE:
        cls_col = f"CLASS_{year}"
        if cls_col not in gdf.columns:
            continue
        sub = gdf[static_cols + [cls_col]].copy()
        sub.columns = [*static_cols, "CLASS"]
        sub["YEAR"] = year
        sub["CLASS"] = pd.to_numeric(sub["CLASS"], errors="coerce")
        frames.append(sub)
    long = pd.concat(frames, ignore_index=True)
    long = long[long["CLASS"].notna()].copy()
    long["CLASS"] = long["CLASS"].astype(int)
    long["TARGETID"] = long["TARGETID"].astype(int)
    return long


# ---------------------------------------------------------------------------
# Check A: every window has a corresponding CSV row
# ---------------------------------------------------------------------------


def check_a_csv_lookup(
    window_names: list[str], csv_df: pd.DataFrame, mode: str
) -> dict[str, pd.Series]:
    """Verify every window name maps to a CSV row. Returns {window_name: row}."""
    csv_df = csv_df.copy()
    csv_df["TARGETID"] = csv_df["TARGETID"].astype(int)
    csv_df["YEAR"] = csv_df["YEAR"].astype(int)
    csv_df["_key"] = csv_df["TARGETID"].astype(str) + "_" + csv_df["YEAR"].astype(str)
    csv_lookup = csv_df.set_index("_key")

    ok, fail = 0, 0
    result: dict[str, pd.Series] = {}
    for wname in window_names:
        if wname in csv_lookup.index:
            result[wname] = csv_lookup.loc[wname]
            ok += 1
        else:
            print(f"  [FAIL] Window {wname} not found in {mode} CSV")
            fail += 1

    print(
        f"  Check A ({mode}): {ok}/{ok + fail} windows found in CSV"
        f"  ({fail} missing)"
    )
    return result


# ---------------------------------------------------------------------------
# Check B: cross-reference with shapefile
# ---------------------------------------------------------------------------


def check_b_shapefile_reference(
    lookup: dict[str, pd.Series],
    ref_df: pd.DataFrame,
    mode: str,
) -> None:
    """Verify TARGETID, CLASS, and (for sparse) LON/LAT match the shapefile."""
    ref_df = ref_df.copy()
    ref_df["_key"] = ref_df["TARGETID"].astype(str) + "_" + ref_df["YEAR"].astype(str)
    ref_lookup = ref_df.set_index("_key")

    ok, fail_missing, fail_class, fail_lonlat = 0, 0, 0, 0
    for wname, csv_row in lookup.items():
        if wname not in ref_lookup.index:
            print(f"  [FAIL] Window {wname}: TARGETID/YEAR not in shapefile")
            fail_missing += 1
            continue

        ref_row = ref_lookup.loc[wname]
        csv_class = int(csv_row["CLASS"])
        ref_class = int(ref_row["CLASS"])
        if csv_class != ref_class:
            print(
                f"  [FAIL] Window {wname}: CLASS mismatch "
                f"(csv={csv_class}, shp={ref_class})"
            )
            fail_class += 1
            continue

        if mode == "sparse":
            # LON/LAT columns are swapped in CSV relative to standard meaning,
            # but the shapefile has the same swap, so compare directly.
            csv_lon, csv_lat = float(csv_row["LON"]), float(csv_row["LAT"])
            ref_lon, ref_lat = float(ref_row["LON"]), float(ref_row["LAT"])
            if not (
                np.isclose(csv_lon, ref_lon, atol=1e-4)
                and np.isclose(csv_lat, ref_lat, atol=1e-4)
            ):
                print(
                    f"  [FAIL] Window {wname}: LON/LAT mismatch "
                    f"(csv=({csv_lon:.6f},{csv_lat:.6f}), "
                    f"shp=({ref_lon:.6f},{ref_lat:.6f}))"
                )
                fail_lonlat += 1
                continue

        ok += 1

    total = ok + fail_missing + fail_class + fail_lonlat
    parts = []
    if fail_missing:
        parts.append(f"{fail_missing} missing")
    if fail_class:
        parts.append(f"{fail_class} class mismatch")
    if fail_lonlat:
        parts.append(f"{fail_lonlat} lon/lat mismatch")
    detail = f"  ({', '.join(parts)})" if parts else ""
    print(f"  Check B ({mode}): {ok}/{total} passed{detail}")


# ---------------------------------------------------------------------------
# Check C (dense only): raster alignment
# ---------------------------------------------------------------------------


def check_c_dense_raster_alignment(
    lookup: dict[str, pd.Series],
    ds_path: Path,
    raster_dir: Path,
    target_size: int,
) -> list[str]:
    """Verify the rslearn 48x48 label matches the MapBiomas 16x16 patch x3.

    Returns list of window names that failed the check.
    """
    raster_patch_size = target_size // UPSCALE_FACTOR
    half_patch = raster_patch_size // 2
    group_name = "mapbiomas_dense_raster"

    ok, fail_shape, fail_align = 0, 0, 0
    failed_windows: list[str] = []
    raster_cache: dict[int, rasterio.DatasetReader] = {}

    try:
        for wname, csv_row in lookup.items():
            year = int(csv_row["YEAR"])
            # LON/LAT swapped in CSV
            longitude = float(csv_row["LAT"])
            latitude = float(csv_row["LON"])

            # Read reference patch from MapBiomas raster
            if year not in raster_cache:
                raster_path = raster_dir / f"brazil_coverage_{year}.tif"
                raster_cache[year] = rasterio.open(raster_path)
            src = raster_cache[year]
            r, c = src.index(longitude, latitude)
            rio_win = rasterio.windows.Window(
                col_off=c - half_patch,
                row_off=r - half_patch,
                width=raster_patch_size,
                height=raster_patch_size,
            )
            ref_patch = src.read(1, window=rio_win, boundless=True, fill_value=0)

            # Read rslearn label raster
            tif_path = (
                ds_path
                / "windows"
                / group_name
                / wname
                / "layers"
                / "label_raster"
                / "label"
                / "geotiff.tif"
            )
            if not tif_path.exists():
                print(f"  [FAIL] Window {wname}: rslearn geotiff not found")
                fail_shape += 1
                failed_windows.append(wname)
                continue

            with rasterio.open(tif_path) as rsrc:
                rslearn_raster = rsrc.read(1)

            if rslearn_raster.shape != (target_size, target_size):
                print(
                    f"  [FAIL] Window {wname}: rslearn raster shape "
                    f"{rslearn_raster.shape} != ({target_size},{target_size})"
                )
                fail_shape += 1
                failed_windows.append(wname)
                continue

            # Downsample rslearn raster back to raster_patch_size and compare
            downsampled = rslearn_raster[::UPSCALE_FACTOR, ::UPSCALE_FACTOR]
            if downsampled.shape != ref_patch.shape:
                print(
                    f"  [FAIL] Window {wname}: downsampled shape "
                    f"{downsampled.shape} != ref {ref_patch.shape}"
                )
                fail_shape += 1
                failed_windows.append(wname)
                continue

            # Only compare pixels where both sides have data. The rslearn
            # raster may have nodata where omitted classes (e.g. 31) were
            # zeroed out during window creation.
            both_valid = (downsampled != NODATA_VALUE) & (ref_patch != NODATA_VALUE)
            mismatch = (downsampled != ref_patch) & both_valid
            if mismatch.any():
                n_diff = int(mismatch.sum())
                print(
                    f"  [FAIL] Window {wname}: {n_diff}/{int(both_valid.sum())} "
                    f"valid pixels differ"
                )
                fail_align += 1
                failed_windows.append(wname)
                continue

            ok += 1
    finally:
        for src in raster_cache.values():
            src.close()

    total = ok + fail_shape + fail_align
    parts = []
    if fail_shape:
        parts.append(f"{fail_shape} shape/missing")
    if fail_align:
        parts.append(f"{fail_align} pixel mismatch")
    detail = f"  ({', '.join(parts)})" if parts else ""
    print(f"  Check C (dense): {ok}/{total} passed{detail}")
    return failed_windows


# ---------------------------------------------------------------------------
# Check D: visualization
# ---------------------------------------------------------------------------


def visualize_sparse(
    lookup: dict[str, pd.Series],
    ds_path: Path,
    out_path: Path,
    n: int = 25,
    seed: int = 42,
) -> None:
    """5x5 grid of sparse windows with labeled pixels highlighted."""
    group_name = "mapbiomas_expert_sparse"
    rng = np.random.default_rng(seed)
    keys = list(lookup.keys())
    chosen = rng.choice(keys, size=min(n, len(keys)), replace=False)

    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
    fig.suptitle("Sparse expert label windows (labeled pixels colored)", fontsize=14)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(chosen):
            ax.axis("off")
            continue

        wname = chosen[idx]
        csv_row = lookup[wname]
        ref_class = int(csv_row["CLASS"])

        tif_path = (
            ds_path
            / "windows"
            / group_name
            / wname
            / "layers"
            / "label_raster"
            / "label"
            / "geotiff.tif"
        )
        if not tif_path.exists():
            ax.set_title(f"{wname}\nMISSING", fontsize=7)
            ax.axis("off")
            continue

        with rasterio.open(tif_path) as src:
            raster = src.read(1)
            raster_crs = src.crs
            raster_transform = src.transform

        rgba = _class_cmap(raster)
        ax.imshow(rgba, interpolation="nearest")

        # Overlay the reference lat/lon as a cyan dot (projected to pixel coords)
        ref_lon = float(csv_row["LAT"])  # LON/LAT swapped in CSV
        ref_lat = float(csv_row["LON"])
        transformer = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
        ref_x, ref_y = transformer.transform(ref_lon, ref_lat)
        inv_transform = ~raster_transform
        ref_col_px, ref_row_px = inv_transform * (ref_x, ref_y)
        ax.plot(
            ref_col_px,
            ref_row_px,
            "o",
            color="cyan",
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.8,
            zorder=5,
        )

        label_vals = np.unique(raster[raster != NODATA_VALUE])
        label_str = ",".join(str(v) for v in label_vals)
        ax.set_title(
            f"{wname}\nlabel={label_str}  ref={ref_class}",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved sparse visualization to {out_path}")
    plt.close(fig)


def visualize_dense(
    lookup: dict[str, pd.Series],
    ds_path: Path,
    raster_dir: Path,
    target_size: int,
    out_path: Path,
    n: int = 25,
    seed: int = 42,
    priority_windows: list[str] | None = None,
) -> None:
    """5x5 grid: each cell shows rslearn 48x48 and reference 16x16 side by side.

    Windows in *priority_windows* (e.g. those that failed Check C) are shown
    first; remaining slots are filled by random sampling.
    """
    group_name = "mapbiomas_dense_raster"
    raster_patch_size = target_size // UPSCALE_FACTOR
    half_patch = raster_patch_size // 2
    rng = np.random.default_rng(seed)
    keys = list(lookup.keys())

    priority = []
    if priority_windows:
        priority = [w for w in priority_windows if w in lookup][:n]
    remaining_n = n - len(priority)
    remaining_pool = [k for k in keys if k not in set(priority)]
    random_pick = (
        list(
            rng.choice(
                remaining_pool,
                size=min(remaining_n, len(remaining_pool)),
                replace=False,
            )
        )
        if remaining_n > 0 and remaining_pool
        else []
    )
    chosen = priority + random_pick

    rows, cols = 5, 5
    fig, axes = plt.subplots(rows, cols * 2, figsize=(24, 18))
    fig.suptitle(
        "Dense raster label windows  (left: rslearn 48x48, right: reference 16x16)",
        fontsize=14,
    )

    raster_cache: dict[int, rasterio.DatasetReader] = {}

    try:
        for idx in range(rows * cols):
            ax_left = axes[idx // cols, (idx % cols) * 2]
            ax_right = axes[idx // cols, (idx % cols) * 2 + 1]

            if idx >= len(chosen):
                ax_left.axis("off")
                ax_right.axis("off")
                continue

            wname = chosen[idx]
            csv_row = lookup[wname]
            year = int(csv_row["YEAR"])
            longitude = float(csv_row["LAT"])
            latitude = float(csv_row["LON"])

            # rslearn raster
            tif_path = (
                ds_path
                / "windows"
                / group_name
                / wname
                / "layers"
                / "label_raster"
                / "label"
                / "geotiff.tif"
            )
            if not tif_path.exists():
                ax_left.set_title(f"{wname}\nMISSING", fontsize=6)
                ax_left.axis("off")
                ax_right.axis("off")
                continue

            with rasterio.open(tif_path) as src:
                rslearn_raster = src.read(1)

            # Reference raster
            if year not in raster_cache:
                rp = raster_dir / f"brazil_coverage_{year}.tif"
                raster_cache[year] = rasterio.open(rp)
            src_ref = raster_cache[year]
            r, c = src_ref.index(longitude, latitude)
            rio_win = rasterio.windows.Window(
                col_off=c - half_patch,
                row_off=r - half_patch,
                width=raster_patch_size,
                height=raster_patch_size,
            )
            ref_patch = src_ref.read(1, window=rio_win, boundless=True, fill_value=0)

            is_failed = priority_windows and wname in set(priority_windows)
            title_prefix = "[FAIL C] " if is_failed else ""
            title_color = "red" if is_failed else "black"

            ax_left.imshow(_class_cmap(rslearn_raster), interpolation="nearest")
            ax_left.set_title(
                f"{title_prefix}{wname}\nrslearn {rslearn_raster.shape}",
                fontsize=6,
                color=title_color,
            )
            ax_left.set_xticks([])
            ax_left.set_yticks([])

            ax_right.imshow(_class_cmap(ref_patch), interpolation="nearest")
            ax_right.set_title(f"ref {ref_patch.shape}", fontsize=6)
            ax_right.set_xticks([])
            ax_right.set_yticks([])
    finally:
        for src in raster_cache.values():
            src.close()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved dense visualization to {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Sanity-check MapBiomas rslearn label windows."""
    parser = argparse.ArgumentParser(
        description="Sanity-check MapBiomas rslearn label windows.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["sparse", "dense"],
        help="Which window set to check.",
    )
    parser.add_argument(
        "--ds-name",
        type=str,
        default=DEFAULT_DS_NAME,
        help="Dataset name under RSLEARN_EAI_ROOT.",
    )
    parser.add_argument(
        "--shp-path",
        type=str,
        default=str(DEFAULT_SHP_PATH),
        help="Path to the 85k validation shapefile.",
    )
    parser.add_argument(
        "--raster-dir",
        type=str,
        default=str(DEFAULT_RASTER_DIR),
        help="Directory containing brazil_coverage_{year}.tif rasters (dense only).",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=48,
        help="Target crop size in 10m pixels (default: 48).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for visualization sampling.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(MY_ROOT / "rslearn_projects/rslp/mapbiomas"),
        help="Directory for output plots.",
    )
    args = parser.parse_args()

    rslearn_root = os.environ.get("RSLEARN_EAI_ROOT")
    if not rslearn_root:
        raise RuntimeError("RSLEARN_EAI_ROOT environment variable is not set")
    ds_path = Path(rslearn_root) / args.ds_name

    mode = args.mode
    raster_dir = Path(args.raster_dir)
    shp_path = Path(args.shp_path)
    out_dir = Path(args.out_dir)
    target_size = args.target_size

    if mode == "sparse":
        group_name = "mapbiomas_expert_sparse"
        csv_path = DEFAULT_SPARSE_CSV
    else:
        group_name = "mapbiomas_dense_raster"
        csv_path = DEFAULT_DENSE_CSV

    windows_dir = ds_path / "windows" / group_name
    if not windows_dir.exists():
        print(f"ERROR: windows directory not found: {windows_dir}")
        sys.exit(1)

    window_names = sorted([p.name for p in windows_dir.iterdir() if p.is_dir()])
    print(f"Found {len(window_names)} windows in {windows_dir}")

    # Load reference data
    print(f"Loading subsample CSV: {csv_path}")
    csv_df = pd.read_csv(csv_path)
    print(f"Loading shapefile: {shp_path}")
    ref_df = load_shapefile_reference(shp_path)

    # --- Check A ---
    print("\n--- Check A: CSV lookup ---")
    lookup = check_a_csv_lookup(window_names, csv_df, mode)

    # --- Check B ---
    print("\n--- Check B: Shapefile cross-reference ---")
    check_b_shapefile_reference(lookup, ref_df, mode)

    # --- Check C (dense only) ---
    failed_c: list[str] = []
    if mode == "dense":
        print("\n--- Check C: Dense raster alignment ---")
        failed_c = check_c_dense_raster_alignment(
            lookup, ds_path, raster_dir, target_size
        )

    # --- Check D: Visualization ---
    print(f"\n--- Check D: Visualization ({mode}) ---")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"sanity_check_{mode}.png"
    if mode == "sparse":
        visualize_sparse(lookup, ds_path, plot_path, seed=args.seed)
    else:
        visualize_dense(
            lookup,
            ds_path,
            raster_dir,
            target_size,
            plot_path,
            seed=args.seed,
            priority_windows=failed_c,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
