"""Estimate how many fire events are large and long-lasting.

This script counts (and optionally lists) fire events that satisfy **both**:

1. **Area** ≥ a fraction of a reference cell area (default: 50 % of 3.2 km × 3.2 km).
2. **Duration** ≥ a minimum number of days (default: 8 days, matching the
   temporal-bin offset).

The script is source-agnostic: column names for fire ID, start date, and end
date are passed as CLI arguments.

Usage
-----
::

    python -m data_preproc_script.preprocess.fire_stats \
        --input-path /path/to/fire_polygons.gdb \
        --fire-id-col id \
        --start-date-col ig_date \
        --end-date-col last_date \
        --min-area-km2 5.12 \
        --min-duration-days 8
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import pandas as pd


def compute_fire_stats(
    input_path: str,
    fire_id_col: str,
    start_date_col: str,
    end_date_col: str,
    min_area_km2: float = 5.12,
    min_duration_days: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load fire polygons and flag those exceeding size and duration thresholds.

    Parameters
    ----------
    input_path:
        Path to the fire-polygon file (shapefile, GeoDatabase, GeoPackage, …).
    fire_id_col:
        Column containing the unique fire-event identifier.
    start_date_col:
        Column containing the fire start date.
    end_date_col:
        Column containing the fire end date.
    min_area_km2:
        Minimum burned area in km² (default: 50 % of a 3.2 km × 3.2 km cell
        = 5.12 km²).
    min_duration_days:
        Minimum fire duration in days (default: 8).

    Returns:
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(summary, flagged_fires)`` where *summary* is a single-row
        DataFrame of aggregate statistics and *flagged_fires* contains the
        subset of events that meet **both** criteria.
    """
    # ------------------------------------------------------------------
    # 1. Load and prepare
    # ------------------------------------------------------------------
    print(f"Loading fire polygons from {input_path} ...")
    gdf = gpd.read_file(input_path)

    # Validate required columns
    for col in (fire_id_col, start_date_col, end_date_col):
        if col not in gdf.columns:
            raise ValueError(
                f"Column '{col}' not found. "
                f"Available columns: {sorted(gdf.columns.tolist())}"
            )

    # Canonical names for internal use
    gdf = gdf.rename(
        columns={
            fire_id_col: "fire_id",
            start_date_col: "start_date",
            end_date_col: "end_date",
        }
    )
    gdf["start_date"] = pd.to_datetime(gdf["start_date"])
    gdf["end_date"] = pd.to_datetime(gdf["end_date"])
    gdf = gdf.dropna(subset=["start_date", "end_date"])

    # ------------------------------------------------------------------
    # 2. Compute per-event area and duration
    # ------------------------------------------------------------------
    # Project to an equal-area CRS for accurate area computation
    gdf_ea = gdf.to_crs(epsg=6933)  # World Cylindrical Equal Area
    gdf["area_km2"] = gdf_ea.geometry.area / 1e6  # m² → km²
    gdf["duration_days"] = (gdf["end_date"] - gdf["start_date"]).dt.days

    # ------------------------------------------------------------------
    # 3. Flag fires meeting both criteria
    # ------------------------------------------------------------------
    large_mask = gdf["area_km2"] >= min_area_km2
    long_mask = gdf["duration_days"] >= min_duration_days
    both_mask = large_mask & long_mask

    total = len(gdf)
    n_large = int(large_mask.sum())
    n_long = int(long_mask.sum())
    n_both = int(both_mask.sum())

    summary = pd.DataFrame(
        [
            {
                "total_fires": total,
                f"n_large (>= {min_area_km2:.2f} km²)": n_large,
                f"n_long (>= {min_duration_days} days)": n_long,
                "n_both": n_both,
                "pct_large": round(100 * n_large / total, 2) if total else 0,
                "pct_long": round(100 * n_long / total, 2) if total else 0,
                "pct_both": round(100 * n_both / total, 2) if total else 0,
            }
        ]
    )

    flagged = gdf.loc[
        both_mask, ["fire_id", "start_date", "end_date", "area_km2", "duration_days"]
    ].copy()

    return summary, flagged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the fire-duration and fire-area summary CLI."""
    parser = argparse.ArgumentParser(
        description="Count fire events exceeding area and duration thresholds."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to the fire-polygon file.",
    )
    parser.add_argument(
        "--fire-id-col",
        required=True,
        help="Column name for the unique fire identifier.",
    )
    parser.add_argument(
        "--start-date-col",
        required=True,
        help="Column name for the fire start date.",
    )
    parser.add_argument(
        "--end-date-col",
        required=True,
        help="Column name for the fire end date.",
    )
    parser.add_argument(
        "--min-area-km2",
        type=float,
        default=5.12,
        help="Minimum area threshold in km² (default: 5.12 = 50%% of 3.2×3.2 km cell).",
    )
    parser.add_argument(
        "--min-duration-days",
        type=int,
        default=8,
        help="Minimum duration threshold in days (default: 8).",
    )
    args = parser.parse_args()

    summary, flagged = compute_fire_stats(
        input_path=args.input_path,
        fire_id_col=args.fire_id_col,
        start_date_col=args.start_date_col,
        end_date_col=args.end_date_col,
        min_area_km2=args.min_area_km2,
        min_duration_days=args.min_duration_days,
    )

    print("\n===== Fire Statistics =====")
    print(summary.to_string(index=False))
    print(f"\nLargest flagged fires (top 20 by area, of {len(flagged)} total):")
    print(
        flagged.sort_values("area_km2", ascending=False).head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
