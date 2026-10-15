"""Build a balanced 4k subsample (3k train + 1k val) from MapBiomas validation points.

Criteria:
- Balanced class representation across as many classes as possible (water-filling).
- Uniform sampling across CARTA_2 tiles.
- Best-effort DECLIVIDAD representation and year uniformity (2016-2022).
- Only best data points (COUNT_year == 3).
- 25% edge pixels (BORDA_year == 1), 75% interior (BORDA_year == 0) best-effort.
- Each spatial pixel (TARGETID) used at most once across all years.
- Excluded classes: 13, 23, 27, 30, 31, 32, 50.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

MY_ROOT = Path(os.environ.get("MY_ROOT", "."))

DEFAULT_LEGEND = (
    MY_ROOT / "datasets/mapbiomas/metadata/Codigos-da-legenda-colecao-10.csv"
)
DEFAULT_HIERARCHY = MY_ROOT / "datasets/mapbiomas/metadata/hierarchy.csv"

EXCLUDED_CLASSES: set[int] = {
    13,
    23,
    27,
    30,
    31,
    32,
    50,
}  # less than 100 points in data
YEAR_RANGE: range = range(2016, 2023)  # 2016-2022 inclusive


# ---------------------------------------------------------------------------
# Step 1: load & melt
# ---------------------------------------------------------------------------


def load_and_melt(shp_path: str | Path) -> pd.DataFrame:
    """Load the shapefile and melt year-specific columns into long format.

    Returns a DataFrame with columns:
        TARGETID, LON, LAT, YEAR, CLASS, BORDA, COUNT, CARTA_2, DECLIVIDAD
    filtered to COUNT == 3 and classes not in EXCLUDED_CLASSES.
    """
    gdf = gpd.read_file(shp_path)

    static_cols = ["TARGETID", "LON", "LAT", "CARTA_2", "DECLIVIDAD"]
    frames: list[pd.DataFrame] = []
    for year in YEAR_RANGE:
        cls_col = f"CLASS_{year}"
        cnt_col = f"COUNT_{year}"
        brd_col = f"BORDA_{year}"
        if cls_col not in gdf.columns:
            continue
        sub = gdf[static_cols + [cls_col, cnt_col, brd_col]].copy()
        sub.columns = [*static_cols, "CLASS", "COUNT", "BORDA"]
        sub["YEAR"] = year
        sub["COUNT"] = pd.to_numeric(sub["COUNT"], errors="coerce")
        sub["CLASS"] = pd.to_numeric(sub["CLASS"], errors="coerce")
        sub["BORDA"] = pd.to_numeric(sub["BORDA"], errors="coerce")
        frames.append(sub)

    long = pd.concat(frames, ignore_index=True)

    # Filter: best quality only, drop excluded classes and NaN class
    long = long[long["COUNT"] == 3].copy()
    long = long[long["CLASS"].notna()].copy()
    long["CLASS"] = long["CLASS"].astype(int)
    long = long[~long["CLASS"].isin(EXCLUDED_CLASSES)].copy()

    long = long.reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# Step 2: water-filling quota
# ---------------------------------------------------------------------------


def compute_quotas(
    long: pd.DataFrame,
    total: int,
) -> dict[int, int]:
    """Water-fill per-class quotas so sum == total, capped by unique-pixel availability."""
    avail = long.groupby("CLASS")["TARGETID"].nunique().to_dict()
    classes = sorted(avail.keys())

    sorted_caps = sorted(avail[c] for c in classes)
    n = len(classes)
    level = 0.0
    remaining = total

    for i, cap in enumerate(sorted_caps):
        slots = n - i
        if (cap - level) * slots >= remaining:
            level += remaining / slots
            break
        remaining -= int(cap - level) * slots
        level = cap
    else:
        level = sorted_caps[-1]

    base = int(np.floor(level))
    quotas = {c: min(avail[c], base) for c in classes}
    deficit = total - sum(quotas.values())

    # distribute remainder one-at-a-time to classes that still have room
    frac_parts: list[tuple[float, int]] = []
    for c in classes:
        headroom = avail[c] - quotas[c]
        if headroom > 0:
            frac_parts.append((level - int(np.floor(level)), c))
    frac_parts.sort(key=lambda x: -x[0])

    idx = 0
    while deficit > 0:
        for c in classes:
            if deficit <= 0:
                break
            if avail[c] - quotas[c] > 0:
                quotas[c] += 1
                deficit -= 1
        idx += 1
        if idx > len(classes):
            break

    return quotas


# ---------------------------------------------------------------------------
# Step 3: stratified selection (rarest class first)
# ---------------------------------------------------------------------------


def select_pixels(
    long: pd.DataFrame,
    quotas: dict[int, int],
    edge_frac: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Select pixels per class, rarest first, with CARTA_2/DECLIVIDAD/YEAR balance.

    Each TARGETID is used at most once globally.  Edge/interior mix is best-effort.
    """
    used_ids: set[int] = set()
    selected_rows: list[pd.DataFrame] = []

    class_order = sorted(quotas.keys(), key=lambda c: quotas[c])
    active_classes = [c for c in class_order if quotas[c] > 0]
    total_quota = sum(quotas[c] for c in active_classes)
    cumulative = 0

    for i, cls in enumerate(active_classes, 1):
        quota = quotas[cls]

        pool = long[(long["CLASS"] == cls) & (~long["TARGETID"].isin(used_ids))].copy()
        if pool.empty:
            print(
                f"  [{i}/{len(active_classes)}] class {cls:>2d}: "
                f"quota={quota}, pool empty — skipped"
            )
            continue

        n_edge_target = int(round(edge_frac * quota))
        n_interior_target = quota - n_edge_target

        edge_pool = pool[pool["BORDA"] == 1]
        interior_pool = pool[pool["BORDA"] == 0]

        chosen_edge = _stratified_pick(
            edge_pool,
            n_edge_target,
            used_ids,
            rng,
            label=f"class {cls} edge",
        )
        newly_used = set(chosen_edge["TARGETID"])
        used_ids.update(newly_used)

        interior_pool = interior_pool[~interior_pool["TARGETID"].isin(used_ids)]
        chosen_interior = _stratified_pick(
            interior_pool,
            n_interior_target,
            used_ids,
            rng,
            label=f"class {cls} interior",
        )
        used_ids.update(chosen_interior["TARGETID"])

        # Backfill shortfall from the other pool
        got = len(chosen_edge) + len(chosen_interior)
        shortfall = quota - got
        if shortfall > 0:
            remaining = pool[~pool["TARGETID"].isin(used_ids)]
            backfill = _stratified_pick(
                remaining,
                shortfall,
                used_ids,
                rng,
                label=f"class {cls} backfill",
            )
            used_ids.update(backfill["TARGETID"])
            selected_rows.extend([chosen_edge, chosen_interior, backfill])
            got += len(backfill)
        else:
            selected_rows.extend([chosen_edge, chosen_interior])

        cumulative += got
        print(
            f"  [{i}/{len(active_classes)}] class {cls:>2d}: "
            f"picked {got}/{quota} — cumulative {cumulative}/{total_quota}"
        )

    result = pd.concat(selected_rows, ignore_index=True)
    return result


def _stratified_pick(
    pool: pd.DataFrame,
    n: int,
    used_ids: set[int],
    rng: np.random.Generator,
    label: str = "",
) -> pd.DataFrame:
    """Pick up to *n* rows from *pool* spread across CARTA_2 / DECLIVIDAD / YEAR.

    Within each tile, prefer rows that fill under-represented DECLIVIDAD and YEAR
    buckets.  Each picked TARGETID is immediately added to used_ids so no pixel
    repeats.
    """
    if n <= 0 or pool.empty:
        return pool.iloc[:0]

    pool = pool[~pool["TARGETID"].isin(used_ids)].copy()
    if pool.empty:
        return pool

    pool = pool.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(
        drop=True
    )

    tiles = sorted(pool["CARTA_2"].unique())
    tile_pools: dict[str, pd.DataFrame] = {
        t: pool[pool["CARTA_2"] == t].copy() for t in tiles
    }

    year_counts: Counter[int] = Counter()
    decliv_counts: Counter[str] = Counter()
    picked_indices: list[int] = []
    local_used: set[int] = set()

    log_interval = max(1, n // 5)
    tag = f"    {label}: " if label else "    pick: "

    tile_idx = 0
    stall_counter = 0
    while len(picked_indices) < n and stall_counter < len(tiles) + 1:
        tile = tiles[tile_idx % len(tiles)]
        tile_idx += 1
        tp = tile_pools[tile]
        tp = tp[~tp["TARGETID"].isin(local_used)]
        tile_pools[tile] = tp
        if tp.empty:
            stall_counter += 1
            continue
        stall_counter = 0

        scores = np.zeros(len(tp))
        for i, (_, row) in enumerate(tp.iterrows()):
            yr = row["YEAR"]
            dc = row["DECLIVIDAD"]
            scores[i] = -(year_counts.get(yr, 0) + decliv_counts.get(dc, 0))

        best_idx = int(np.argmax(scores))
        chosen_row = tp.iloc[best_idx]
        picked_indices.append(tp.index[best_idx])
        tid = chosen_row["TARGETID"]
        local_used.add(tid)
        used_ids.add(tid)
        year_counts[chosen_row["YEAR"]] += 1
        decliv_counts[chosen_row["DECLIVIDAD"]] += 1

        for t2 in tiles:
            tp2 = tile_pools[t2]
            if not tp2.empty:
                tile_pools[t2] = tp2[tp2["TARGETID"] != tid]

        picked = len(picked_indices)
        if picked % log_interval == 0 or picked == n:
            print(f"{tag}{picked}/{n}", flush=True)

    if not picked_indices:
        return pool.iloc[:0]
    return pool.loc[pool.index.isin(picked_indices)].copy()


# ---------------------------------------------------------------------------
# Step 4: train / val split
# ---------------------------------------------------------------------------


def train_val_split(
    selected: pd.DataFrame,
    n_train: int,
    n_val: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Stratified per-class 75/25 train/val split."""
    total = n_train + n_val
    train_frac = n_train / total

    parts: list[pd.DataFrame] = []
    for cls, grp in selected.groupby("CLASS"):
        grp = grp.sample(frac=1, random_state=int(rng.integers(2**31)))
        n_tr = int(round(len(grp) * train_frac))
        grp = grp.copy()
        grp["split"] = "val"
        grp.iloc[:n_tr, grp.columns.get_loc("split")] = "train"
        parts.append(grp)

    result = pd.concat(parts, ignore_index=True)

    # Adjust global counts to hit exact targets
    n_train_actual = (result["split"] == "train").sum()
    diff = n_train_actual - n_train
    if diff > 0:
        train_idx = (
            result[result["split"] == "train"]
            .sample(n=diff, random_state=int(rng.integers(2**31)))
            .index
        )
        result.loc[train_idx, "split"] = "val"
    elif diff < 0:
        val_idx = (
            result[result["split"] == "val"]
            .sample(n=-diff, random_state=int(rng.integers(2**31)))
            .index
        )
        result.loc[val_idx, "split"] = "train"

    return result


# ---------------------------------------------------------------------------
# Step 5: summary report
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
    df: pd.DataFrame,
    legend: dict[int, str],
    hierarchy: dict[int, dict] | None = None,
) -> pd.DataFrame:
    """Build a class-level summary with train/val breakdown and diversity metrics."""
    n_total = len(df)
    rows: list[dict] = []

    for cls, grp in df.groupby("CLASS"):
        n = len(grp)
        cid = int(cls)
        train_mask = grp["split"] == "train"
        n_train = int(train_mask.sum())
        n_val = n - n_train

        rec: dict = {
            "class_id": cid,
            "class_name": legend.get(cid, "unknown"),
            "leaf_level": None,
            "parent_class_id": None,
            "parent_leaf_level": None,
            "parent_class_desc": None,
            "total_points": n,
            "frac_of_all": n / n_total,
            "train_points": n_train,
            "train_frac": n_train / n if n else 0.0,
            "val_points": n_val,
            "val_frac": n_val / n if n else 0.0,
            "edge_frac": float((grp["BORDA"] == 1).mean()),
            "n_tiles": int(grp["CARTA_2"].nunique()),
            "n_years": int(grp["YEAR"].nunique()),
        }

        if hierarchy and cid in hierarchy:
            hi = hierarchy[cid]
            rec["leaf_level"] = hi["leaf_level"]
            rec["parent_class_id"] = hi["parent_class_id"]
            rec["parent_leaf_level"] = hi["parent_leaf_level"]
            rec["parent_class_desc"] = hi["parent_class_desc"]

        rows.append(rec)

    summary = (
        pd.DataFrame(rows)
        .sort_values("total_points", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def print_summary(
    df: pd.DataFrame,
    summary: pd.DataFrame | None = None,
) -> None:
    """Print a diagnostic summary of the subsample."""
    print("\n" + "=" * 70)
    print("SUBSAMPLE SUMMARY")
    print("=" * 70)

    print(f"\nTotal rows: {len(df)}")
    print(f"  Train: {(df['split'] == 'train').sum()}")
    print(f"  Val:   {(df['split'] == 'val').sum()}")
    print(f"  Unique TARGETIDs: {df['TARGETID'].nunique()} (should == {len(df)})")

    print("\n--- Per-class counts ---")
    cls_split = df.groupby(["CLASS", "split"]).size().unstack(fill_value=0)
    cls_split["total"] = cls_split.sum(axis=1)
    print(cls_split.to_string())

    if (
        summary is not None
        and "leaf_level" in summary.columns
        and summary["leaf_level"].notna().any()
    ):
        print("\n--- Class hierarchy ---")
        fmt = "{:<6s}  {:<35s}  {:<5s}  {:<6s}  {:<5s}  {:<25s}"
        print(fmt.format("ID", "Class", "Lvl", "ParID", "ParLv", "Parent"))
        print("-" * 90)
        for _, r in summary.sort_values("class_id").iterrows():
            lvl = str(int(r["leaf_level"])) if pd.notna(r["leaf_level"]) else ""
            par_id = (
                str(int(r["parent_class_id"]))
                if pd.notna(r["parent_class_id"])
                else "-"
            )
            par_lv = (
                str(int(r["parent_leaf_level"]))
                if pd.notna(r["parent_leaf_level"])
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
                    str(r["class_name"])[:35],
                    lvl,
                    par_id,
                    par_lv,
                    par_desc,
                )
            )

    print("\n--- BORDA distribution ---")
    borda_split = df.groupby(["BORDA", "split"]).size().unstack(fill_value=0)
    borda_split["total"] = borda_split.sum(axis=1)
    print(borda_split.to_string())
    n_edge = (df["BORDA"] == 1).sum()
    print(f"  Edge fraction: {n_edge / len(df):.3f} (target 0.250)")

    print("\n--- YEAR distribution ---")
    year_split = df.groupby(["YEAR", "split"]).size().unstack(fill_value=0)
    year_split["total"] = year_split.sum(axis=1)
    print(year_split.to_string())

    print("\n--- DECLIVIDAD distribution ---")
    dec_split = df.groupby(["DECLIVIDAD", "split"]).size().unstack(fill_value=0)
    dec_split["total"] = dec_split.sum(axis=1)
    print(dec_split.to_string())

    print("\n--- CARTA_2 tile coverage ---")
    print(f"  Tiles represented: {df['CARTA_2'].nunique()}")

    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build a balanced MapBiomas subsample (3k train + 1k val)."""
    parser = argparse.ArgumentParser(
        description="Build a balanced MapBiomas subsample (3k train + 1k val)."
    )
    parser.add_argument(
        "--shp-path",
        type=str,
        default=str(
            MY_ROOT / "datasets/mapbiomas/metadata/mapbiomas_85k_points_validation.shp"
        ),
        help="Path to the 85k validation shapefile.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=str(
            MY_ROOT
            / "rslearn_projects/rslp/mapbiomas/subsampling/sample_expert_points_4k.csv"
        ),
        help="Output CSV path.",
    )
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-frac", type=float, default=0.25)
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

    total = args.n_train + args.n_val
    rng = np.random.default_rng(args.seed)

    print(f"Loading shapefile: {args.shp_path}")
    long = load_and_melt(args.shp_path)
    n_unique = long["TARGETID"].nunique()
    n_classes = long["CLASS"].nunique()
    print(f"  Long-format rows (COUNT==3, kept classes): {len(long)}")
    print(f"  Unique pixels: {n_unique}, Classes: {n_classes}")

    quotas = compute_quotas(long, total)
    print(f"\nPer-class quotas (total {sum(quotas.values())}):")
    for cls in sorted(quotas, key=lambda c: -quotas[c]):
        avail = long[long["CLASS"] == cls]["TARGETID"].nunique()
        print(f"  class {cls:>2d}: quota={quotas[cls]:>4d}  (avail={avail})")

    print("\nSelecting pixels …")
    selected = select_pixels(long, quotas, args.edge_frac, rng)
    print(f"  Selected: {len(selected)} rows")

    selected = train_val_split(selected, args.n_train, args.n_val, rng)

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
    selected = (
        selected[out_cols]
        .sort_values(["split", "CLASS", "YEAR"])
        .reset_index(drop=True)
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)
    print(f"\nWrote {len(selected)} rows to {out_path}")

    legend = load_legend(Path(args.legend_path))
    hierarchy_path = Path(args.hierarchy_path)
    hierarchy = load_hierarchy(hierarchy_path) if hierarchy_path.exists() else None
    summary = build_summary(selected, legend, hierarchy)
    summary_path = out_path.parent / "sample_expert_points_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote summary stats:      {summary_path}  ({len(summary)} rows)")

    print_summary(selected, summary)


if __name__ == "__main__":
    main()
