"""Create a distribution-preserving subset of Canada NBAC for OlmoEarth evaluation.

Subsamples train and val windows while preserving the joint (month, LC, FWI)
distribution and enforcing a ~2:1 NEG:POS ratio.  Tags selected windows with
``"oep_eval": ""`` in ``metadata.json`` options.

Data sources for stratification variables:
- **month** -- parsed from window directory name.
- **LC** (land cover) -- looked up from the grid-LC GDB via ``grid_id``.
- **FWI** (fire weather index) -- negatives from the negative-samples GDB,
  positives computed from FWI NetCDF via nearest-neighbour spatial+temporal
  lookup.

FWI is binned with the same edges used during negative sampling:
``[0, 2, 4, 6, 10, 15, 20, 25, 30, 40, 100]``.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DATASET_PATH = "/weka/dfive-default/rslearn-eai/datasets/wildfire/canada_nbac"
PREPROC_DIR = Path("./datasets/Canada_nbac")

NEG_GDB_PATH = PREPROC_DIR / "burned_area_preproc" / "negative_samples_unified.gdb"
POS_GDB_PATH = (
    PREPROC_DIR / "burned_area_preproc" / "temporally_gridded_fire_samples.gdb"
)
GRID_LC_GDB_PATH = PREPROC_DIR / "sampling_analysis" / "Canada_grid_adaptive_lc.gdb"
FWI_BASE_PATH = PREPROC_DIR / "fwi"

FWI_BINS = [0, 2, 4, 6, 10, 15, 20, 25, 30, 40, 100]
MANIFEST_FILENAME = "oep_eval_manifest.json"


# ---------------------------------------------------------------------------
# Window name parsing
# ---------------------------------------------------------------------------
def parse_window_name(name: str) -> dict:
    """Parse ``{Region}_{grid_id}_{POS|NEG}_{YYYYMMDD}`` into components."""
    parts = name.split("_")
    date_str = parts[-1]
    sample_type = parts[-2]
    grid_id = int(parts[-3])
    region = "_".join(parts[:-3])
    start_date = datetime.strptime(date_str, "%Y%m%d")
    return {
        "window_name": name,
        "region": region,
        "grid_id": grid_id,
        "is_positive": sample_type == "POS",
        "start_date": start_date,
        "start_date_str": start_date.strftime("%Y-%m-%d"),
        "month": start_date.month,
        "year": start_date.year,
    }


# ---------------------------------------------------------------------------
# LC lookup
# ---------------------------------------------------------------------------
def load_lc_lookup(grid_lc_path: Path) -> dict[int, int]:
    """Return ``{grid_id: lc}`` from the grid-LC GDB."""
    print(f"Loading LC grid from {grid_lc_path} ...")
    grid_lc = gpd.read_file(grid_lc_path)
    lc_map = dict(zip(grid_lc["id"].astype(int), grid_lc["lc"].astype(int)))
    print(f"  {len(lc_map)} grid cells loaded")
    return lc_map


# ---------------------------------------------------------------------------
# FWI lookup – negatives (from GDB)
# ---------------------------------------------------------------------------
def load_neg_fwi_lookup(neg_gdb_path: Path) -> pd.DataFrame:
    """Return DataFrame with ``(grid_id, start_date_str, fwi)`` for negatives."""
    print(f"Loading negative-sample FWI from {neg_gdb_path} ...")
    neg = gpd.read_file(neg_gdb_path)
    neg["start_date"] = pd.to_datetime(neg["start_date"])
    neg["start_date_str"] = neg["start_date"].dt.strftime("%Y-%m-%d")
    out = neg[["grid_id", "start_date_str", "fwinx_mean"]].copy()
    out = out.rename(columns={"fwinx_mean": "fwi"})
    out["grid_id"] = out["grid_id"].astype(int)
    out = out.drop_duplicates(subset=["grid_id", "start_date_str"])
    print(f"  {len(out)} negative FWI entries")
    return out


# ---------------------------------------------------------------------------
# FWI computation – positives (from NetCDF)
# ---------------------------------------------------------------------------
def compute_pos_fwi_lookup(
    pos_gdb_path: Path,
    fwi_base_path: Path,
) -> pd.DataFrame:
    """Compute FWI for positives via nearest-neighbour lookup in FWI NetCDF.

    Returns DataFrame with ``(grid_id, start_date_str, fwi)``.
    """
    print(f"Loading positive samples from {pos_gdb_path} ...")
    pos = gpd.read_file(pos_gdb_path)
    pos["start_date"] = pd.to_datetime(pos["start_date"])
    pos["grid_id"] = pos["grid_id"].astype(int)

    records: list[dict] = []

    for year in sorted(pos["start_date"].dt.year.unique()):
        year_pos = pos[pos["start_date"].dt.year == year]
        fwi_file = fwi_base_path / str(year) / f"fwi_dc_agg_{year}.nc"

        if not fwi_file.exists():
            print(f"  WARNING: FWI file not found for {year}: {fwi_file}")
            continue

        print(f"  Year {year}: {len(year_pos)} positives ...")
        ds = xr.open_dataset(fwi_file)

        lats = xr.DataArray(year_pos["center_y"].values, dims="points")
        lons = xr.DataArray(year_pos["center_x"].values, dims="points")
        times = xr.DataArray(
            pd.DatetimeIndex(year_pos["start_date"]).values, dims="points"
        )

        values = (
            ds["fwinx_mean"]
            .sel(latitude=lats, longitude=lons, valid_time=times, method="nearest")
            .values
        )
        ds.close()

        for idx, (_, row) in enumerate(year_pos.iterrows()):
            val = float(values[idx])
            if not np.isnan(val):
                records.append(
                    {
                        "grid_id": int(row["grid_id"]),
                        "start_date_str": row["start_date"].strftime("%Y-%m-%d"),
                        "fwi": val,
                    }
                )

    out = pd.DataFrame(records)
    if len(out) > 0:
        out = out.drop_duplicates(subset=["grid_id", "start_date_str"])
    print(f"  Computed FWI for {len(out)} positive samples")
    return out


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------
def stratified_sample(
    df: pd.DataFrame,
    target: int,
    stratum_col: str,
    seed: int,
) -> pd.DataFrame:
    """Sample *target* rows proportionally to the *stratum_col* distribution.

    Uses largest-remainder allocation followed by a redistribute step for
    strata that are too small to fulfil their quota.
    """
    rng = random.Random(seed)

    if len(df) <= target:
        return df

    stratum_counts = df[stratum_col].value_counts()
    total = len(df)

    exact = {s: c / total * target for s, c in stratum_counts.items()}
    allocations = {s: int(v) for s, v in exact.items()}
    remainders = {s: exact[s] - allocations[s] for s in exact}

    allocated_sum = sum(allocations.values())
    remainder_budget = target - allocated_sum
    for s in sorted(remainders, key=lambda k: -remainders[k])[:remainder_budget]:
        allocations[s] += 1

    sampled_indices: list[int] = []
    shortfall = 0
    strata_with_surplus: list[str] = []

    for stratum, alloc in allocations.items():
        stratum_df = df[df[stratum_col] == stratum]
        avail = len(stratum_df)
        take = min(alloc, avail)
        if take < alloc:
            shortfall += alloc - take
        if avail > alloc:
            strata_with_surplus.append(stratum)
        if take > 0:
            idx = list(stratum_df.index)
            rng.shuffle(idx)
            sampled_indices.extend(idx[:take])

    if shortfall > 0 and strata_with_surplus:
        already = set(sampled_indices)
        pool = df[df[stratum_col].isin(strata_with_surplus) & ~df.index.isin(already)]
        if len(pool) > 0:
            extra = list(pool.index)
            rng.shuffle(extra)
            sampled_indices.extend(extra[:shortfall])

    return df.loc[sampled_indices]


# ---------------------------------------------------------------------------
# Metadata I/O
# ---------------------------------------------------------------------------
def _load_metadata(window_path: Path) -> dict:
    with open(window_path / "metadata.json") as f:
        return json.load(f)


def _save_metadata(window_path: Path, metadata: dict) -> None:
    with open(window_path / "metadata.json", "w") as f:
        json.dump(metadata, f)


# ---------------------------------------------------------------------------
# Manifest (for idempotent re-runs)
# ---------------------------------------------------------------------------
def _load_manifest(dataset_path: Path) -> dict[str, list[str]]:
    path = dataset_path / MANIFEST_FILENAME
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_manifest(
    dataset_path: Path,
    selected_by_split: dict[str, list[str]],
) -> None:
    with open(dataset_path / MANIFEST_FILENAME, "w") as f:
        json.dump(selected_by_split, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def create_oep_eval_subset(
    dataset_path: str,
    train_size: int = 1000,
    val_size: int = 500,
    neg_pos_ratio: float = 2.0,
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    """Create distribution-preserving subsets for OEP evaluation.

    Returns a statistics dict.
    """
    ds_root = Path(dataset_path)
    splits_config = {"train": train_size, "val": val_size}

    # ------------------------------------------------------------------
    # Step 1: Parse window inventory
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Parsing window inventory")
    print("=" * 60)

    all_records: list[dict] = []
    for split in splits_config:
        win_dir = ds_root / "windows" / split
        if not win_dir.exists():
            raise FileNotFoundError(f"Not found: {win_dir}")
        print(f"  Listing {split} windows ...")
        names = [d.name for d in win_dir.iterdir() if d.is_dir()]
        print(f"    {len(names)} windows")
        for n in names:
            rec = parse_window_name(n)
            rec["split"] = split
            all_records.append(rec)

    df = pd.DataFrame(all_records)
    print(f"\nTotal: {len(df)} windows")
    for sp in splits_config:
        s = df[df["split"] == sp]
        print(
            f"  {sp}: {len(s)} "
            f"({s['is_positive'].sum()} POS + {(~s['is_positive']).sum()} NEG)"
        )

    # ------------------------------------------------------------------
    # Step 2: Assign LC
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Assigning land cover")
    print("=" * 60)

    lc_map = load_lc_lookup(GRID_LC_GDB_PATH)
    df["lc"] = df["grid_id"].map(lc_map).fillna(-999).astype(int)
    n_miss = (df["lc"] == -999).sum()
    if n_miss:
        print(f"  WARNING: {n_miss} windows without LC mapping")

    # ------------------------------------------------------------------
    # Step 3: Assign FWI
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Assigning FWI")
    print("=" * 60)

    neg_fwi = load_neg_fwi_lookup(NEG_GDB_PATH)
    pos_fwi = compute_pos_fwi_lookup(POS_GDB_PATH, FWI_BASE_PATH)

    df_neg = df[~df["is_positive"]].merge(
        neg_fwi, on=["grid_id", "start_date_str"], how="left"
    )
    df_pos = df[df["is_positive"]].merge(
        pos_fwi, on=["grid_id", "start_date_str"], how="left"
    )
    df = pd.concat([df_pos, df_neg], ignore_index=True)

    n_fwi_ok = df["fwi"].notna().sum()
    print(f"\n  FWI coverage: {n_fwi_ok}/{len(df)} ({100 * n_fwi_ok / len(df):.1f}%)")

    # ------------------------------------------------------------------
    # Step 4: Bin FWI → combined stratum
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Creating strata (month, LC, FWI bin)")
    print("=" * 60)

    df["fwi_bin"] = (
        pd.cut(df["fwi"], bins=FWI_BINS, labels=False, include_lowest=True)
        .fillna(-1)
        .astype(int)
    )
    df["stratum"] = (
        df["month"].astype(str)
        + "_"
        + df["lc"].astype(str)
        + "_"
        + df["fwi_bin"].astype(str)
    )
    print(f"  {df['stratum'].nunique()} unique strata")

    # ------------------------------------------------------------------
    # Step 5: Stratified sampling
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 5: Stratified sampling")
    print("=" * 60)

    all_selected: list[pd.DataFrame] = []
    split_stats: dict[str, dict] = {}

    for split, target_size in splits_config.items():
        print(f"\n--- {split} (target {target_size}) ---")
        sdf = df[df["split"] == split].copy()

        target_neg = round(target_size * neg_pos_ratio / (1 + neg_pos_ratio))
        target_pos = target_size - target_neg
        print(f"  Target: {target_pos} POS + {target_neg} NEG")

        pos_pool = sdf[sdf["is_positive"]].copy()
        neg_pool = sdf[~sdf["is_positive"]].copy()
        print(f"  Pool:   {len(pos_pool)} POS + {len(neg_pool)} NEG")

        s_pos = stratified_sample(pos_pool, target_pos, "stratum", seed)
        s_neg = stratified_sample(neg_pool, target_neg, "stratum", seed + 1)

        selected = pd.concat([s_pos, s_neg])
        all_selected.append(selected)
        print(f"  Selected: {len(s_pos)} POS + {len(s_neg)} NEG = {len(selected)}")

        split_stats[split] = {
            "total": len(sdf),
            "total_pos": int(sdf["is_positive"].sum()),
            "total_neg": int((~sdf["is_positive"]).sum()),
            "selected": len(selected),
            "selected_pos": len(s_pos),
            "selected_neg": len(s_neg),
            "target_pos": target_pos,
            "target_neg": target_neg,
        }

    selected_df = pd.concat(all_selected, ignore_index=True)
    selected_by_split: dict[str, list[str]] = {}
    for split in splits_config:
        names = sorted(
            selected_df[selected_df["split"] == split]["window_name"].tolist()
        )
        selected_by_split[split] = names

    # ------------------------------------------------------------------
    # Step 6: Tag windows
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 6: Tagging selected windows")
    print("=" * 60)

    old_manifest = _load_manifest(ds_root)
    tagged_add = 0
    tagged_remove = 0

    # Remove old tags (idempotency)
    for split, old_names in old_manifest.items():
        win_dir = ds_root / "windows" / split
        new_set = set(selected_by_split.get(split, []))
        to_remove = [n for n in old_names if n not in new_set]
        if to_remove:
            print(f"  Removing old tags from {len(to_remove)} {split} windows ...")
        for name in tqdm(to_remove, desc=f"  Untag {split}", disable=not to_remove):
            wp = win_dir / name
            if not (wp / "metadata.json").exists():
                continue
            meta = _load_metadata(wp)
            if "oep_eval" in meta.get("options", {}):
                meta["options"].pop("oep_eval")
                if not dry_run:
                    _save_metadata(wp, meta)
                tagged_remove += 1

    # Add new tags
    for split, names in selected_by_split.items():
        win_dir = ds_root / "windows" / split
        print(f"  Tagging {len(names)} {split} windows ...")
        for name in tqdm(names, desc=f"  Tag {split}"):
            wp = win_dir / name
            meta = _load_metadata(wp)
            meta["options"]["oep_eval"] = ""
            if not dry_run:
                _save_metadata(wp, meta)
            tagged_add += 1

    # Save manifest for future cleanup
    if not dry_run:
        _save_manifest(ds_root, selected_by_split)
        print(f"  Manifest saved to {ds_root / MANIFEST_FILENAME}")

    # ------------------------------------------------------------------
    # Step 7: Print statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for split, ss in split_stats.items():
        print(f"\n  {split}:")
        print(
            f"    Original: {ss['total']} "
            f"({ss['total_pos']} POS + {ss['total_neg']} NEG)"
        )
        print(
            f"    Selected: {ss['selected']} "
            f"({ss['selected_pos']} POS + {ss['selected_neg']} NEG)"
        )
        ratio = (
            ss["selected_neg"] / ss["selected_pos"]
            if ss["selected_pos"] > 0
            else float("inf")
        )
        print(f"    NEG:POS ratio: {ratio:.2f}:1")

    # Stratum distribution comparison
    print("\n  Stratum distribution (top 15 by count):")
    for split in splits_config:
        sdf = df[df["split"] == split]
        sel = selected_df[selected_df["split"] == split]
        orig_dist = sdf["stratum"].value_counts(normalize=True).head(15)
        sel_dist = sel["stratum"].value_counts(normalize=True)
        print(f"\n    {split}:")
        print(f"    {'stratum':<30s} {'orig%':>7s} {'sel%':>7s} {'diff':>7s}")
        for stratum in orig_dist.index:
            o = orig_dist[stratum] * 100
            s = sel_dist.get(stratum, 0) * 100
            print(f"    {stratum:<30s} {o:6.2f}% {s:6.2f}% {s - o:+6.2f}%")

    print(f"\n  Tags added: {tagged_add}")
    print(f"  Tags removed (old): {tagged_remove}")
    if dry_run:
        print("\n  [DRY RUN – no changes written]")
    print("=" * 60)

    stats = {
        "splits": split_stats,
        "tagged_add": tagged_add,
        "tagged_remove": tagged_remove,
        "seed": seed,
        "neg_pos_ratio": neg_pos_ratio,
        "fwi_bins": FWI_BINS,
        "selected_by_split": selected_by_split,
    }
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Build the OEP evaluation subset and write the dataset statistics summary."""
    parser = argparse.ArgumentParser(
        description=(
            "Create OEP evaluation subset of Canada NBAC dataset, "
            "preserving (month, LC, FWI) distribution"
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=DATASET_PATH,
        help="Path to the rslearn dataset root",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=1000,
        help="Target number of train windows (default: 1000)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="Target number of val windows (default: 500)",
    )
    parser.add_argument(
        "--neg-pos-ratio",
        type=float,
        default=2.0,
        help="Target NEG:POS ratio (default: 2.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be done without writing changes",
    )
    args = parser.parse_args()

    stats = create_oep_eval_subset(
        dataset_path=args.dataset_path,
        train_size=args.train_size,
        val_size=args.val_size,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    stats_path = Path(args.dataset_path) / "oep_eval_stats.json"
    if not args.dry_run:
        serialisable = {k: v for k, v in stats.items() if k != "selected_by_split"}
        with open(stats_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nStatistics saved to {stats_path}")


if __name__ == "__main__":
    main()
