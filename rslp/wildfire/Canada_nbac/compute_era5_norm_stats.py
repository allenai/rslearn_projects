"""Compute per-channel mean/std normalization stats for ERA5 layers.

Scans all train-split windows in the Canada NBAC dataset and computes
running mean/std for each of the 14 ERA5 variables, for both:
  - era5_365dhistory  (365 daily steps)
  - era5_8dforecast   (8 daily steps)

The ERA5 data source uses -9999.0 as a nodata sentinel
(see earthdatahub.py NODATA_VALUE). These are excluded from statistics.

Outputs two JSON files (one per layer) in the same directory as this script,
using the same {var: {mean, std}} format as era5d_norm_stats.json.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

DATASET_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/wildfire/canada_nbac")
SPLIT = "train"
NODATA_VALUE = -9999.0

ERA5_VARS = [
    "d2m",
    "e",
    "pev",
    "ro",
    "sp",
    "ssr",
    "ssrd",
    "str",
    "swvl1",
    "swvl2",
    "t2m",
    "tp",
    "u10",
    "v10",
]
NUM_VARS = len(ERA5_VARS)

LAYERS = {
    "era5_365dhistory": "era5_365dhistory_norm_stats.json",
    "era5_8dforecast": "era5_8dforecast_norm_stats.json",
}

OUTPUT_DIR = Path(__file__).resolve().parent


def load_era5_layer(window_path: Path, layer_name: str) -> np.ndarray | None:
    """Load a single ERA5 layer from a window directory.

    Returns array of shape [T, C] or None if missing.
    """
    npy_paths = list((window_path / "layers" / layer_name).rglob("data.npy"))
    if not npy_paths:
        return None
    arr = np.load(npy_paths[0])  # [C, T, 1, 1]
    c = arr.shape[0]
    if c != NUM_VARS:
        raise ValueError(f"Expected {NUM_VARS} channels, got {c} in {window_path.name}")
    return arr[:, :, 0, 0].T  # -> [T, C]


def compute_norm_stats(
    layer_name: str,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Accumulate running stats across all train windows for one layer."""
    train_root = DATASET_ROOT / "windows" / SPLIT
    if not train_root.exists():
        raise RuntimeError(f"Missing split directory: {train_root}")

    sum_vec = np.zeros(NUM_VARS, dtype=np.float64)
    sq_sum_vec = np.zeros(NUM_VARS, dtype=np.float64)
    count_vec = np.zeros(NUM_VARS, dtype=np.int64)
    nodata_count_vec = np.zeros(NUM_VARS, dtype=np.int64)

    missing_layer = 0
    total_windows = 0
    # Per-channel: windows where every timestep is nodata (fully invalid pixel)
    fully_nodata_windows = np.zeros(NUM_VARS, dtype=np.int64)
    # Per-channel: windows where some but not all timesteps are nodata
    partially_nodata_windows = np.zeros(NUM_VARS, dtype=np.int64)

    window_dirs = sorted(p for p in train_root.iterdir() if p.is_dir())
    n_windows = len(window_dirs)

    for idx, window_dir in enumerate(window_dirs):
        if idx % 5000 == 0:
            print(
                f"  [{layer_name}] {idx}/{n_windows} windows processed ...",
                flush=True,
            )
        total_windows += 1
        x = load_era5_layer(window_dir, layer_name)

        if x is None:
            missing_layer += 1
            continue

        # x shape: [T, C]
        is_nodata = x == NODATA_VALUE
        valid_mask = np.isfinite(x) & ~is_nodata

        nodata_per_channel = is_nodata.sum(axis=0)  # [C]
        nodata_count_vec += nodata_per_channel.astype(np.int64)
        t_steps = x.shape[0]
        for c_idx in range(NUM_VARS):
            nd = int(nodata_per_channel[c_idx])
            if nd == t_steps:
                fully_nodata_windows[c_idx] += 1
            elif nd > 0:
                partially_nodata_windows[c_idx] += 1

        x64 = x.astype(np.float64, copy=False)
        x_safe = np.where(valid_mask, x64, 0.0)

        sum_vec += x_safe.sum(axis=0)
        sq_sum_vec += np.square(x_safe).sum(axis=0)
        count_vec += valid_mask.sum(axis=0).astype(np.int64)

    if np.any(count_vec == 0):
        bad = [ERA5_VARS[i] for i in np.where(count_vec == 0)[0]]
        raise RuntimeError(f"No valid values for variable(s): {bad}")

    mean_vec = sum_vec / count_vec
    var_vec = (sq_sum_vec / count_vec) - np.square(mean_vec)
    var_vec = np.clip(var_vec, a_min=0.0, a_max=None)
    std_vec = np.sqrt(var_vec)

    print(f"\n=== {layer_name} normalization stats ===")
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  Split        : {SPLIT}")
    print(f"  Total windows: {total_windows}  (missing layer: {missing_layer})")
    print()
    for i, var in enumerate(ERA5_VARS):
        total_pixels = int(count_vec[i] + nodata_count_vec[i])
        nd_pct = 100.0 * nodata_count_vec[i] / total_pixels if total_pixels else 0
        print(
            f"  {var:6s}  mean={mean_vec[i]: .8f}  std={std_vec[i]: .8f}"
            f"  valid={int(count_vec[i])}  nodata={int(nodata_count_vec[i])} ({nd_pct:.2f}%)"
            f"  fully_nodata_windows={int(fully_nodata_windows[i])}"
            f"  partially_nodata_windows={int(partially_nodata_windows[i])}"
        )

    norm_dict = {
        var: {"mean": round(float(mean_vec[i]), 8), "std": round(float(std_vec[i]), 8)}
        for i, var in enumerate(ERA5_VARS)
    }

    diagnostics = {
        "dataset_root": str(DATASET_ROOT),
        "split": SPLIT,
        "nodata_value": NODATA_VALUE,
        "total_windows": total_windows,
        "missing_layer_windows": missing_layer,
        "per_variable": {
            var: {
                "valid_count": int(count_vec[i]),
                "nodata_count": int(nodata_count_vec[i]),
                "fully_nodata_windows": int(fully_nodata_windows[i]),
                "partially_nodata_windows": int(partially_nodata_windows[i]),
            }
            for i, var in enumerate(ERA5_VARS)
        },
    }

    return norm_dict, diagnostics


def main() -> None:
    """Compute and save normalization statistics for each configured ERA5 layer."""
    for layer_name, output_filename in LAYERS.items():
        print(f"\nProcessing layer: {layer_name}")
        stats, diagnostics = compute_norm_stats(layer_name)

        out_path = OUTPUT_DIR / output_filename
        out_path.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"  -> Saved norm stats to {out_path}")

        diag_path = out_path.with_suffix(".diagnostics.json")
        diag_path.write_text(json.dumps(diagnostics, indent=2) + "\n")
        print(f"  -> Saved diagnostics to {diag_path}")


if __name__ == "__main__":
    main()
