"""Compute mean/std of LFMC target values over a split (and optional tag).

These statistics are intended to be plugged into the ``target_mean`` / ``target_std``
arguments of ``rslearn.train.tasks.per_pixel_regression.PerPixelRegressionTask`` so that
the regression target is normalized during training while metrics are still reported in
the original units.

The statistics are computed only over *valid* pixels: pixels equal to the ignore value
(``-1`` by default, matching the task's ``nodata_value``) and non-finite pixels are
excluded. Windows are filtered the same way the training data module filters them: by
group, by the ``split`` option, and (optionally) by requiring a tag option to be present.

IMPORTANT: only compute these statistics on the train split to avoid leaking validation
or test information into the normalization.

Usage:
    python -m rslp.lfmc.compute_target_stats \
        --dataset_path /weka/dfive-default/rslearn-eai/datasets/lfmc/20251023/woody/scratch/dataset \
        --split train \
        --tag oep_eval_big
"""

import argparse
import json
import math
import os
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import rasterio


@dataclass
class PixelStats:
    """Streaming accumulator for per-pixel statistics over valid pixels."""

    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def merge(self, other: "PixelStats") -> None:
        """Merge another accumulator into this one."""
        self.count += other.count
        self.total += other.total
        self.total_sq += other.total_sq
        self.minimum = min(self.minimum, other.minimum)
        self.maximum = max(self.maximum, other.maximum)

    @property
    def mean(self) -> float:
        """Mean over valid pixels."""
        if self.count == 0:
            raise ValueError("no valid pixels found")
        return self.total / self.count

    def std(self, ddof: int = 0) -> float:
        """Standard deviation over valid pixels.

        Args:
            ddof: delta degrees of freedom. Use 0 (population std) for normalization.
        """
        if self.count - ddof <= 0:
            raise ValueError("not enough valid pixels to compute std")
        # Var = E[x^2] - E[x]^2, scaled to the requested ddof. Clamp tiny negatives that
        # can arise from floating point error.
        variance = (self.total_sq - self.total**2 / self.count) / (self.count - ddof)
        return math.sqrt(max(variance, 0.0))


def _matches_filters(
    options: dict,
    split: str | None,
    tag: str | None,
) -> bool:
    """Return whether a window's options pass the split/tag filters.

    Mirrors the tag filtering done by rslearn's ModelDataset: the tag key must be present
    in the window options (an empty configured value just requires presence).
    """
    if split is not None and options.get("split") != split:
        return False
    if tag is not None and tag not in options:
        return False
    return True


def _select_window_dirs(
    dataset_path: str,
    group: str,
    split: str | None,
    tag: str | None,
) -> list[str]:
    """Find window directories matching the split/tag filters."""
    windows_dir = os.path.join(dataset_path, "windows", group)
    names = sorted(os.listdir(windows_dir))
    selected = []
    for name in names:
        window_dir = os.path.join(windows_dir, name)
        meta_path = os.path.join(window_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if _matches_filters(meta.get("options", {}), split, tag):
            selected.append(window_dir)
    return selected


# Globals set per worker process so the geotiff path can be built without re-passing
# constant arguments for every task.
_LAYER = "labels"
_IGNORE_VALUE = -1.0


def _init_worker(layer: str, ignore_value: float) -> None:
    global _LAYER, _IGNORE_VALUE
    _LAYER = layer
    _IGNORE_VALUE = ignore_value


def _window_stats(window_dir: str) -> PixelStats:
    """Read one window's label raster and accumulate stats over valid pixels."""
    stats = PixelStats()
    geotiff_path = os.path.join(window_dir, "layers", _LAYER, "value", "geotiff.tif")
    if not os.path.isfile(geotiff_path):
        return stats
    with rasterio.open(geotiff_path) as ds:
        array = ds.read().astype(np.float64)
    valid = np.isfinite(array) & (array != _IGNORE_VALUE)
    values = array[valid]
    if values.size == 0:
        return stats
    stats.count = int(values.size)
    stats.total = float(values.sum())
    stats.total_sq = float(np.square(values).sum())
    stats.minimum = float(values.min())
    stats.maximum = float(values.max())
    return stats


def compute_stats(
    window_dirs: Iterable[str],
    layer: str,
    ignore_value: float,
    workers: int,
) -> PixelStats:
    """Compute aggregate pixel statistics across the given window directories."""
    window_dirs = list(window_dirs)
    total = PixelStats()
    if workers <= 1:
        _init_worker(layer, ignore_value)
        for i, window_dir in enumerate(window_dirs):
            total.merge(_window_stats(window_dir))
            if (i + 1) % 5000 == 0:
                print(f"  ... processed {i + 1}/{len(window_dirs)} windows")
        return total

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(layer, ignore_value),
    ) as executor:
        futures = {
            executor.submit(_window_stats, window_dir): window_dir
            for window_dir in window_dirs
        }
        for i, future in enumerate(as_completed(futures)):
            total.merge(future.result())
            if (i + 1) % 5000 == 0:
                print(f"  ... processed {i + 1}/{len(window_dirs)} windows")
    return total


def main() -> None:
    """Compute and report LFMC target mean/std for a split."""
    parser = argparse.ArgumentParser(
        description="Compute mean/std of LFMC target values over a split."
    )
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--group",
        type=str,
        default="spatial_split",
        help="Window group to read (default: spatial_split).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Only include windows whose 'split' option equals this. "
        "Pass an empty string to disable the split filter.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="If set, only include windows that have this tag option (e.g. "
        "oep_eval_big).",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="labels",
        help="Raster layer holding the target (default: labels).",
    )
    parser.add_argument(
        "--ignore_value",
        type=float,
        default=-1.0,
        help="Pixel value to treat as invalid/nodata (default: -1).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker processes for reading rasters (default: 16).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the computed stats as JSON.",
    )
    args = parser.parse_args()

    split = args.split if args.split else None

    print(
        f"Selecting windows from group '{args.group}' "
        f"(split={split}, tag={args.tag})..."
    )
    window_dirs = _select_window_dirs(args.dataset_path, args.group, split, args.tag)
    print(f"  Selected {len(window_dirs)} windows")
    if not window_dirs:
        raise RuntimeError("no windows matched the given filters")

    print("Reading target rasters and accumulating statistics...")
    stats = compute_stats(window_dirs, args.layer, args.ignore_value, args.workers)

    if stats.count == 0:
        raise RuntimeError("no valid target pixels found")

    mean = stats.mean
    std_pop = stats.std(ddof=0)
    std_sample = stats.std(ddof=1)

    result = {
        "dataset_path": args.dataset_path,
        "group": args.group,
        "split": split,
        "tag": args.tag,
        "layer": args.layer,
        "ignore_value": args.ignore_value,
        "num_windows": len(window_dirs),
        "num_valid_pixels": stats.count,
        "mean": mean,
        "std": std_pop,
        "std_sample": std_sample,
        "min": stats.minimum,
        "max": stats.maximum,
    }

    print("\n=== LFMC target statistics (valid pixels only) ===")
    print(f"  windows:       {len(window_dirs)}")
    print(f"  valid pixels:  {stats.count}")
    print(f"  mean:          {mean:.6f}")
    print(f"  std (pop):     {std_pop:.6f}")
    print(f"  std (sample):  {std_sample:.6f}")
    print(f"  min / max:     {stats.minimum:.6f} / {stats.maximum:.6f}")
    print("\nUse these in the PerPixelRegressionTask config:")
    print(f"  target_mean: {mean:.6f}")
    print(f"  target_std: {std_pop:.6f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote stats to {args.output}")


if __name__ == "__main__":
    main()
