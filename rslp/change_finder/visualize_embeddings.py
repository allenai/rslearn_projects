"""Visualize per-year OlmoEarth embeddings alongside raw Sentinel-2 images.

For each window, produces a PNG with:
- Image grid: 10 rows (y0..y9) x 6 columns (per-year timesteps), RGB composites
- Chart below: cosine-similarity-based change score (0 = like y0, 1 = like y9)
"""

import argparse
import io
import os
from pathlib import Path

import h5py
import matplotlib
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from rslearn.dataset import Dataset
from rslearn.train.dataset import DataInput, ModelDataset, SplitConfig
from rslearn.train.model_context import RasterImage
from upath import UPath

from .train import ChangeFinderTask

OLMOEARTH_BAND_ORDER = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]
# In OLMOEARTH_BAND_ORDER: B04 is index 2, B03 is index 1, B02 is index 0
RGB_BAND_INDICES = [2, 1, 0]
NUM_YEARS = 10
NUM_TIMESTEPS = 6


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        return 0.0
    return float(dot / norm)


def _compute_change_scores(embeddings: np.ndarray) -> np.ndarray | None:
    """Compute change score for each year: 0 = like y0, 1 = like y9.

    Args:
        embeddings: (NUM_YEARS, embed_dim) array, may contain NaN rows for missing years.

    Returns:
        (NUM_YEARS,) array of scores, or None if y0 or y9 is missing.
    """
    emb_first = embeddings[0]
    emb_last = embeddings[-1]
    if np.any(np.isnan(emb_first)) or np.any(np.isnan(emb_last)):
        return None

    scores = np.full(NUM_YEARS, np.nan)
    for i in range(NUM_YEARS):
        if np.any(np.isnan(embeddings[i])):
            continue
        dist_first = 1.0 - _cosine_similarity(embeddings[i], emb_first)
        dist_last = 1.0 - _cosine_similarity(embeddings[i], emb_last)
        denom = dist_first + dist_last
        if denom < 1e-12:
            scores[i] = 0.5
        else:
            scores[i] = dist_first / denom
    return scores


def _raster_to_rgb(raster: RasterImage) -> np.ndarray:
    """Convert a RasterImage (C, T, H, W) to RGB array (T, H, W, 3) in uint8.

    Applies a simple percentile stretch for visualization.
    """
    img = raster.image.numpy()  # (C, T, H, W)
    rgb = img[RGB_BAND_INDICES]  # (3, T, H, W)
    rgb = rgb.transpose(1, 2, 3, 0).astype(np.float32)  # (T, H, W, 3)

    lo = np.nanpercentile(rgb, 2)
    hi = np.nanpercentile(rgb, 98)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    rgb = (rgb - lo) / (hi - lo)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def visualize_embeddings(
    ds_path: str,
    embeddings_filename: str = "embeddings.h5",
    mask_filename: str = "embeddings.png",
    out_dir: str = "change_finder_vis",
    num_windows: int | None = None,
    split: str | None = "val",
    workers: int = 32,
) -> None:
    """Produce per-window visualization PNGs.

    Args:
        ds_path: path to the rslearn dataset.
        embeddings_filename: H5 filename inside each window directory (from compute_embeddings).
        mask_filename: PNG filename for per-patch change mask (from compute_embeddings).
        out_dir: local directory to write output PNGs.
        num_windows: maximum number of windows to visualize (None = all).
        split: filter windows by split tag. None = all.
        workers: number of workers for dataset initialization.
    """
    dataset = Dataset(UPath(ds_path))

    inputs_config = {}
    for i in range(NUM_YEARS):
        inputs_config[f"sentinel2_y{i}"] = DataInput(
            data_type="raster",
            layers=[f"sentinel2_y{i}"],
            bands=OLMOEARTH_BAND_ORDER,
            passthrough=True,
            load_all_item_groups=True,
            load_all_layers=True,
        )

    tags = {"split": split} if split else None
    split_config = SplitConfig(tags=tags)
    task = ChangeFinderTask()

    model_dataset = ModelDataset(
        dataset=dataset,
        inputs=inputs_config,
        task=task,
        split_config=split_config,
        workers=workers,
    )

    os.makedirs(out_dir, exist_ok=True)
    count = 0

    for idx in tqdm.tqdm(range(len(model_dataset)), desc="Visualizing"):
        if num_windows is not None and count >= num_windows:
            break

        input_dict, _target_dict, metadata = model_dataset[idx]

        window_root = dataset.storage.get_window_root(
            metadata.window_group, metadata.window_name
        )
        emb_path = window_root / embeddings_filename
        try:
            with emb_path.open("rb") as fp:
                buf = io.BytesIO(fp.read())
            with h5py.File(buf, "r") as f:
                embeddings = f["embeddings"][:]
        except (FileNotFoundError, OSError):
            continue

        scores = _compute_change_scores(embeddings)
        if scores is None:
            continue

        year_rgbs: dict[int, np.ndarray] = {}
        for i in range(NUM_YEARS):
            key = f"sentinel2_y{i}"
            if key in input_dict:
                year_rgbs[i] = _raster_to_rgb(input_dict[key])

        if 0 not in year_rgbs or (NUM_YEARS - 1) not in year_rgbs:
            continue

        h, w = next(iter(year_rgbs.values())).shape[1:3]

        mask_path = window_root / mask_filename
        try:
            with mask_path.open("rb") as fp:
                mask_img = np.array(Image.open(io.BytesIO(fp.read())).convert("L"))
        except (FileNotFoundError, OSError):
            continue

        chart_ratio = 1.5
        n_rows = NUM_YEARS + 2
        height_ratios = [1] * NUM_YEARS + [1, chart_ratio]

        fig = plt.figure(figsize=(NUM_TIMESTEPS * 2, n_rows * 2))
        gs = fig.add_gridspec(
            n_rows,
            NUM_TIMESTEPS,
            height_ratios=height_ratios,
            hspace=0.15,
            wspace=0.05,
        )

        for yi in range(NUM_YEARS):
            for ti in range(NUM_TIMESTEPS):
                ax = fig.add_subplot(gs[yi, ti])
                if yi in year_rgbs and ti < year_rgbs[yi].shape[0]:
                    ax.imshow(year_rgbs[yi][ti])
                else:
                    ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))
                ax.set_xticks([])
                ax.set_yticks([])
                if ti == 0:
                    ax.set_ylabel(f"y{yi}", fontsize=9)

        for ti in range(NUM_TIMESTEPS):
            ax_mask = fig.add_subplot(gs[NUM_YEARS, ti])
            ax_mask.imshow(mask_img, cmap="hot", vmin=0, vmax=255)
            ax_mask.set_xticks([])
            ax_mask.set_yticks([])
            if ti == 0:
                ax_mask.set_ylabel("mask", fontsize=9)

        ax_chart = fig.add_subplot(gs[NUM_YEARS + 1, :])
        valid_mask = ~np.isnan(scores)
        years = np.arange(NUM_YEARS)
        ax_chart.plot(
            years[valid_mask],
            scores[valid_mask],
            "o-",
            color="tab:blue",
            linewidth=2,
            markersize=6,
        )
        ax_chart.set_xlim(-0.3, NUM_YEARS - 0.7)
        ax_chart.set_ylim(-0.05, 1.05)
        ax_chart.set_xticks(years)
        ax_chart.set_xticklabels([f"y{i}" for i in range(NUM_YEARS)])
        ax_chart.set_ylabel("Change score")
        ax_chart.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_chart.axhline(1, color="gray", linewidth=0.5, linestyle="--")

        safe_name = metadata.window_name.replace("/", "_")
        out_path = Path(out_dir) / f"{metadata.window_group}_{safe_name}.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        count += 1

    print(f"Saved {count} visualizations to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize change finder embeddings")
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--embeddings_filename", default="embeddings.h5", help="H5 filename per window"
    )
    parser.add_argument(
        "--mask_filename",
        default="embeddings.png",
        help="PNG filename for per-patch change mask",
    )
    parser.add_argument(
        "--out_dir", default="change_finder_vis", help="Output directory for PNGs"
    )
    parser.add_argument(
        "--num_windows", type=int, default=None, help="Max windows to visualize"
    )
    parser.add_argument("--split", default="val", help="Split tag filter (empty=all)")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    visualize_embeddings(
        ds_path=args.ds_path,
        embeddings_filename=args.embeddings_filename,
        mask_filename=args.mask_filename,
        out_dir=args.out_dir,
        num_windows=args.num_windows,
        split=args.split or None,
        workers=args.workers,
    )
