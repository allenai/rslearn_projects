"""Find examples where the land cover transitions to or from a given category."""

import argparse
import multiprocessing
import random

import numpy as np
import tqdm
from rslearn.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.colors import DEFAULT_COLORS
from rslearn.utils.mp import make_pool_and_star_imap_unordered
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

RASTER_FORMAT = GeotiffRasterFormat()

RGB_BANDS = ["B04", "B03", "B02"]
NUM_CLASSES = 13
NUM_TIMESTEPS = 6
COLOR_TABLE = np.array(DEFAULT_COLORS[:NUM_CLASSES], dtype=np.uint8)


def _process_window(
    dataset: Dataset,
    window: Window,
    class_id: int,
    is_source: bool,
    high_threshold: float,
    low_threshold: float,
    num_early: int,
    num_late: int,
    min_pixels: int,
    out_dir: str,
    change_filename: str,
) -> bool:
    window_root = window.storage.get_window_root(window.group, window.name)
    if not (window_root / change_filename).exists():
        return False

    change_raster: RasterArray = RASTER_FORMAT.decode_raster(
        window_root, window.projection, window.bounds, fname=change_filename
    )
    change_data = change_raster.get_chw_array()

    h, w = change_data.shape[1], change_data.shape[2]

    header_bands = 3
    early_probs = []
    for i in range(num_early):
        band_idx = header_bands + i * NUM_CLASSES + class_id
        early_probs.append(change_data[band_idx].astype(np.float32) / 255.0)

    late_probs = []
    for i in range(num_late):
        band_idx = header_bands + num_early * NUM_CLASSES + i * NUM_CLASSES + class_id
        late_probs.append(change_data[band_idx].astype(np.float32) / 255.0)

    if is_source:
        mask = np.ones((h, w), dtype=bool)
        for p in early_probs:
            mask &= p > high_threshold
        for p in late_probs:
            mask &= p < low_threshold
    else:
        mask = np.ones((h, w), dtype=bool)
        for p in early_probs:
            mask &= p < low_threshold
        for p in late_probs:
            mask &= p > high_threshold

    if mask.sum() < min_pixels:
        return False

    print(f"found a match in window {window.group}/{window.name}")
    out_root = UPath(out_dir)
    mask_arr = (mask * 255).astype(np.uint8)[np.newaxis]  # (1, H, W)
    RASTER_FORMAT.encode_raster(
        out_root,
        window.projection,
        window.bounds,
        RasterArray(chw_array=mask_arr),
        fname=f"{window.name}_mask.tif",
    )

    for year_idx in [0, 9]:
        layer_name = f"sentinel2_y{year_idx}"
        for group_idx in range(NUM_TIMESTEPS):
            if not window.is_layer_completed(layer_name, group_idx):
                continue
            band_names = dataset.layers[layer_name].band_sets[0].bands
            rgb_band_indices = [band_names.index(b) for b in RGB_BANDS]
            raster_dir = window.get_raster_dir(layer_name, band_names, group_idx)
            s2_raster = RASTER_FORMAT.decode_raster(
                raster_dir, window.projection, window.bounds
            )
            data = s2_raster.get_chw_array()
            rgb = np.clip(data[rgb_band_indices] // 10, 0, 255).astype(np.uint8)
            RASTER_FORMAT.encode_raster(
                out_root,
                window.projection,
                window.bounds,
                RasterArray(chw_array=rgb),
                fname=f"{window.name}_y{year_idx}_{group_idx}.tif",
            )

    early_class = change_data[1]  # (H, W) early dominant class
    late_class = change_data[2]  # (H, W) late dominant class
    for year_idx, class_map in [(0, early_class), (9, late_class)]:
        colored = COLOR_TABLE[class_map].transpose(2, 0, 1)  # (3, H, W)
        RASTER_FORMAT.encode_raster(
            out_root,
            window.projection,
            window.bounds,
            RasterArray(chw_array=colored),
            fname=f"{window.name}_y{year_idx}_pred.tif",
        )

    return True


def find_examples(
    ds_path: str,
    class_id: int,
    is_source: bool = True,
    high_threshold: float = 0.75,
    low_threshold: float = 0.25,
    num_early: int = 3,
    num_late: int = 3,
    min_pixels: int = 10,
    out_dir: str = "land_cover_change_examples",
    change_filename: str = "land_cover_change.tif",
    num_windows: int | None = None,
    workers: int = 32,
) -> None:
    """Find windows where a given class transitions and save visualizations.

    Args:
        ds_path: path to the rslearn dataset.
        class_id: the land cover class ID to look for.
        is_source: if True, look for class_id in early years (source of transition).
            If False, look for class_id in late years (destination of transition).
        high_threshold: probability must exceed this in the "present" period.
        low_threshold: probability must be below this in the "absent" period.
        num_early: number of early timesteps stored in the change tif.
        num_late: number of late timesteps stored in the change tif.
        min_pixels: minimum number of pixels satisfying the constraint per window.
        out_dir: directory to write output GeoTIFFs.
        change_filename: filename of the land cover change GeoTIFF per window.
        num_windows: maximum number of matching windows to output (None = all).
        workers: number of multiprocessing workers.
    """
    dataset = Dataset(UPath(ds_path))
    windows = dataset.load_windows(workers=128, show_progress=True)
    random.shuffle(windows)

    kwargs_list = [
        dict(
            dataset=dataset,
            window=window,
            class_id=class_id,
            is_source=is_source,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            num_early=num_early,
            num_late=num_late,
            min_pixels=min_pixels,
            out_dir=out_dir,
            change_filename=change_filename,
        )
        for window in windows
    ]

    found = 0
    with make_pool_and_star_imap_unordered(
        workers, _process_window, kwargs_list
    ) as results:
        for result in tqdm.tqdm(
            results, total=len(kwargs_list), desc="Scanning windows"
        ):
            if not result:
                continue
            found += 1
            if num_windows is not None and found >= num_windows:
                break

    print(f"Saved {found} matching windows to {out_dir}/")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Find examples of land cover change for a given class"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--class_id", type=int, required=True, help="Land cover class ID"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source",
        action="store_true",
        help="Look for class_id as source (present in early, absent in late)",
    )
    group.add_argument(
        "--destination",
        action="store_true",
        help="Look for class_id as destination (absent in early, present in late)",
    )
    parser.add_argument(
        "--high_threshold",
        type=float,
        default=0.75,
        help="Prob must exceed this in the 'present' period",
    )
    parser.add_argument(
        "--low_threshold",
        type=float,
        default=0.25,
        help="Prob must be below this in the 'absent' period",
    )
    parser.add_argument(
        "--num_early",
        type=int,
        default=3,
        help="Number of early timesteps in the change tif",
    )
    parser.add_argument(
        "--num_late",
        type=int,
        default=3,
        help="Number of late timesteps in the change tif",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=10,
        help="Minimum pixels satisfying constraint",
    )
    parser.add_argument(
        "--out_dir",
        default="land_cover_change_examples",
        help="Output directory",
    )
    parser.add_argument(
        "--change_filename",
        default="land_cover_change.tif",
        help="Change GeoTIFF filename per window",
    )
    parser.add_argument(
        "--num_windows",
        type=int,
        default=None,
        help="Max matching windows to output",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    find_examples(
        ds_path=args.ds_path,
        class_id=args.class_id,
        is_source=args.source,
        high_threshold=args.high_threshold,
        low_threshold=args.low_threshold,
        num_early=args.num_early,
        num_late=args.num_late,
        min_pixels=args.min_pixels,
        out_dir=args.out_dir,
        change_filename=args.change_filename,
        num_windows=args.num_windows,
        workers=args.workers,
    )
