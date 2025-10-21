"""Expand single pixel label to 3x3 pixels."""

import argparse
import multiprocessing

import tqdm
from rslearn.dataset.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

BAND_NAME = "category"


def expand_single_pixel_label(window: Window) -> None:
    """Expand single pixel label to 3x3 pixels."""
    label_dir = window.get_raster_dir("label", [BAND_NAME])
    split = window.options["split"]
    np_array = GeotiffRasterFormat().decode_raster(
        label_dir, window.projection, window.bounds
    )
    print(np_array.shape)
    exit(0)
    center_x, center_y = np_array.shape[0] // 2, np_array.shape[1] // 2
    center_val = np_array[center_x, center_y]
    expanded_np_array = np_array.copy()
    # Only transform for the center pixel for train split, keep the original for val and test
    if split == "train":
        for x_offset in [-1, 0, 1]:
            for y_offset in [-1, 0, 1]:
                if x_offset == 0 and y_offset == 0:
                    continue
                expanded_np_array[center_x + x_offset, center_y + y_offset] = int(
                    center_val
                )
    # OK got the expanded array, now save it to a new window
    raster_dir = window.get_raster_dir("label_expanded", [BAND_NAME])
    GeotiffRasterFormat().encode_raster(
        raster_dir, window.projection, window.bounds, expanded_np_array
    )
    window.mark_layer_completed("label_expanded")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of worker processes to use",
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(workers=args.workers, show_progress=True)
    p = multiprocessing.Pool(args.workers)
    outputs = p.imap_unordered(expand_single_pixel_label, windows)
    for _ in tqdm.tqdm(outputs, total=len(windows)):
        pass
    p.close()
