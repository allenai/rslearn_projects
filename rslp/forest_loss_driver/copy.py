"""Copy important portions of rslearn forest loss driver dataset.

This is used for Ai2 workflow because we initially create the dataset on Weka since it
makes both materialization and prediction much faster, but then we need to serve the
images from GCS.

It may not be needed in other workflows.
"""

import multiprocessing

import tqdm
from rslearn.dataset import Dataset
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from rslp.utils.fs import copy_file

NEEDED_LAYERS = [
    "best_pre_0",
    "best_pre_1",
    "best_pre_2",
    "best_post_0",
    "best_post_1",
    "best_post_2",
    "mask",
    "output",
]


def get_relative_suffix(base_dir: UPath, fname: UPath) -> str:
    """Get the suffix of fname relative to base_dir.

    Args:
        base_dir: the base directory.
        fname: a filename within the base directory.

    Returns:
        the suffix on base_dir that would yield the given filename.
    """
    if not fname.path.startswith(base_dir.path):
        raise ValueError(
            f"filename {fname.path} must start with base directory {base_dir.path}"
        )
    suffix = fname.path[len(base_dir.path) :]
    if suffix.startswith("/"):
        suffix = suffix[1:]
    return suffix


def copy_dataset(src_path: UPath, dst_path: UPath, workers: int = 32) -> None:
    """Copy the rslearn forest loss driver dataset.

    Only images needed for inference and visualization are needed. These include the
    best_pre/post_X, mask, and output layers, as well as the dataset config, and
    metadata.json for each window (along with forest-loss-driver-specific info.json).

    Args:
        src_path: the source dataset to copy from.
        dst_path: the path to copy the dataset to.
        workers: number of worker processes to use.
    """
    # Copy dataset configuration.
    copy_file(src_path / "config.json", dst_path / "config.json")

    # Copy per-window files. We create argument list for copy_file and then run it in
    # parallel.
    dataset = Dataset(src_path)
    windows = dataset.load_windows(workers=workers)
    copy_file_jobs = []
    for window in windows:
        # Copy files in the needed layers.
        for layer_name in NEEDED_LAYERS:
            layer_dir = window.get_layer_dir(layer_name)
            for _, _, fnames in layer_dir.walk():
                for src_fname in fnames:
                    dst_fname = dst_path / get_relative_suffix(src_path, src_fname)
                    copy_file_jobs.append(
                        dict(
                            src_fname=src_fname,
                            dst_fname=dst_fname,
                        )
                    )

        # Copy info.json and metadata.json.
        for window_suffix in ["info.json", "metadata.json"]:
            src_fname = window.path / window_suffix
            dst_fname = dst_path / get_relative_suffix(src_path, src_fname)
            copy_file_jobs.append(
                dict(
                    src_fname=src_fname,
                    dst_fname=dst_fname,
                )
            )

    p = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(p, copy_file, copy_file_jobs)
    for _ in tqdm.tqdm(outputs, total=len(copy_file_jobs)):
        pass
    p.close()
