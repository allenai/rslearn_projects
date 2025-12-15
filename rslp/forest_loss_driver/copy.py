"""Copy important portions of rslearn forest loss driver dataset.

This is used for Ai2 workflow because we initially create the dataset on Weka since it
makes both materialization and prediction much faster, but then we need to serve the
images from GCS.

It may not be needed in other workflows.
"""

import multiprocessing
from typing import Any

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from rslp.utils.fs import copy_file

from .const import WINDOWS_FNAME

NEEDED_LAYERS = [
    "best_pre_0",
    "best_pre_1",
    "best_pre_2",
    "best_post_0",
    "best_post_1",
    "best_post_2",
    "planet_pre_0",
    "planet_pre_1",
    "planet_pre_2",
    "planet_post_0",
    "planet_post_1",
    "planet_post_2",
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


def get_copy_jobs_for_window(
    window: Window, src_path: UPath, dst_path: UPath
) -> list[dict[str, Any]]:
    """Get the job dictionaries for copying all the needed files for this window.

    Args:
        window: the window to copy from.
        src_path: the source base directory containing the window.
        dst_path: the destination base directory to copy to.

    Returns:
        list of dictionaries with src_fname and dst_fname keys suitable to pass to
            copy_file function via star_imap.
    """
    copy_file_jobs = []

    # Copy files in the needed layers.
    for layer_name in NEEDED_LAYERS:
        layer_dir = window.get_layer_dir(layer_name)
        for src_fname in layer_dir.rglob("*"):
            # Only copy certain files.
            # This also ensures directories are skipped.
            if src_fname.name != "completed" and src_fname.name.split(".")[-1] not in [
                "png",
                "geojson",
            ]:
                continue

            dst_fname = dst_path / get_relative_suffix(src_path, src_fname)
            copy_file_jobs.append(
                dict(
                    src_fname=src_fname,
                    dst_fname=dst_fname,
                )
            )

    # Copy additional JSON files.
    window_path = window.storage.get_window_root(window.group, window.name)
    for window_suffix in ["info.json", "metadata.json", "least_cloudy_times.json"]:
        src_fname = window_path / window_suffix
        if not src_fname.exists():
            # Could happen if the window did not end up being used due to missing some
            # layers.
            continue
        dst_fname = dst_path / get_relative_suffix(src_path, src_fname)
        copy_file_jobs.append(
            dict(
                src_fname=src_fname,
                dst_fname=dst_fname,
            )
        )

    return copy_file_jobs


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

    # Copy index of windows with outputs.
    copy_file(src_path / WINDOWS_FNAME, dst_path / WINDOWS_FNAME)

    # Copy per-window files. We first list the windows, and then in parallel we check
    # each window to enumerate the files that need to be copied.
    dataset = Dataset(src_path)
    windows = dataset.load_windows(workers=workers)
    p = multiprocessing.Pool(workers)
    list_jobs = [
        dict(
            window=window,
            src_path=src_path,
            dst_path=dst_path,
        )
        for window in windows
    ]
    outputs = star_imap_unordered(p, get_copy_jobs_for_window, list_jobs)
    copy_file_jobs = []
    for window_copy_file_jobs in tqdm.tqdm(
        outputs, total=len(list_jobs), desc="Listing source files"
    ):
        copy_file_jobs.extend(window_copy_file_jobs)

    # Now perform the actual copies, also in parallel.
    outputs = star_imap_unordered(p, copy_file, copy_file_jobs)
    for _ in tqdm.tqdm(outputs, total=len(copy_file_jobs), desc="Copying files"):
        pass
    p.close()
