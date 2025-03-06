"""Create the index of good windows that should be served in web app."""

import hashlib
import json
import multiprocessing

import tqdm
from upath import UPath

from rslp.forest_loss_driver.const import GROUP, WINDOWS_FNAME

DEFAULT_WORKERS = 32
OUTPUT_GEOJSON_SUFFIX = "layers/output/data.geojson"


def _check_window(window_root: UPath) -> tuple[str, bool]:
    """Helper function for multiprocessing to check whether this window has an output.

    Args:
        window_root: the window path.

    Returns:
        a tuple (window_name, is_good) where is_good indicates whether the window has a
            prediction computed.
    """
    output_fname = window_root / OUTPUT_GEOJSON_SUFFIX
    return window_root.name, output_fname.exists()


def index_windows(ds_root: str, workers: int = DEFAULT_WORKERS) -> None:
    """Create the index of good windows that should be served in the web app.

    Good windows are ones where a prediction is available.

    Args:
        ds_root: the inference dataset path.
        workers: the number of workers to use during indexing.
    """
    # List the windows.
    ds_path = UPath(ds_root)
    group_dir = ds_path / "windows" / GROUP
    window_roots = list(group_dir.iterdir())

    # Check which windows have outputs computed.
    p = multiprocessing.Pool(workers)
    outputs = p.imap_unordered(_check_window, window_roots)
    good_window_names = []
    for window_name, is_good in tqdm.tqdm(outputs, total=len(window_roots)):
        if not is_good:
            continue
        good_window_names.append(window_name)

    # Shuffle the windows deterministically using hash.
    # This way there is no issue running index_windows multiple times (since it is
    # deterministic), but also if these are being validated then the user can get a
    # random sample by just going from the beginning.
    good_window_names.sort(
        key=lambda window_name: hashlib.sha256(window_name.encode()).hexdigest()
    )

    with (ds_path / WINDOWS_FNAME).open("w") as f:
        json.dump(good_window_names, f)
