"""Create a JSON list of windows containing embedding-based change.

Scans every window for the per-patch change mask written by
``compute_embeddings.py`` (the ``mask_filename`` output, by default
``embeddings.tif``).  A window is kept when at least one patch has change
score strictly greater than ``threshold``.  The output is a plain JSON list
of ``{"window_group": ..., "window_name": ...}`` entries, suitable for
feeding to :mod:`rslp.change_finder.embedding_change_viewer.server`.
"""

import argparse
import json
import multiprocessing
import multiprocessing.pool
from collections.abc import Iterable

import tqdm
from rslearn.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.fsspec import open_rasterio_upath_reader
from upath import UPath


def _process_window(
    window: Window,
    threshold: float,
    mask_filename: str,
) -> dict | None:
    """Return an entry for this window if its mask exceeds ``threshold``."""
    window_root = window.storage.get_window_root(window.group, window.name)
    mask_path = window_root / mask_filename
    if not mask_path.exists():
        return None

    try:
        with open_rasterio_upath_reader(mask_path) as src:
            arr = src.read(1)
    except Exception as e:
        print(
            f"[create_embedding_change_list] skipping "
            f"{window.group}/{window.name}: failed to read {mask_filename}: {e}"
        )
        return None

    if arr.size == 0 or arr.max() <= threshold:
        return None

    return {"window_group": window.group, "window_name": window.name}


def _process_window_star(kwargs: dict) -> dict | None:
    return _process_window(**kwargs)


def create_list(
    ds_path: str,
    out_path: str = "embedding_change_list.json",
    mask_filename: str = "embeddings.tif",
    threshold: float = 150,
    workers: int = 32,
) -> None:
    """Scan all windows and write a JSON list of those with change above threshold."""
    dataset = Dataset(UPath(ds_path))
    windows = dataset.load_windows(workers=128, show_progress=True)

    kwargs_list = [
        dict(
            window=window,
            threshold=threshold,
            mask_filename=mask_filename,
        )
        for window in windows
    ]

    entries: list[dict] = []

    pool: multiprocessing.pool.Pool | None = None
    results: Iterable[dict | None]
    if workers <= 0:
        results = map(_process_window_star, kwargs_list)
    else:
        pool = multiprocessing.Pool(workers)
        results = pool.imap_unordered(_process_window_star, kwargs_list)

    try:
        for entry in tqdm.tqdm(
            results, total=len(kwargs_list), desc="Scanning windows"
        ):
            if entry is not None:
                entries.append(entry)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    entries.sort(key=lambda e: (e["window_group"], e["window_name"]))

    with open(out_path, "w") as f:
        json.dump(entries, f)

    print(f"Wrote {len(entries)} windows to {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Create JSON list of windows with embedding-based change"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--out_path",
        default="embedding_change_list.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--mask_filename",
        default="embeddings.tif",
        help="Per-patch change mask GeoTIFF filename per window",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=150,
        help="Keep windows with at least one patch value strictly greater than this",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    create_list(
        ds_path=args.ds_path,
        out_path=args.out_path,
        mask_filename=args.mask_filename,
        threshold=args.threshold,
        workers=args.workers,
    )
