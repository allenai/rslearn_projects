"""Concatenate the per-window summary JSONs written by ``from_embeddings.py``.

``from_embeddings.py`` writes a small JSON file (``change_summary.json`` by
default) inside each window directory.  This script walks every window in a
dataset in parallel, reads its summary file if present, and writes a single
JSON list of all per-window dicts.
"""

import argparse
import json
import multiprocessing
import multiprocessing.pool
from collections.abc import Iterable

import tqdm
from rslearn.dataset import Dataset
from rslearn.dataset.window import Window
from upath import UPath


def _load_summary(window: Window, summary_filename: str) -> dict | None:
    """Return the parsed summary dict for a window, or None if not present."""
    window_root = window.storage.get_window_root(window.group, window.name)
    summary_path = window_root / summary_filename
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(
            f"[from_embeddings_concat] skipping {window.group}/{window.name} "
            f"due to read error: {e}"
        )
        return None


def _load_summary_star(kwargs: dict) -> dict | None:
    return _load_summary(**kwargs)


def concat_summaries(
    ds_path: str,
    summary_filename: str = "change_summary.json",
    out_path: str = "change_summary_all.json",
    min_max_score: float | None = None,
    workers: int = 32,
) -> None:
    """Walk all windows and concatenate per-window summaries into one JSON list.

    If ``min_max_score`` is not None, only windows whose summary has
    ``max_score > min_max_score`` are kept.
    """
    dataset = Dataset(UPath(ds_path))
    windows = dataset.load_windows(workers=128, show_progress=True)

    kwargs_list = [
        dict(window=window, summary_filename=summary_filename) for window in windows
    ]

    results: Iterable[dict | None]
    pool: multiprocessing.pool.Pool | None = None
    if workers <= 0:
        results = map(_load_summary_star, kwargs_list)
    else:
        pool = multiprocessing.Pool(workers)
        results = pool.imap_unordered(_load_summary_star, kwargs_list)

    summaries: list[dict] = []
    try:
        for summary in tqdm.tqdm(
            results, total=len(kwargs_list), desc="Reading summaries"
        ):
            if summary is None:
                continue
            if (
                min_max_score is not None
                and summary.get("max_score", float("-inf")) <= min_max_score
            ):
                continue
            summaries.append(summary)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    with open(out_path, "w") as f:
        json.dump(summaries, f)

    print(f"Wrote {len(summaries)} window summaries to {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser(
        description="Concatenate per-window summaries from from_embeddings.py"
    )
    parser.add_argument("--ds_path", required=True, help="Path to rslearn dataset")
    parser.add_argument(
        "--summary_filename",
        default="change_summary.json",
        help="Per-window summary JSON filename written by from_embeddings.py",
    )
    parser.add_argument(
        "--out_path",
        default="change_summary_all.json",
        help="Output JSON path (list of per-window summary dicts)",
    )
    parser.add_argument(
        "--min_max_score",
        type=float,
        default=None,
        help="If set, only include windows with max_score > this threshold",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    concat_summaries(
        ds_path=args.ds_path,
        summary_filename=args.summary_filename,
        out_path=args.out_path,
        min_max_score=args.min_max_score,
        workers=args.workers,
    )
