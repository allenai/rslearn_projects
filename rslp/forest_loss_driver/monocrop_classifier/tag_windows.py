"""Tag monocrop dataset windows with derived options for config filtering.

Adds ``class_group`` to ``window.options``: ``"soy"`` for windows whose
``class_name`` is mennonites_soybean or soybean, and ``"other"`` for the rest.
Model configs can then select only soybean windows with
``tags: {class_group: soy}`` (rslearn tag filtering only supports exact
key/value matches, so set membership on class_name must be precomputed).

Note that boolean-looking tag values such as "true" or "yes" must be avoided:
jsonargparse YAML-parses Any-typed config values, so they would be coerced to
Python booleans and never match the string stored in window options.

The script is idempotent; re-running it leaves already-tagged windows unchanged.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter

import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath

CLASS_GROUP_TAG = "class_group"
SOY_CLASS_NAMES = frozenset({"mennonites_soybean", "soybean"})
DEFAULT_WORKERS = 32


def tag_window(window: Window) -> str:
    """Tag one window with its class group.

    Args:
        window: the window to tag.

    Returns:
        the outcome key for aggregate counting.
    """
    class_name = window.options.get("class_name")
    if class_name is None:
        return "skipped_no_class_name"
    class_group = "soy" if class_name in SOY_CLASS_NAMES else "other"
    if window.options.get(CLASS_GROUP_TAG) == class_group:
        return f"already_tagged_{class_group}"
    window.options[CLASS_GROUP_TAG] = class_group
    window.save()
    return f"tagged_{class_group}"


def tag_windows(ds_path: str, workers: int = DEFAULT_WORKERS) -> dict[str, int]:
    """Tag all windows in the dataset with their class group.

    Args:
        ds_path: path to the monocrop rslearn dataset.
        workers: number of processes for loading and tagging windows.

    Returns:
        counts of tagged and already-tagged windows per class group.
    """
    dataset = Dataset(UPath(ds_path))
    windows = dataset.load_windows(workers=workers, show_progress=True)
    counts: Counter[str] = Counter()
    with multiprocessing.Pool(workers) as pool:
        outcomes = pool.imap_unordered(tag_window, windows)
        for outcome in tqdm.tqdm(outcomes, total=len(windows), desc="Tagging"):
            counts[outcome] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    """Tag windows in the dataset given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds-path", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    print(json.dumps(tag_windows(args.ds_path, workers=args.workers), indent=2))


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
