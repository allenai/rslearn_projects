"""Prepare the randomized slot layers (freq_rand_{k} / infreq_rand_{k}).

This is the LCC-style augmentation where each window gets its own random time series
end offsets, instead of the fixed 7/35/62/90 day offsets. For each slot k, each window
draws an end offset uniformly from [7, 90] days after the change (seeded by the window
name and k). The rand layers in config.json end at the window time (the change date),
so we prepare them on in-memory copies of the windows whose time range is shifted by
that offset.

This works because prepare computes the request time range from the window's time
range and saves the matched item groups along with the request time range of each
group (each mosaic period). Materialize then uses those saved group time ranges rather
than the window time range, so after this script, the rand layers can be materialized
normally with `rslearn dataset materialize`.

The offsets are also saved in the window options as rand_end_days for reference.
"""

import argparse
import copy
import hashlib
import multiprocessing
import random
from datetime import timedelta

import tqdm
from common import DETECTION_RANGE, FREQUENT_PERIOD, NUM_SLOTS
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import prepare_dataset_windows
from upath import UPath


def rand_end_days(window_name: str, slot: int) -> int:
    """Get the random end offset in days for one window and slot."""
    seed = hashlib.sha256(f"{window_name}/{slot}".encode()).hexdigest()
    return random.Random(seed).randint(FREQUENT_PERIOD.days, DETECTION_RANGE.days)


def shifted(window: Window, days: int) -> Window:
    """Get a copy of the window (not to be saved) with its time range shifted."""
    assert window.time_range is not None
    offset = timedelta(days=days)
    copied = copy.copy(window)
    copied.time_range = (window.time_range[0] + offset, window.time_range[1] + offset)
    return copied


def prepare_batch(
    ds_path: str, group: str | None, names: list[str], retry_max_attempts: int
) -> None:
    """Prepare the rand layers for a batch of windows."""
    dataset = Dataset(UPath(ds_path))
    windows = dataset.load_windows(groups=[group] if group else None, names=names)
    for slot in range(NUM_SLOTS):
        slot_dataset = Dataset(
            UPath(ds_path),
            enabled_layers=[f"freq_rand_{slot}", f"infreq_rand_{slot}"],
        )
        prepare_dataset_windows(
            slot_dataset,
            [shifted(window, rand_end_days(window.name, slot)) for window in windows],
            retry_max_attempts=retry_max_attempts,
        )
    for window in windows:
        offsets = [rand_end_days(window.name, slot) for slot in range(NUM_SLOTS)]
        if window.options.get("rand_end_days") != offsets:
            window.options["rand_end_days"] = offsets
            window.save()


def main() -> None:
    """Prepare the rand layers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--group", default=None)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--retry_max_attempts", type=int, default=3)
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(groups=[args.group] if args.group else None)
    names = sorted(window.name for window in windows)
    jobs = [
        (
            args.ds_path,
            args.group,
            names[i : i + args.batch_size],
            args.retry_max_attempts,
        )
        for i in range(0, len(names), args.batch_size)
    ]
    with multiprocessing.Pool(args.workers) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(_prepare_batch_star, jobs), total=len(jobs)
        ):
            pass


def _prepare_batch_star(job: tuple[str, str | None, list[str], int]) -> None:
    prepare_batch(*job)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
