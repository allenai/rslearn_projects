"""Add helios_split to rslearn dataset windows."""

import argparse
import hashlib
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath


def add_helios_split(window: Window) -> None:
    """Add helios_split to a window."""
    digest = hashlib.sha256(window.name.split("_")[0].encode()).hexdigest()[0]
    if digest in ("0", "1"):
        split = "val"
    elif digest in ("2", "3"):
        split = "test"
    else:
        split = "train"
    window.options["helios_split"] = split
    window.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assign helios_split to rslearn dataset windows."
    )
    parser.add_argument("--ds_path", type=UPath, required=True)
    parser.add_argument("--ds_group", action="append")
    parser.add_argument("-j", "--jobs", type=int, default=128)
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn")

    windows = Dataset(args.ds_path).load_windows(
        groups=args.ds_group, workers=args.jobs, show_progress=True
    )

    with multiprocessing.Pool(args.jobs) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(add_helios_split, windows), total=len(windows)
        ):
            pass
