"""Add helios_split to rslearn windows."""
import argparse
import hashlib
import multiprocessing

import tqdm
from rslearn.dataset import Dataset, Window
from upath import UPath


def add_helios_split(window: Window) -> None:
    """Add helios_split to a window."""
    digest = hashlib.sha256(window.name.split("_")[0].encode()).hexdigest()[0]

    if digest in ["0", "1"]:
        split = "val"
    elif digest in ["2", "3"]:
        split = "test"
    else:
        split = "train"

    window.options["helios_split"] = split
    window.save()


def parse_args():
    parser = argparse.ArgumentParser(description="Assign helios_split to rslearn dataset windows.")
    parser.add_argument("--ds_path", type=UPath, required=True, help="Path to the dataset")
    parser.add_argument("--ds_group", action="append", help="Dataset group(s) to load")
    parser.add_argument("-j", "--jobs", type=int, default=128, help="Number of parallel workers (default: 128)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    multiprocessing.set_start_method("forkserver")

    windows = Dataset(args.ds_path).load_windows(groups=args.ds_group, workers=args.jobs, show_progress=True)
    with multiprocessing.Pool(args.jobs) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(add_helios_split, windows), total=len(windows)):
            pass