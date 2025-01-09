import argparse
import hashlib
import json
import multiprocessing

import tqdm
from upath import UPath


def check_window(window_root):
    output_fname = window_root / "layers" / "output" / "data.geojson"
    return window_root.name, output_fname.exists()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", help="Dataset root path", type=str, required=True)
    parser.add_argument(
        "--group",
        help="Name of the group to check",
        type=str,
        default="default",
    )
    parser.add_argument(
        "--workers",
        help="Number of worker processes to use",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--out_fname",
        help="Filename to write the good windows (that have output)",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    window_dir = ds_path / "windows" / args.group
    window_roots = list(tqdm.tqdm(window_dir.iterdir(), desc="Loading window names"))
    p = multiprocessing.Pool(64)
    outputs = p.imap_unordered(check_window, window_roots)
    window_names = []
    for window_name, is_good in tqdm.tqdm(outputs, total=len(window_roots)):
        if not is_good:
            continue
        window_names.append(window_name)
    p.close()

    window_names.sort(
        key=lambda window_name: hashlib.sha256(window_name.encode()).hexdigest()
    )

    with open(args.out_fname, "w") as f:
        json.dump(window_names, f)
