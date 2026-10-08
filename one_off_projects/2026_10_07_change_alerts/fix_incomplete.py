"""Find image layers that failed to materialize and re-prepare them.

The olmoearth_datasets Sentinel-2 backfill renames scenes (e.g. appending
"_S<time>" to the name), so items prepared before a rename fail to materialize with
"Expected 1 item for X, got 0 from OlmoEarth API", and the whole layer is left
incomplete for that window. This script scans the dataset for image layers that are
prepared but not completed, deletes their partial outputs, and prepares those layers
again with force=True so that they reference the current item names. The randomized
layers are prepared on shifted copies of the windows as in prepare_randomized.py.

Afterwards, run `rslearn dataset materialize` again (completed layers are skipped).
Repeat until the scan reports no incomplete layers or no progress is made.

Usage:
    python fix_incomplete.py --ds_path PATH [--groups GROUP,...] [--scan_only]
"""

import argparse
import multiprocessing
import re
from collections import Counter

import tqdm
from prepare_randomized import rand_end_days, shifted
from rslearn.dataset import Dataset, Window
from rslearn.dataset.manage import prepare_dataset_windows
from upath import UPath

LABEL_LAYERS = {"label_category", "label_change_day"}
RAND_LAYER_RE = re.compile(r"^(freq|infreq)_rand_(\d+)$")


def get_incomplete_layers(window: Window) -> list[str]:
    """Get the prepared image layers of the window that are not completed."""
    completed = {layer for layer, group_idx in window.list_completed_layers()}
    return sorted(
        layer_name
        for layer_name, layer_data in window.load_layer_datas().items()
        if layer_name not in LABEL_LAYERS
        and layer_data.serialized_item_groups
        and layer_name not in completed
    )


def scan_window(args: tuple[str, str, str]) -> tuple[str, str, list[str]]:
    """Scan one window, returning (group, name, incomplete layers)."""
    ds_path, group, name = args
    window = Dataset(UPath(ds_path)).load_windows(groups=[group], names=[name])[0]
    return group, name, get_incomplete_layers(window)


def fix_window(args: tuple[str, str, str, list[str]]) -> int:
    """Delete the partial outputs of the layers and prepare them again.

    Returns:
        the number of item groups in the re-prepared layers.
    """
    ds_path, group, name, layers = args
    dataset = Dataset(UPath(ds_path))
    window = dataset.load_windows(groups=[group], names=[name])[0]

    layer_datas = window.load_layer_datas()
    for layer_name in layers:
        num_groups = len(layer_datas[layer_name].serialized_item_groups)
        for group_idx in range(num_groups):
            layer_dir = window.get_layer_dir(layer_name, group_idx)
            if layer_dir.exists():
                for fname in layer_dir.rglob("*"):
                    if fname.is_file():
                        fname.unlink()

    # Layers that are prepared from the window time range directly.
    direct = [layer for layer in layers if not RAND_LAYER_RE.match(layer)]
    if direct:
        prepare_dataset_windows(
            Dataset(UPath(ds_path), enabled_layers=direct), [window], force=True
        )
    # Randomized layers, by slot.
    rand_by_slot: dict[int, list[str]] = {}
    for layer in layers:
        match = RAND_LAYER_RE.match(layer)
        if match:
            rand_by_slot.setdefault(int(match.group(2)), []).append(layer)
    for slot, slot_layers in rand_by_slot.items():
        prepare_dataset_windows(
            Dataset(UPath(ds_path), enabled_layers=slot_layers),
            [shifted(window, rand_end_days(window.name, slot))],
            force=True,
        )

    layer_datas = window.load_layer_datas()
    return sum(len(layer_datas[layer].serialized_item_groups) for layer in layers)


def main() -> None:
    """Scan and fix the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--groups",
        default=None,
        help="Comma-separated window groups to process (default all)",
    )
    parser.add_argument(
        "--scan_only", action="store_true", help="Only report incomplete layers"
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    groups = args.groups.split(",") if args.groups else None
    windows = dataset.load_windows(groups=groups, workers=args.workers)
    jobs = [(args.ds_path, window.group, window.name) for window in windows]
    incomplete: list[tuple[str, str, list[str]]] = []
    with multiprocessing.Pool(args.workers) as pool:
        for group, name, layers in tqdm.tqdm(
            pool.imap_unordered(scan_window, jobs), total=len(jobs), desc="Scanning"
        ):
            if layers:
                incomplete.append((group, name, layers))

    layer_counts = Counter(layer for _, _, layers in incomplete for layer in layers)
    print(
        f"{len(windows) - len(incomplete)}/{len(windows)} windows complete; "
        f"incomplete layers: {dict(sorted(layer_counts.items()))}"
    )
    if args.scan_only or not incomplete:
        return

    fix_jobs = [
        (args.ds_path, group, name, layers) for group, name, layers in incomplete
    ]
    with multiprocessing.Pool(args.workers) as pool:
        for _ in tqdm.tqdm(
            pool.imap_unordered(fix_window, fix_jobs),
            total=len(fix_jobs),
            desc="Re-preparing",
        ):
            pass
    print(
        f"re-prepared {sum(layer_counts.values())} layers in {len(incomplete)} windows"
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    main()
