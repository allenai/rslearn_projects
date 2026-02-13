"""Set window properties to configure class-balanced folds for training."""

import json
import random
import sys

import tqdm
from rslearn.utils.fsspec import open_atomic
from upath import UPath

if __name__ == "__main__":
    ds_path = UPath(sys.argv[1])
    num_folds = int(sys.argv[2])

    # Load the categories of each window.
    window_dirs = list((ds_path / "windows" / "train").iterdir())
    windows_by_category: dict[str, list[UPath]] = {}
    for window_dir in tqdm.tqdm(window_dirs, desc="Loading categories"):
        fname = window_dir / "layers" / "label" / "data.geojson"
        with fname.open() as f:
            fc = json.load(f)
        category = fc["features"][0]["properties"]["label"]
        if category not in windows_by_category:
            windows_by_category[category] = []
        windows_by_category[category].append(window_dir)

    category_counts = {
        category: len(window_list)
        for category, window_list in windows_by_category.items()
    }
    min_count = min(category_counts.values())
    print(f"Got counts={category_counts} min_count={min_count}")

    for fold_idx in tqdm.tqdm(list(range(num_folds)), desc="Writing folds"):
        for category, window_list in windows_by_category.items():
            selected_windows = random.sample(window_list, min_count)
            for window in selected_windows:
                with (window / "metadata.json").open() as f:
                    metadata = json.load(f)
                metadata["options"][f"fold{fold_idx}"] = f"fold{fold_idx}"
                with open_atomic(window / "metadata.json", "w") as f:
                    json.dump(metadata, f)
