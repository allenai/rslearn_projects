"""Select windows for phase 2 annotation.

- Select 50 each from Brazil/Colombia predicted as road/logging/mining/river/landslide (500 total)
- Select 150 each from Brazil/Colombia predicted not as the above classes with max(probabilities) < 0.6 (300 total)
"""

import multiprocessing
import random
import shutil
from typing import Any

import tqdm
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.grid_index import GridIndex
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

DATASET_PATH = "/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/"
TARGET_GROUPS = {
    "20250428_brazil": "20250428_brazil_phase2",
    "20250428_colombia": "20250428_colombia_phase2",
}
RARE_CATEGORIES = ["road", "logging", "mining", "river", "landslide"]
PROB_THRESHOLD = 0.6
DISTANCE_THRESHOLD = 500 / 111111


def load_properties(window: Window) -> dict[str, Any] | None:
    """Load the properties of the label for this window."""
    if not window.is_layer_completed("output"):
        return None
    features = GeojsonVectorFormat().decode_vector(
        window.get_layer_dir("output"),
        window.projection,
        (
            window.bounds[0] - 1,
            window.bounds[1] - 1,
            window.bounds[2] + 1,
            window.bounds[3] + 1,
        ),
    )
    if len(features) != 1:
        raise ValueError(
            f"got {len(features)} features for window {window.group}/{window.name}"
        )
    return features[0].properties


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    # Load windows.
    dataset = Dataset(UPath(DATASET_PATH))
    windows = dataset.load_windows(
        groups=list(TARGET_GROUPS.keys()), show_progress=True, workers=64
    )

    # Load prediction feature properties.
    p = multiprocessing.Pool(64)
    window_properties = list(
        tqdm.tqdm(
            p.imap(load_properties, windows),
            total=len(windows),
            desc="Loading output properties",
        )
    )

    # Now create candidates for the different selection criteria.
    by_class_options: dict[tuple[str, str], list[Window]] = {
        (group, category): []
        for group in TARGET_GROUPS.keys()
        for category in RARE_CATEGORIES
    }
    by_prob_options: dict[str, list[Window]] = {
        group: [] for group in TARGET_GROUPS.keys()
    }
    for window, properties in zip(windows, window_properties):
        if properties is None:
            continue

        category = properties["new_label"]
        if category in RARE_CATEGORIES:
            by_class_options[(window.group, category)].append(window)
        elif max(properties["probs"]) < 0.6:
            by_prob_options[window.group].append(window)

    for (group, category), candidates in by_class_options.items():
        print(
            f"got {len(candidates)} options by class for group={group} category={category}"
        )
    for group, candidates in by_prob_options.items():
        print(f"got {len(candidates)} options by prob for group={group}")

    # Select windows, we make sure their center points are at least 500 m away from
    # each other.
    grid_index = GridIndex(size=DISTANCE_THRESHOLD)
    selected: list[Window] = []

    def contains_bbox(box: tuple[float, float, float, float]) -> bool:
        """Check whether the box intersects a point in grid_index."""
        for other in grid_index.query(box):
            if (
                other[0] > box[0]
                and other[1] > box[1]
                and other[0] < box[2]
                and other[1] < box[3]
            ):
                return True
        return False

    def add_random_sample_of_windows(windows: list[Window], max_count: int) -> int:
        """Add a random sample of windows from the list to the selected set."""
        # Add up to max_count from the windows list.
        random.shuffle(windows)
        cur_selected: list[Window] = []
        for window in windows:
            center_point = (
                window.get_geometry().to_projection(WGS84_PROJECTION).shp.centroid
            )
            if contains_bbox(
                (
                    center_point.x - DISTANCE_THRESHOLD,
                    center_point.y - DISTANCE_THRESHOLD,
                    center_point.x + DISTANCE_THRESHOLD,
                    center_point.y + DISTANCE_THRESHOLD,
                )
            ):
                continue

            cur_selected.append(window)
            grid_index.insert(
                (center_point.x, center_point.y, center_point.x, center_point.y),
                (center_point.x, center_point.y),
            )
            if len(cur_selected) >= max_count:
                break

        selected.extend(cur_selected)
        return len(cur_selected)

    for (group, category), candidates in by_class_options.items():
        count = add_random_sample_of_windows(candidates, 50)
        print(
            f"by class group={group} category={category} picked {count}/{len(candidates)} windows"
        )
    for group, candidates in by_prob_options.items():
        count = add_random_sample_of_windows(candidates, 150)
        print(f"by prob group={group} picked {count}/{len(candidates)} windows")
    print(f"got {len(selected)} total to remap")

    for window in selected:
        new_group = TARGET_GROUPS[window.group]
        new_window_path = window.path.parent.parent / new_group / window.name
        print("move", window.path, new_window_path)
        # Assume local filesystem here since we're using mounted WEKA.
        shutil.move(
            window.path.path,
            new_window_path.path,
        )
        # Fix the group in metadata.json.
        window.path = new_window_path
        window.group = new_group
        window.save()
