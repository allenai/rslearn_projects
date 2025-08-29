"""Partners said we should focus on Amazon basin.

So this script moves windows that were outside of that polygon to different groups.
"""

import multiprocessing
import shutil

import tqdm
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

BOUNDARY_FNAME = "/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/amazon_boundary.geojson"
DATASET_PATH = "/weka/dfive-default/rslearn-eai/datasets/forest_loss_driver/dataset_v1/brazil_and_colombia/"
TARGET_GROUPS = {
    "20250428_brazil": "20250428_brazil_outsideamazon",
    "20250428_colombia": "20250428_colombia_outsideamazon",
}


def process_window(window: Window, boundary_geom: STGeometry) -> None:
    """Move the window to new group if it does not intersect the bounding polygon."""
    window_geom = window.get_geometry()
    if boundary_geom.intersects(window_geom):
        return

    dst_group = TARGET_GROUPS[window.group]
    new_window_path = window.path.parent.parent / dst_group / window.name
    print("move", window.path, new_window_path)
    # Assume local filesystem here since we're using mounted WEKA.
    shutil.move(
        window.path.path,
        new_window_path.path,
    )
    # Fix the group in metadata.json.
    window.path = new_window_path
    window.group = dst_group
    window.save()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    # Read boundary polygon.
    features = GeojsonVectorFormat().decode_from_file(UPath(BOUNDARY_FNAME))
    boundary_geom = features[0].geometry

    dataset = Dataset(UPath(DATASET_PATH))
    windows = dataset.load_windows(
        groups=list(TARGET_GROUPS.keys()), show_progress=True, workers=128
    )
    jobs = [
        dict(
            window=window,
            boundary_geom=boundary_geom,
        )
        for window in windows
    ]
    p = multiprocessing.Pool(128)
    outputs = star_imap_unordered(p, process_window, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
