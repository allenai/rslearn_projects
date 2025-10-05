"""Create label_raster from label."""

import argparse
import multiprocessing

import numpy as np
import tqdm
from rslearn.dataset.dataset import Dataset
from rslearn.dataset.window import Window
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

CLASS_NAMES = [
    "invalid",
    "Agriculture/Settlement",
    "Grassland/barren",
    "Herbaceous wetland",
    "Lava forest",
    "Montane forest",
    "Open water",
    "Shrubland/Savanna",
    "Urban/dense development",
    "Woodland forest (>40% canopy)",
]
PROPERTY_NAME = "lulc"
BAND_NAME = "label"


def create_label_raster(window: Window) -> None:
    """Create label raster for the given window."""
    label_dir = window.get_layer_dir("label")
    features = GeojsonVectorFormat().decode_vector(
        label_dir, window.projection, window.bounds
    )
    class_name = features[0].properties[PROPERTY_NAME]
    class_id = CLASS_NAMES.index(class_name)

    # Draw the class_id in the middle 1x1 of the raster.
    raster = np.zeros(
        (1, window.bounds[3] - window.bounds[1], window.bounds[2] - window.bounds[0]),
        dtype=np.uint8,
    )
    raster[:, raster.shape[1] // 2, raster.shape[2] // 2] = class_id
    raster_dir = window.get_raster_dir("label_raster", [BAND_NAME])
    GeotiffRasterFormat().encode_raster(
        raster_dir, window.projection, window.bounds, raster
    )
    window.mark_layer_completed("label_raster")


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ds_path",
        type=str,
        required=True,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=64,
        help="Number of worker processes to use",
    )
    args = parser.parse_args()

    dataset = Dataset(UPath(args.ds_path))
    windows = dataset.load_windows(workers=args.workers, show_progress=True)
    p = multiprocessing.Pool(args.workers)
    outputs = p.imap_unordered(create_label_raster, windows)
    for _ in tqdm.tqdm(outputs, total=len(windows)):
        pass
    p.close()
