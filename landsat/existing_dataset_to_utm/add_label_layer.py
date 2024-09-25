"""In prepare_windows.py we just created gt.json and mask.png not in actual layer.

So here we just put the data into the layer.

Oh and we also assign split. And mark the landsat layer completed and make sure the
layer has the bounds file.
"""

import hashlib
import json
import multiprocessing
import sys

import shapely
import tqdm
from rslearn.dataset import Window
from rslearn.utils import Feature, STGeometry
from rslearn.utils.fsspec import open_atomic
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath


def handle_window(window_root: UPath):
    """Create label/mask layers and assign split for the given window.

    Args:
        window_root: the root of the window to process.
    """
    window = Window.load(window_root)

    # Convert detections.
    features = []
    with (window_root / "gt.json").open() as f:
        for x1, y1, x2, y2, category in json.load(f):
            assert category == "vessel"
            shp = shapely.box(
                window.bounds[0] + x1,
                window.bounds[1] + y1,
                window.bounds[0] + x2,
                window.bounds[1] + y2,
            )
            geom = STGeometry(window.projection, shp, None)
            props = dict(
                category=category,
            )
            features.append(Feature(geom, props))

    layer_dir = window_root / "layers" / "label"
    layer_dir.mkdir(parents=True, exist_ok=True)
    GeojsonVectorFormat().encode_vector(layer_dir, window.projection, features)
    (layer_dir / "completed").touch()

    # Copy mask.png too.
    mask_src = window_root / "mask.png"
    layer_dir = window_root / "layers" / "mask"
    mask_dst = layer_dir / "mask" / "image.png"
    mask_dst.parent.mkdir(parents=True, exist_ok=True)
    window_root.fs.copy(mask_src.path, mask_dst.path)
    (layer_dir / "completed").touch()

    # Assign split.
    with (window_root / "metadata.json").open() as f:
        metadata = json.load(f)
    if "options" not in metadata:
        metadata["options"] = {}
    is_val = hashlib.sha256(window_root.name.encode()).hexdigest()[0] in ["0", "1"]
    if is_val:
        metadata["options"]["split"] = "val"
    else:
        metadata["options"]["split"] = "train"
    with open_atomic(window_root / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # Mark landsat layer completed (if all bands are available).
    landsat_layer_ok = True
    for band in ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]:
        expected_image_fname = window_root / "layers" / "landsat" / band / "image.png"
        if not expected_image_fname.exists():
            landsat_layer_ok = False
    if landsat_layer_ok:
        (window_root / "layers" / "landsat" / "completed").touch()

    # Put bounds file to landsat and mask layers.
    single_image_fnames = window_root.glob("layers/*/*/image.png")
    for fname in single_image_fnames:
        metadata_fname = fname.parent / "metadata.json"
        if metadata_fname.exists():
            continue
        with metadata_fname.open("w") as f:
            json.dump({"bounds": window.bounds}, f)


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    ds_path = UPath(sys.argv[1])

    metadata_fnames = list(ds_path.glob("windows/*/*/metadata.json"))
    window_roots = [fname.parent for fname in metadata_fnames]

    p = multiprocessing.Pool(64)
    outputs = p.imap_unordered(handle_window, window_roots)
    for _ in tqdm.tqdm(outputs, total=len(window_roots)):
        pass
    p.close()
