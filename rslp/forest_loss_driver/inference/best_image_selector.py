"""Select the best Satelitte Image in a given window for a forest loss driver prediction."""

import json
import multiprocessing

import numpy as np
import tqdm
from PIL import Image
from upath import UPath

from rslp.log_utils import get_logger

logger = get_logger(__name__)


def select_best_images(window_path: UPath) -> None:
    """Select the best images for the specified window.

    Best just means least cloudy pixels based on a brightness threshold.

    It writes best pre and post images to the best_pre_X/best_post_X layers and also
    produces a best_times.json indicating the timestamp of the images selected for
    those layers.

    Args:
        window_path: the window root.
    """
    num_outs = 3
    min_choices = 5

    items_fname = window_path / "items.json"
    if not items_fname.exists():
        return

    # Get the timestamp of each expected layer.
    layer_times = {}
    with items_fname.open() as f:
        item_data = json.load(f)
        for layer_data in item_data:
            layer_name = layer_data["layer_name"]
            if "planet" in layer_name:
                continue
            for group_idx, group in enumerate(layer_data["serialized_item_groups"]):
                if group_idx == 0:
                    cur_layer_name = layer_name
                else:
                    cur_layer_name = f"{layer_name}.{group_idx}"
                layer_times[cur_layer_name] = group[0]["geometry"]["time_range"][0]

    # Find best pre and post images.
    image_lists: dict = {"pre": [], "post": []}
    options = window_path.glob("layers/*/R_G_B/image.png")
    for fname in options:
        # "pre" or "post"
        layer_name = fname.parent.parent.name
        k = layer_name.split(".")[0].split("_")[0]
        if "planet" in k or "best" in k:
            continue
        with fname.open("rb") as f:
            im = np.array(Image.open(f))[32:96, 32:96, :]
        image_lists[k].append((im, fname))

    # Copy the images to new "best" layer.
    # Keep track of the timestamps and write them to a separate file.
    best_times = {}
    for k, image_list in image_lists.items():
        if len(image_list) < min_choices:
            return
        image_list.sort(
            key=lambda t: np.count_nonzero(
                (t[0].max(axis=2) == 0) | (t[0].min(axis=2) > 140)
            )
        )
        for idx, (im, fname) in enumerate(image_list[0:num_outs]):
            dst_layer = f"best_{k}_{idx}"
            layer_dir = window_path / "layers" / dst_layer
            (layer_dir / "R_G_B").mkdir(parents=True, exist_ok=True)
            fname.fs.cp(fname.path, (layer_dir / "R_G_B" / "image.png").path)
            (layer_dir / "completed").touch()

            src_layer = fname.parent.parent.name
            layer_time = layer_times[src_layer]
            best_times[dst_layer] = layer_time
    logger.info(f"Writing best_times.json to {window_path / 'best_times.json'}...")
    with (window_path / "best_times.json").open("w") as f:
        json.dump(best_times, f)


def select_best_images_pipeline(ds_path: str | UPath, workers: int = 64) -> None:
    """Run the best image pipeline.

    This picks the best three pre/post images and puts them in the corresponding layers
    so the model can read them.

    It is based on amazon_conservation/make_dataset/select_images.py.

    Args:
        ds_path: the dataset root path
        workers: number of workers to use.

    Outputs:
        best_times.json: a file containing the timestamps of the best images for each layer.
    """
    ds_path = UPath(ds_path) if not isinstance(ds_path, UPath) else ds_path
    window_paths = list(ds_path.glob("windows/*/*"))
    p = multiprocessing.Pool(workers)
    outputs = p.imap_unordered(select_best_images, window_paths)
    for _ in tqdm.tqdm(outputs, total=len(window_paths)):
        pass
    p.close()
