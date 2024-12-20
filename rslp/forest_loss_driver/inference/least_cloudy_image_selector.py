"""Select the least cloudy Satelitte Image in a given window for a forest loss driver prediction."""

import json
import multiprocessing
from functools import partial

import numpy as np
import tqdm
from PIL import Image
from upath import UPath

from rslp.forest_loss_driver.inference.config import SelectLeastCloudyImagesArgs
from rslp.log_utils import get_logger

logger = get_logger(__name__)


def select_least_cloudy_images(
    window_path: UPath,
    num_outs: int,
    min_choices: int,
) -> None:
    """Select the least cloudy images for the specified window.

    It writes least cloudy pre and post images to the best_pre_X/best_post_X layers and also
    produces a best_times.json indicating the timestamp of the images selected for
    those layers.

    Args:
        window_path: the window root.
        num_outs: the number of least cloudy images to select.
        min_choices: the minimum number of images to select.
    """
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

    # Find least cloudy pre and post images.
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

    # Copy the images to new "least_cloudy" layer.
    # Keep track of the timestamps and write them to a separate file.
    least_cloudy_times = {}
    for k, image_list in image_lists.items():
        if len(image_list) < min_choices:
            return
        image_list.sort(
            key=lambda t: np.count_nonzero(
                (t[0].max(axis=2) == 0) | (t[0].min(axis=2) > 140)
            )
        )
        for idx, (im, fname) in enumerate(image_list[0:num_outs]):
            # TODO: f"best_{k}_{idx}" MUST MATCH THE LAYER NAMES IN THE MODEL and DATA CONFIGS
            # TODO: This is very brittle and should not be hidden here
            dst_layer = f"best_{k}_{idx}"
            layer_dir = window_path / "layers" / dst_layer
            (layer_dir / "R_G_B").mkdir(parents=True, exist_ok=True)
            fname.fs.cp(fname.path, (layer_dir / "R_G_B" / "image.png").path)
            (layer_dir / "completed").touch()
            src_layer = fname.parent.parent.name
            layer_time = layer_times[src_layer]
            least_cloudy_times[dst_layer] = layer_time
    output_fname = "least_cloudy_times.json"
    logger.info(f"Writing least cloudy times to {window_path / output_fname}...")
    with (window_path / output_fname).open("w") as f:
        json.dump(least_cloudy_times, f)


def select_least_cloudy_images_pipeline(
    ds_path: str | UPath,
    select_least_cloudy_images_args: SelectLeastCloudyImagesArgs,
) -> None:
    """Run the least cloudy image pipeline.

    This picks the least cloudy three pre/post images and puts them in the corresponding layers
    so the model can read them.

    It is based on amazon_conservation/make_dataset/select_images.py.

    Args:
        ds_path: the dataset root path
        select_least_cloudy_images_args: the arguments for the select_least_cloudy_images step.

    Outputs:
        least_cloudy_times.json: a file containing the timestamps of the least cloudy images for each layer.
    """
    ds_path = UPath(ds_path) if not isinstance(ds_path, UPath) else ds_path
    window_paths = list(ds_path.glob("windows/*/*"))
    p = multiprocessing.Pool(select_least_cloudy_images_args.workers)
    select_least_cloudy_images_partial = partial(
        select_least_cloudy_images,
        num_outs=select_least_cloudy_images_args.num_outs,
        min_choices=select_least_cloudy_images_args.min_choices,
    )
    outputs = p.imap_unordered(
        select_least_cloudy_images_partial,
        window_paths,
    )
    for _ in tqdm.tqdm(outputs, total=len(window_paths)):
        pass
    p.close()
