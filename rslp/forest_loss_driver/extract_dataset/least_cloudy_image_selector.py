"""Select the least cloudy Satelitte Image in a given window for a forest loss driver prediction."""

import json
import multiprocessing
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import numpy as np
import tqdm
from rslearn.config import LayerType
from rslearn.data_sources import Item
from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.utils.raster_format import SingleImageRasterFormat
from upath import UPath

from rslp.log_utils import get_logger

logger = get_logger(__name__)


@dataclass
class SelectLeastCloudyImagesArgs:
    """Arguments for select_least_cloudy_images_pipeline.

    Args:
        min_choices: the minimum number of images to select.
        num_outs: the number of best images to select.
        workers: the number of workers to use.
    """

    min_choices: int = 5
    num_outs: int = 3
    workers: int = min(multiprocessing.cpu_count(), 128)


def compute_cloudiness_score(im: np.ndarray) -> int:
    """Compute the cloudiness score of an image.

    Uses the R, G, B channels to compute the cloudiness score.
    This heuristic is specific to the Forest Loss Region where cloudy images
    will have very little green and other images will have a bunch as it is
    in the forest.

    Args:
        im: the image to score.

    Returns:
        The cloudiness score of the image.
    """
    # Count pixels that are either missing (0) or cloudy.
    return np.count_nonzero((im.max(axis=0) == 0) | (im.min(axis=0) > 140))


def select_least_cloudy_images(
    window: Window,
    dataset: Dataset,
    num_outs: int,
    min_choices: int,
) -> None:
    """Select the least cloudy images for the specified window.

    It writes least cloudy pre and post images to the best_pre_X/best_post_X layers and also
    produces a best_times.json indicating the timestamp of the images selected for
    those layers.

    Args:
        window: the Window.
        dataset: the Dataset.
        num_outs: the number of least cloudy images to select.
        min_choices: the minimum number of images to select.
    """
    layer_datas = window.load_layer_datas()
    if len(layer_datas) == 0:
        return

    # Get the timestamp of each expected layer.
    # layer_times is map from (layer_name, group_idx) to the timestamp.
    layer_times: dict[tuple[str, int], datetime] = {}
    for layer_name, layer_data in layer_datas.items():
        pre_or_post = layer_name.split("_")[0]
        if pre_or_post not in ["pre", "post"]:
            continue
        for group_idx, group in enumerate(layer_data.serialized_item_groups):
            item = Item.deserialize(group[0])
            layer_times[(layer_name, group_idx)] = item.geometry.time_range[0]

    # Find least cloudy pre and post images.
    # layer_cloudiness maps from (layer_name, group_idx) => (cloudiness, RGB image).
    layer_cloudiness: dict[tuple[str, int], tuple[int, np.ndarray]] = {}
    for layer_name, group_idx in layer_times.keys():
        if not window.is_layer_completed(layer_name, group_idx):
            continue

        layer_config = dataset.layers[layer_name]
        assert layer_config.type == LayerType.RASTER
        assert len(layer_config.band_sets) == 1
        raster_format = layer_config.band_sets[0].instantiate_raster_format()
        bands = layer_config.band_sets[0].bands

        raster_dir = window.get_raster_dir(layer_name, bands, group_idx)
        array = raster_format.decode_raster(
            raster_dir, window.projection, window.bounds
        )

        # Get RGB by selecting (B04, B03, B02) and dividing by 10.
        rgb_indices = (
            bands.index("B04"),
            bands.index("B03"),
            bands.index("B02"),
        )
        rgb_array = array[rgb_indices, :, :] // 10

        # Use the center crop since that's the most important part.
        center_crop = rgb_array[:, 32:96, 32:96]
        cloudiness = compute_cloudiness_score(center_crop)

        layer_cloudiness[(layer_name, group_idx)] = (cloudiness, rgb_array)

    # Determine the least cloudy pre and post images.
    # We copy those images to a new "best_X" layer.
    # We keep track of the timestamps of the source images and write them to a separate
    # file, which is used by forest-loss.allen.ai. We also copy the layer datas to
    # match with the new layer, so that any users trying to find the time range via the
    # layer data (such as earth-system-studio rslearn_import.py script) can find it
    # there.
    least_cloudy_times = {}
    for pre_or_post in ["pre", "post"]:
        image_list = [
            (layer_name, group_idx, cloudiness, rgb_array)
            for (layer_name, group_idx), (
                cloudiness,
                rgb_array,
            ) in layer_cloudiness.items()
            if layer_name.startswith(pre_or_post)
        ]
        if len(image_list) < min_choices:
            return
        # Sort by cloudiness (third element of tuple) so we can pick the least cloudy.
        image_list.sort(key=lambda t: t[2])
        for idx, (layer_name, group_idx, _, rgb_array) in enumerate(
            image_list[0:num_outs]
        ):
            # The layer name for the best images is e.g. "best_pre_0".
            dst_layer_name = f"best_{pre_or_post}_{idx}"

            # Write the RGB version of the data.
            dst_raster_dir = window.get_raster_dir(dst_layer_name, ["R", "G", "B"])
            SingleImageRasterFormat().encode_raster(
                dst_raster_dir,
                window.projection,
                window.bounds,
                np.clip(rgb_array, 0, 255).astype(np.uint8),
            )
            window.mark_layer_completed(dst_layer_name)

            layer_time = layer_times[(layer_name, group_idx)]
            least_cloudy_times[dst_layer_name] = layer_time.isoformat()

            # Copy the items for the source layer under the destination layer.
            serialized_items = layer_datas[layer_name].serialized_item_groups[group_idx]
            layer_datas[dst_layer_name] = WindowLayerData(
                layer_name=dst_layer_name,
                serialized_item_groups=[serialized_items],
            )

    output_fname = "least_cloudy_times.json"
    window_path = window.storage.get_window_root(window.group, window.name)
    logger.debug(f"Writing least cloudy times to {window_path / output_fname}...")
    with (window_path / output_fname).open("w") as f:
        json.dump(least_cloudy_times, f)

    logger.debug(f"Saving updated layer datas for {window_path}")
    window.save_layer_datas(layer_datas)


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
    dataset = Dataset(ds_path)
    windows = dataset.load_windows(workers=select_least_cloudy_images_args.workers)
    select_least_cloudy_images_partial = partial(
        select_least_cloudy_images,
        dataset=dataset,
        num_outs=select_least_cloudy_images_args.num_outs,
        min_choices=select_least_cloudy_images_args.min_choices,
    )
    p = multiprocessing.Pool(select_least_cloudy_images_args.workers)
    outputs = p.imap_unordered(
        select_least_cloudy_images_partial,
        windows,
    )
    for _ in tqdm.tqdm(outputs, total=len(windows)):
        pass
    p.close()
