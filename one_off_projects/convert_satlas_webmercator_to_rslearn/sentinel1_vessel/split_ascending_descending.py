"""Split Sentinel-1 windows into separate ascending and descending groups.

This script also writes the layer datas file (items.json). It is populated with a
single scene, i.e. the scene referenced in the original annotations (which is saved in
windows/x/y/image_name_from_siv.txt by convert_siv_labels.py).
"""

import argparse
import functools
import json
import multiprocessing
import shutil
from urllib.parse import quote

import tqdm
from upath import UPath

from rslearn.dataset import Dataset, Window, WindowLayerData
from rslearn.data_sources.copernicus import Sentinel1, Sentinel1ProductType, Sentinel1Polarisation


@functools.cache
def get_planetary_computer() -> Sentinel1:
    return Sentinel1(product_type=Sentinel1ProductType.IW_GRDH, polarisation=Sentinel1Polarisation.VV_VH)


def process_window(window: Window) -> None:
    # Get the Sentinel-1 product.
    sentinel1 = get_planetary_computer()
    with (window.path / "image_name_from_siv.txt").open() as f:
        image_name = f.read().strip()
    filter_string = sentinel1._build_filter_string(f"Name eq '{quote(image_name)}'")
    path = f"/Products?$filter={filter_string}&$expand=Attributes"
    try:
        response = sentinel1._get(path)
    except Exception as e:
        # Got an error, user's going to have to rerun this script with the remaining windows.
        print(f"ignoring exception {e} for window {window.name}")
        return
    products = response["value"]
    assert len(products) == 1
    product = products[0]

    # Get the orbit direction, which should be ascending or descending.
    attribute_by_name = {
        attribute["Name"]: attribute["Value"]
        for attribute in product["Attributes"]
    }
    orbit_direction = attribute_by_name["orbitDirection"]
    assert orbit_direction in ["ASCENDING", "DESCENDING"]

    # Write the items.json.
    item = sentinel1._product_to_item(product)
    window.save_layer_datas({"sentinel1": WindowLayerData("sentinel1", [[item.serialize()]])})

    # Move to separate ascending or descending group.
    with (window.path / "metadata.json").open() as f:
        metadata = json.load(f)
    group_name = window.path.parent.name
    new_group_name = f"{group_name}_{orbit_direction.lower()}"
    new_window_dir = window.path.parent.parent / new_group_name / window.path.name
    print("move", window.path, new_window_dir)
    new_window_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(
        window.path.path,
        new_window_dir.path,
    )
    metadata["group"] = new_group_name
    with (new_window_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(
        description="Split ascending and descending Sentinel-1 windows",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="The Sentinel-1 rslearn dataset path",
        required=True,
    )
    parser.add_argument(
        "--groups",
        type=str,
        nargs="*",
        help="Which group(s) to apply the splitting on",
        required=True,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes",
        default=64,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    windows = Dataset(ds_path).load_windows(groups=args.groups, show_progress=True, workers=args.workers)
    p = multiprocessing.Pool(args.workers)
    outputs = p.imap_unordered(process_window, windows)
    for _ in tqdm.tqdm(outputs, total=len(windows)):
        pass
    p.close()
