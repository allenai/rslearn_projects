"""Get the number of vessels that have values for each attribute in the dataset."""

import argparse
import json
import multiprocessing
from typing import Any

import tqdm
from upath import UPath

import rslp.utils.mp


def get_attributes(info_fname: UPath) -> dict[str, Any]:
    """Get the attributes that the GeoJSON for info layer includes.

    Args:
        info_fname: the filename for a layers/info/data.geojson file.

    Returns:
        the attributes that are present in the JSON.
    """
    with info_fname.open() as f:
        fc = json.load(f)

    if len(fc["features"]) != 1:
        raise ValueError(
            f"expected info JSON {info_fname} to contain exactly one GeoJSON Feature"
        )

    properties = fc["features"][0]["properties"]
    attributes = {
        "length": 0,
        "width": 0,
        "cog": 0,
        "sog": 0,
        "type": 0,
    }
    for attr in attributes.keys():
        if attr in properties:
            attributes[attr] = 1
    return attributes


if __name__ == "__main__":
    rslp.utils.mp.init_mp()

    parser = argparse.ArgumentParser(
        description="Get dataset statistics",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Dataset path",
        required=True,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    fnames = list(ds_path.glob("windows/*/*/layers/info/data.geojson"))

    # Identify attributes present in each GeoJSON thread in parallel (via the
    # get_attributes function). Then here we go through and add them up.
    p = multiprocessing.Pool(32)
    outputs = p.imap_unordered(get_attributes, fnames)
    total_attributes = {}
    for attributes in tqdm.tqdm(outputs, total=len(fnames)):
        for attr, count in attributes.items():
            if attr not in total_attributes:
                total_attributes[attr] = 0
            total_attributes[attr] += count

    print(total_attributes)
