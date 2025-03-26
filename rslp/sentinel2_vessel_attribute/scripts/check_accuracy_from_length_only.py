"""Ship type accuracy using only length attribute.

It splits up vessels into buckets based on length, and then checks the accuracy when
mapping each bucket to the most common ship type category in the bucket.
"""

import argparse
import json
import multiprocessing
from typing import Any

import tqdm
from upath import UPath

import rslp.utils.mp


def get_json(fname: UPath) -> dict[str, Any]:
    """Read a JSON file. This is for multiprocessing.

    Args:
        fname: the filename to read.

    Returns:
        the decoded JSON from the file.
    """
    with fname.open() as f:
        return json.load(f)


if __name__ == "__main__":
    rslp.utils.mp.init_mp()

    parser = argparse.ArgumentParser(
        description="Determine potential ship type classification accuracy using only length attribute",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Dataset path",
        required=True,
    )
    parser.add_argument(
        "--bucket_size",
        type=int,
        help="Size of buckets to group up vessel length",
        required=True,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    fnames = list(ds_path.glob("windows/default/*/layers/info/data.geojson"))

    p = multiprocessing.Pool(32)
    outputs = p.imap_unordered(get_json, fnames)
    vessel_datas = list(tqdm.tqdm(outputs, total=len(fnames)))
    p.close()

    # Length bucket -> {vessel type -> count}
    # So this gives the count of each vessel type within each bucket.
    buckets: dict[int, dict[str, int]] = {}
    for vessel_data in vessel_datas:
        properties = vessel_data["features"][0]["properties"]

        # Some vessels don't have all the labels we need.
        if "length" not in properties:
            continue
        if "type" not in properties:
            continue

        length_bucket_idx = int(properties["length"]) // args.bucket_size
        if length_bucket_idx not in buckets:
            buckets[length_bucket_idx] = {}
        bucket_dict = buckets[length_bucket_idx]

        vessel_type = properties["type"]
        bucket_dict[vessel_type] = bucket_dict.get(vessel_type, 0) + 1

    correct = 0
    incorrect = 0
    for bucket_idx, bucket_dict in buckets.items():
        # Get most common type in this bucket and add the numbers that would be
        # correctly and incorrectly classified.
        most_common_type = None
        for vessel_type, count in bucket_dict.items():
            if most_common_type is not None and count <= bucket_dict[most_common_type]:
                continue
            most_common_type = vessel_type
        assert most_common_type is not None

        cur_correct = bucket_dict[most_common_type]
        cur_incorrect = 0
        for vessel_type, count in bucket_dict.items():
            if vessel_type == most_common_type:
                continue
            cur_incorrect += count

        lo = bucket_idx * args.bucket_size
        hi = (bucket_idx + 1) * args.bucket_size
        print(
            f"bucket {lo} to {hi}: correct={cur_correct}, incorrect={cur_incorrect}, most_common={most_common_type}"
        )

        correct += cur_correct
        incorrect += cur_incorrect

    accuracy = correct / (correct + incorrect)
    print(f"correct={correct}, incorrect={incorrect}, accuracy={accuracy}")
