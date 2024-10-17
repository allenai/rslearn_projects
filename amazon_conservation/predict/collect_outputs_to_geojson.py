"""Create a GeoJSON showing all the predictions.

It uses output of the JSON file created by find_windows_with_outputs.py so that the
features can be labeled with an "index" property that corresponds to their positions in
the web app.
"""

import argparse
import json
import multiprocessing
from typing import Any

import shapely
import tqdm
from rslearn.utils.mp import star_imap_unordered
from upath import UPath


def get_feature(index: int, window_root: UPath) -> list[dict[str, Any]]:
    # Get the polygon from the info.json.
    with (window_root / "info.json").open() as f:
        info = json.load(f)
    shp = shapely.from_wkt(info["wkt"])
    geom_dict = json.loads(shapely.to_geojson(shp))

    # Also need category.
    with (window_root / "layers" / "output" / "data.geojson").open() as f:
        output_data = json.load(f)
    category = output_data["features"][0]["properties"]["new_label"]

    properties = dict(
        index=index,
        date=info["date"],
        category=category,
        window_name=window_root.name,
    )

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": geom_dict,
        "tippecanoe": dict(
            layer=category,
        ),
    }


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_path", help="Dataset root path", type=str, required=True)
    parser.add_argument(
        "--group",
        help="Name of the group",
        type=str,
        default="default",
    )
    parser.add_argument(
        "--windows_fname",
        help="Path to JSON file containing list of window names created by find_windows_with_outputs.py",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--workers",
        help="Number of worker processes to use",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--out_fname",
        help="Filename to write GeoJSON",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    jobs = []
    with open(args.windows_fname) as f:
        for index, window_name in enumerate(json.load(f)):
            jobs.append(
                dict(
                    index=index + 1,
                    window_root=ds_path / "windows" / args.group / window_name,
                )
            )

    p = multiprocessing.Pool(args.workers)
    features = list(
        tqdm.tqdm(star_imap_unordered(p, get_feature, jobs), total=len(jobs))
    )

    with open(args.out_fname, "w") as f:
        json.dump(
            {
                "type": "FeatureCollection",
                "features": features,
            },
            f,
        )
