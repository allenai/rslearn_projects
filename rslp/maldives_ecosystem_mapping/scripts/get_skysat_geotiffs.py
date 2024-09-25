"""Download the SkySat images to local filesystem."""

import argparse
import os
import random
import shutil

from upath import UPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Skysat GeoTIFF images",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Path to rslearn dataset for Maldives ecosystem mapping project",
        default="gs://rslearn-eai/datasets/maldives_ecosystem_mapping/dataset_v1/20240924/",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Local directory to write the GeoTIFFs",
        required=True,
    )
    parser.add_argument(
        "--count",
        type=int,
        help="How many GeoTIFFs to download (default is to get all of them)",
        default=None,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)
    tif_fnames = list(
        ds_path.glob(
            "windows/images_skysat/*/layers/planet/b01_b02_b03_b04/geotiff.tif"
        )
    )
    random.shuffle(tif_fnames)

    if args.count:
        tif_fnames = tif_fnames[0 : args.count]

    for tif_fname in tif_fnames:
        print(tif_fname)
        local_fname = os.path.join(
            args.out_dir, tif_fname.parent.parent.parent.parent.name + ".tif"
        )
        with tif_fname.open("rb") as src:
            with open(local_fname, "wb") as dst:
                shutil.copyfileobj(src, dst)

        vis_fname = (
            tif_fname.parent.parent.parent
            / "skysat_vis"
            / "b03_b02_b01"
            / "geotiff.tif"
        )
        if not vis_fname.exists():
            print(
                f"warning: {vis_fname} does not exist, only got the non-vis version for this one"
            )
            continue
        local_fname = os.path.join(
            args.out_dir, vis_fname.parent.parent.parent.parent.name + "_vis.tif"
        )
        print(vis_fname)
        with vis_fname.open("rb") as src:
            with open(local_fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
