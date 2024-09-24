"""Download the SkySat images to local filesystem."""

import os
import random
import shutil

from upath import UPath

if __name__ == "__main__":
    ds_path = UPath(
        "gs://rslearn-eai/datasets/maldives_ecosystem_mapping/dataset_v1/20240924/"
    )
    out_dir = "/home/favyenb/vis/"
    tif_fnames = list(
        ds_path.glob(
            "windows/images_skysat/*/layers/planet/b01_b02_b03_b04/geotiff.tif"
        )
    )
    random.shuffle(tif_fnames)
    for tif_fname in tif_fnames:
        print(tif_fname)
        local_fname = os.path.join(
            out_dir, tif_fname.parent.parent.parent.parent.name + ".tif"
        )
        with tif_fname.open("rb") as src:
            with open(local_fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
