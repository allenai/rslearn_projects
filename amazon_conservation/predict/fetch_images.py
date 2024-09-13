"""Fetch images in forest loss driver prediction dataset from GCS."""

import random
import shutil

from upath import UPath

ds_path = UPath(
    "gs://rslearn-eai/datasets/forest_loss_driver/prediction/dataset_20240828/"
)
paths = list((ds_path / "windows" / "default").iterdir())
paths = random.sample(paths, 100)
# paths = [ds_path / "windows" / "default" / "feat_x_1170630_2152543_4759_47488"]
for path in paths:
    path = UPath(path)
    label = path.name
    for fname in path.glob("layers/*/R_G_B/image.png"):
        local_fname = f"{label}_{fname.parent.parent.name}.png"
        with fname.open("rb") as src:
            with open(local_fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
