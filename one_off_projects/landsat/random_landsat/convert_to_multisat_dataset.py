import glob
import json
import os
import shutil

in_dir = "/data/favyenb/rslearn_landsat/windows/utm/"
out_dir = "/data/favyenb/rslearn_landsat/landsat_utm_as_multisat_dataset/"
split_fname = "/data/favyenb/rslearn_landsat/landsat_utm_as_multisat_dataset.json"

fnames = glob.glob("*/layers/landsat/B8/image.png", root_dir=in_dir)

example_ids = []
for fname in fnames:
    example_id = fname.split("/")[0]
    example_ids.append(example_id)
    example_dir = os.path.join(out_dir, example_id)
    image_dir = os.path.join(example_dir, "images", "image")
    os.makedirs(image_dir, exist_ok=True)
    with open(os.path.join(example_dir, "gt.json"), "w") as f:
        f.write("[]")
    for band in ["B2", "B3", "B4", "B5", "B6", "B7", "B8"]:
        shutil.copyfile(
            os.path.join(in_dir, fname.replace("B8", band)),
            os.path.join(image_dir, band.lower() + ".png"),
        )

with open(split_fname, "w") as f:
    json.dump(example_ids, f)
