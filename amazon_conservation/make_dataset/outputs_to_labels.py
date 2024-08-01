"""This converts outputs from multisat.train.infer --out_dir to labels in rslearn dataset.
It is a bit annoying because the labels will overwrite the old ones.
But we are only getting outputs for peru2 I think so it seems okay.
"""

import json
import os

categories = [
    "mining",
    "agriculture",
    "airstrip",
    "road",
    "logging",
    "burned",
    "landslide",
    "hurricane",
    "river",
    "none",
]
input_dirs = [
    "/data/favyenb/rslearn_amazon_conservation_closetime/outputs/v71_peru2/",
    "/data/favyenb/rslearn_amazon_conservation_closetime/outputs/v72_peru2/",
]
out_dir = "/data/favyenb/rslearn_amazon_conservation_closetime/windows/peru2/"
annotations = []
for fname in os.listdir(input_dirs[0]):
    with open(os.path.join(input_dirs[0], fname)) as f:
        category0 = categories[int(f.read())]
    with open(os.path.join(input_dirs[1], fname)) as f:
        category1 = categories[int(f.read())]
    example_id = fname.split("_amazon_conservation.")[0]
    out_fname = os.path.join(out_dir, example_id, "label.json")
    assert not os.path.exists(out_fname)
    with open(out_fname, "w") as f:
        json.dump(
            {
                "old_label": category0,
                "new_label": category1,
            },
            f,
        )
