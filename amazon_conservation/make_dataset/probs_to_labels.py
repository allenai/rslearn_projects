"""This converts outputs from multisat.train.infer --out_dir to labels in rslearn dataset.
It is a bit annoying because the labels will overwrite the old ones.
But we are only getting outputs for peru2 I think so it seems okay.
"""

import json
import os

import numpy as np

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
    "/data/favyenb/rslearn_amazon_conservation_closetime/outputs/v71_peru2_probs/",
    "/data/favyenb/rslearn_amazon_conservation_closetime/outputs/v72_peru2_probs/",
]
out_dir = "/data/favyenb/rslearn_amazon_conservation_closetime/windows/peru2/"
annotations = []
for fname in os.listdir(input_dirs[0]):
    probs0 = np.load(os.path.join(input_dirs[0], fname))
    probs1 = np.load(os.path.join(input_dirs[1], fname))

    category0 = probs0.argmax()
    prob0 = probs0[category0]
    probs0[category0] = 0
    category0b = probs0.argmax()
    prob0b = probs0[category0b]

    category1 = probs1.argmax()
    prob1 = probs1[category1]
    probs1[category1] = 0
    category1b = probs1.argmax()
    prob1b = probs1[category1b]

    example_id = fname.split("_amazon_conservation.")[0]
    out_fname = os.path.join(out_dir, example_id, "label.json")
    assert not os.path.exists(out_fname)
    with open(out_fname, "w") as f:
        json.dump(
            {
                "old_label": f"{categories[category0]} ({prob0:.2f}), {categories[category0b]} ({prob0b:.2f})",
                "new_label": f"{categories[category1]} ({prob1:.2f}), {categories[category1b]} ({prob1b:.2f})",
            },
            f,
        )
