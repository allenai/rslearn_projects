import json
import multiprocessing
import os
import shutil

import tqdm
from PIL import Image

in_dir = "/data/favyenb/rslearn_amazon_conservation_closetime/windows/brazil/"
out_dir = "/data/favyenb/multisat/labels/amazon_conservation_new/"
num_outs = 3
min_choices = 5


def handle_example(example_id):
    cur_in_dir = os.path.join(in_dir, example_id)
    cur_out_dir = os.path.join(out_dir, example_id)

    label_fname = os.path.join(cur_in_dir, "label.json")
    if os.path.exists(label_fname):
        with open(label_fname) as f:
            category = json.load(f)["new_label"]
    else:
        category = "unknown"
    if category in [
        "agriculture-generic",
        "agriculture-small",
        "agriculture-rice",
        "agriculture-mennonite",
        "coca",
    ]:
        category = "agriculture"
    if category == "flood":
        category = "river"
    if category not in [
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
    ]:
        return

    # Use images selected by "select_images.py".
    with open(os.path.join(cur_in_dir, "good_images.json")) as f:
        best_images = json.load(f)

    # Write images.
    for idx in range(num_outs):
        cur_img_dir = os.path.join(cur_out_dir, "images", f"image_{idx}")
        os.makedirs(cur_img_dir, exist_ok=True)
        Image.open(os.path.join(cur_in_dir, best_images["pre"][idx][0])).save(
            os.path.join(cur_img_dir, "pre.png")
        )
        Image.open(os.path.join(cur_in_dir, best_images["post"][idx][0])).save(
            os.path.join(cur_img_dir, "post.png")
        )
        shutil.copyfile(
            os.path.join(cur_in_dir, "mask.png"),
            os.path.join(cur_img_dir, "mask.png"),
        )

    # Create gt.txt.
    with open(os.path.join(cur_out_dir, "gt.txt"), "w") as f:
        f.write(category)


example_ids = os.listdir(in_dir)
p = multiprocessing.Pool(64)
outputs = p.imap_unordered(handle_example, example_ids)
for _ in tqdm.tqdm(outputs, total=len(example_ids)):
    pass
p.close()
