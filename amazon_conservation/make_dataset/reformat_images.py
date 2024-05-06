import glob
import multiprocessing
import os

import numpy as np
from PIL import Image
import shutil
import tqdm

in_dir = "/data/favyenb/rslearn_amazon_conservation/windows/peru3/"
out_dir = "/data/favyenb/multisat/labels/amazon_conservation_peru3/amazon_conservation/"
num_outs = 3
min_choices = 5

def handle_example(example_id):
    cur_in_dir = os.path.join(in_dir, example_id)
    cur_out_dir = os.path.join(out_dir, example_id)

    # Find best pre and post images.
    image_lists = {"pre": [], "post": []}
    options = glob.glob(os.path.join(cur_in_dir, "layers/*/R_G_B/image.png"))
    for fname in options:
        # "pre" or "post"
        k = fname.split("/")[-3].split("_")[0]
        im = np.array(Image.open(fname))
        image_lists[k].append(im)
    best_images = {}
    for k, image_list in image_lists.items():
        if len(image_list) < min_choices:
            return
        image_list.sort(key=lambda im: np.count_nonzero((im.max(axis=2) == 0) | (im.min(axis=2) > 230)))
        best_images[k] = image_list[0:num_outs]

    # Write images.
    for idx in range(num_outs):
        cur_img_dir = os.path.join(cur_out_dir, "images", f"image_{idx}")
        os.makedirs(cur_img_dir)
        Image.fromarray(best_images["pre"][idx]).save(os.path.join(cur_img_dir, "pre.png"))
        Image.fromarray(best_images["post"][idx]).save(os.path.join(cur_img_dir, "post.png"))
        shutil.copyfile(
            os.path.join(cur_in_dir, "mask.png"),
            os.path.join(cur_img_dir, "mask.png"),
        )

    # Create gt.json.
    with open(os.path.join(cur_out_dir, "gt.json"), "w") as f:
        f.write("0")

example_ids = os.listdir(in_dir)
p = multiprocessing.Pool(64)
outputs = p.imap_unordered(handle_example, example_ids)
for _ in tqdm.tqdm(outputs, total=len(example_ids)):
    pass
p.close()
