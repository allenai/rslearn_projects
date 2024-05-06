"""
Reformat from rslearn to multisat format.
"""
import multiprocessing
import numpy as np
import os
from PIL import Image
import shutil
import tqdm

in_dir = "/data/favyenb/rslearn_landsat/windows/labels_utm/"
out_dir = "/data/favyenb/rslearn_landsat/labels_utm_as_multisat/landsat_vessels/"
bands = ["b2", "b3", "b4", "b5", "b6", "b7", "b8"]

def reformat(example_id):
    cur_in_dir = os.path.join(in_dir, example_id)
    if not os.path.exists(os.path.join(cur_in_dir, "layers", "landsat", bands[0].upper(), "image.png")):
        print(f"warning: missing {example_id}")
        return

    cur_out_dir = os.path.join(out_dir, example_id)
    cur_img_dir = os.path.join(cur_out_dir, "images", "x")
    os.makedirs(cur_img_dir, exist_ok=True)

    # Copy the vessel labels.
    shutil.copyfile(
        os.path.join(cur_in_dir, "gt.json"),
        os.path.join(cur_out_dir, "gt.json"),
    )

    # Mask each image to erase parts that were not in the WebMercator image.
    mask = np.array(Image.open(os.path.join(cur_in_dir, "mask.png")))
    mask[mask == 255] = 1
    for band in bands:
        in_img_fname = os.path.join(cur_in_dir, "layers", "landsat", band.upper(), "image.png")
        out_img_fname = os.path.join(cur_img_dir, f"{band}.png")
        im = np.array(Image.open(in_img_fname))

        if band != "b8":
            im = im.repeat(axis=0, repeats=2).repeat(axis=1, repeats=2)
        im *= mask

        Image.fromarray(im).save(out_img_fname)

example_ids = os.listdir(in_dir)
p = multiprocessing.Pool(64)
outputs = p.imap_unordered(reformat, example_ids)
for _ in tqdm.tqdm(outputs, total=len(example_ids)):
    pass
p.close()
