"""Delete non-matching images from items.json.

We had a specific image ID in siv, but didn't have a good way to specify that in
rslearn. So it could end up with a mosaic that covers areas that were not covered in
the original image.

So we set items.json to be populated with mosaic (multiple images) and we delete the
ones that don't match, hopefully the one that matchs was found.

We can't create items.json directly because there might be some details like blob path
that are hard to reproduce.
"""

import json
import os

ds_dir = "/weka/dfive-default/rslearn-eai/datasets/sentinel1_vessels/dataset_v1/20250521/"

for group in os.listdir(os.path.join(ds_dir, "windows")):
    if group != "nov-2021-point-val":
        continue
    group_dir = os.path.join(ds_dir, "windows", group)
    for window_id in os.listdir(group_dir):
        window_dir = os.path.join(group_dir, window_id)
        with open(os.path.join(window_dir, "image_name_from_siv.txt")) as f:
            image_name = f.read().strip()
        prefix = "_".join(image_name.split("_")[0:6])
        with open(os.path.join(window_dir, "items.json")) as f:
            item_data = json.load(f)
        assert len(item_data) == 1
        layer = item_data[0]
        good_group = None
        for group in layer["serialized_item_groups"]:
            assert len(group) == 1
            if not group[0]["name"].startswith(prefix):
                continue
            good_group = group
            break
        if not good_group:
            print(f"warning: no image matching prefix for {window_id}", prefix, layer)
            continue
        layer["serialized_item_groups"] = [good_group]
        with open(os.path.join(window_dir, "items.json"), "w") as f:
            json.dump(item_data, f)
