"""Add positive and negative examples from 20240811 annotation.

This was annotation done on false positive detections in the train set. Some of these
false positives are actually correct (so the ground truth in the train set was just
wrong), but most are wrong.

This does mean the detector is not fully overfitting to the train set, which is typical
for detection/segmentation tasks.

Note that the _raw.png crops here are already 256x256 since they were taken from the
rslearn dataset output by minderoo_predict.py, so no additional cropping is needed
unlike the previous
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import shapely
import tqdm
from PIL import Image
from rasterio.crs import CRS
from rslearn.dataset import Window
from rslearn.utils import Feature, LocalFileAPI, Projection, STGeometry
from rslearn.utils.raster_format import SingleImageRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat

if __name__ == "__main__":
    json_fname = (
        "/multisat/datasets/dvim/train_outputs/2024-07-30_0.5/20240811_annotations.json"
    )
    big_crop_dir = (
        "/multisat/datasets/dvim/train_outputs/2024-07-30_0.5/annotate_crops/"
    )
    out_dir = "/multisat/datasets/dvim/rslearn_classify/"

    # We are just using fake projection for training this model.
    projection = Projection(CRS.from_epsg(3857), 1, -1)
    time_range = (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    raster_format = SingleImageRasterFormat()
    vector_format = GeojsonVectorFormat()

    with open(json_fname) as f:
        annotations = json.load(f)

    for example_id, label in tqdm.tqdm(annotations.items()):
        if label not in ["positive", "negative"]:
            continue

        if label == "positive":
            weight = 1
            group = "20240811_positives"
        else:
            weight = 5
            group = "20240811_negatives"

        image_fname, crop_id = example_id.split("/")
        prefix = image_fname.split(".tif")[0]

        region_name = "_".join(prefix.split("_")[0:3])
        is_val = hashlib.sha256(region_name.encode()).hexdigest()[0] in ["0", "1"]
        if is_val:
            split = "val"
        else:
            split = "train"

        crop_fname = os.path.join(big_crop_dir, image_fname, f"{crop_id}_raw.png")
        image = np.array(
            Image.open(os.path.join(big_crop_dir, image_fname, crop_fname))
        )
        image = image.transpose(2, 0, 1)
        window_bounds = [0, 0, image.shape[2], image.shape[1]]

        window_name = f"20240811_{prefix}_{crop_id}"
        window_root = os.path.join(out_dir, "windows", group, window_name)
        os.makedirs(window_root, exist_ok=True)
        window = Window(
            file_api=LocalFileAPI(window_root),
            group=group,
            name=window_name,
            projection=projection,
            bounds=window_bounds,
            time_range=time_range,
            options={"split": split, "weight": weight},
        )
        window.save()

        # Write the GeoTIFF.
        layer_file_api = window.file_api.get_folder("layers", "maxar")
        raster_format.encode_raster(
            layer_file_api.get_folder("R_G_B"), projection, window_bounds, image
        )
        with layer_file_api.open("completed", "w") as f:
            pass

        # Write the label.
        layer_file_api = window.file_api.get_folder("layers", "label")
        feature = Feature(
            STGeometry(projection, shapely.Point(0, 0), None), {"label": label}
        )
        vector_format.encode_vector(layer_file_api, projection, [feature])
        with layer_file_api.open("completed", "w") as f:
            pass
