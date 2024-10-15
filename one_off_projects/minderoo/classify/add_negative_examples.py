"""Add negative examples from 20240730_annotation in other_satlas_projects/dvim/.

These are from labeling 1000 test images -- we label them either "all detections are
incorrect" or "some correct detections" so we just use the former.
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
    json_fname = "/multisat/datasets/dvim/outputs/annotations.json"
    big_crop_dir = "/multisat/datasets/dvim/outputs/big_crops_centered/"
    out_dir = "/multisat/datasets/dvim/rslearn_classify/"
    group = "negatives"

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

    for image_fname, label in tqdm.tqdm(annotations.items()):
        if label != "fp":
            continue
        prefix = image_fname.split(".tif")[0]

        region_name = "_".join(prefix.split("_")[0:3])
        is_val = hashlib.sha256(region_name.encode()).hexdigest()[0] in ["0", "1"]
        if is_val:
            split = "val"
        else:
            split = "train"

        crop_fnames = [
            fname
            for fname in os.listdir(os.path.join(big_crop_dir, image_fname))
            if fname.endswith("_raw.png")
        ]
        for crop_fname in crop_fnames:
            image = np.array(
                Image.open(os.path.join(big_crop_dir, image_fname, crop_fname))
            )
            image = image[128:384, 128:384, :].transpose(2, 0, 1)
            window_bounds = [0, 0, image.shape[2], image.shape[1]]

            crop_idx = crop_fname.split("_")[0]
            window_name = f"{prefix}_{crop_idx}"
            window_root = os.path.join(out_dir, "windows", group, window_name)
            os.makedirs(window_root, exist_ok=True)
            window = Window(
                file_api=LocalFileAPI(window_root),
                group=group,
                name=window_name,
                projection=projection,
                bounds=window_bounds,
                time_range=time_range,
                options={"split": split, "weight": 5},
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
                STGeometry(projection, shapely.Point(0, 0), None), {"label": "negative"}
            )
            vector_format.encode_vector(layer_file_api, projection, [feature])
            with layer_file_api.open("completed", "w") as f:
                pass
