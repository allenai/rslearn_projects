"""Add negative examples from 20240808_annotation in other_satlas_projects/dvim/."""

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
    json_fnames = [
        "/multisat/datasets/dvim/outputs/202408_08_annotations.json",
        "/multisat/datasets/dvim/outputs/202408_08_annotations_2.json",
    ]
    big_crop_dir = "/multisat/datasets/dvim/outputs/big_crops_centered/"
    out_dir = "/multisat/datasets/dvim/rslearn_classify/"
    group = "20240808"

    # We are just using fake projection for training this model.
    projection = Projection(CRS.from_epsg(3857), 1, -1)
    time_range = (
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    raster_format = SingleImageRasterFormat()
    vector_format = GeojsonVectorFormat()

    for json_fname in json_fnames:
        print(json_fname)
        with open(json_fname) as f:
            annotations = json.load(f)

        for example_id, label in tqdm.tqdm(annotations.items()):
            if label not in ["positive", "negative"]:
                continue
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
            image = image[128:384, 128:384, :].transpose(2, 0, 1)
            window_bounds = [0, 0, image.shape[2], image.shape[1]]

            window_name = f"20240808_{prefix}_{crop_id}"
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
