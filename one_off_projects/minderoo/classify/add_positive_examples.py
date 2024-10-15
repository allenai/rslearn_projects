"""Add positive examples from the training data."""

import json
import multiprocessing
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

data_root = "/data/favyenb/dvim/2024-07-31/default/orig/"
out_dir = "/data/favyenb/dvim/rslearn_classify/"
group = "positives"
crop_size = 256

# We are just using fake projection for training this model.
projection = Projection(CRS.from_epsg(3857), 1, -1)
time_range = (
    datetime(2024, 1, 1, tzinfo=timezone.utc),
    datetime(2024, 1, 2, tzinfo=timezone.utc),
)
raster_format = SingleImageRasterFormat()
vector_format = GeojsonVectorFormat()


def process(job):
    data_root, split, prefix = job

    # Read annotations. We can skip reading image if there are none.
    with open(os.path.join(data_root, split, prefix + ".json")) as f:
        vessel_labels = json.load(f)
    if len(vessel_labels) == 0:
        return

    image = np.array(Image.open(os.path.join(data_root, split, prefix + ".png")))
    image = image.transpose(2, 0, 1)
    # Pad to guarantee each vessel center will be within the image bounds.
    image = np.pad(
        image,
        [(0, 0), (crop_size // 2, crop_size // 2), (crop_size // 2, crop_size // 2)],
    )

    for vessel_points in vessel_labels:
        xs = [point[0] for point in vessel_points]
        ys = [point[1] for point in vessel_points]
        cx = (min(xs) + max(xs)) // 2
        cy = (min(ys) + max(ys)) // 2
        if cx < 0 or cx + crop_size > image.shape[2]:
            continue
        if cy < 0 or cy + crop_size > image.shape[1]:
            continue

        crop = image[:, cy : cy + crop_size, cx : cx + crop_size]
        window_bounds = [0, 0, crop_size, crop_size]

        window_name = f"{prefix}_{cx}_{cy}"
        window_root = os.path.join(out_dir, "windows", group, window_name)
        os.makedirs(window_root, exist_ok=True)
        window = Window(
            file_api=LocalFileAPI(window_root),
            group=group,
            name=window_name,
            projection=projection,
            bounds=window_bounds,
            time_range=time_range,
            options={"split": split, "weight": 1},
        )
        window.save()

        # Write the GeoTIFF.
        layer_file_api = window.file_api.get_folder("layers", "maxar")
        raster_format.encode_raster(
            layer_file_api.get_folder("R_G_B"), projection, window_bounds, crop
        )
        with layer_file_api.open("completed", "w") as f:
            pass

        # Write the label.
        layer_file_api = window.file_api.get_folder("layers", "label")
        feature = Feature(
            STGeometry(projection, shapely.Point(0, 0), None), {"label": "positive"}
        )
        vector_format.encode_vector(layer_file_api, projection, [feature])
        with layer_file_api.open("completed", "w") as f:
            pass


if __name__ == "__main__":
    jobs = []
    for split in os.listdir(data_root):
        fnames = [
            fname
            for fname in os.listdir(os.path.join(data_root, split))
            if fname.endswith(".png")
        ]
        for fname in fnames:
            jobs.append((data_root, split, fname.split(".png")[0]))
    p = multiprocessing.Pool(64)
    outputs = p.imap_unordered(process, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
