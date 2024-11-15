"""Visualize the outputs from `model predict`."""

import json
import os

import numpy as np
import shapely
from PIL import Image
from upath import UPath

if __name__ == "__main__":
    # python script.py /path/to/dataset /path/to/output
    ds_path = UPath(
        "gs://rslearn-eai/datasets/landsat_vessel_detection/detector/dataset_20240924/"
    )
    out_dir = "../visualizations/"

    metadata_fnames = ds_path.glob(
        "windows/labels_utm/*/metadata.json"
    )  # original default
    window_roots = [fname.parent for fname in metadata_fnames]

    for window_root in window_roots:
        window_name = window_root.name
        with (window_root / "metadata.json").open() as f:
            metadata = json.load(f)
            # Check option, split
            if metadata["options"]["split"] == "train":
                continue
        window_bounds = metadata["bounds"]

        # red = rasterio.open(
        #     window_root / "layers" / "landsat" / "B4" / "geotiff.tif"
        # ).read(1)
        # green = rasterio.open(
        #     window_root / "layers" / "landsat" / "B3" / "geotiff.tif"
        # ).read(1)
        # blue = rasterio.open(
        #     window_root / "layers" / "landsat" / "B2" / "geotiff.tif"
        # ).read(1)
        # load the image.png file instead
        # each image.png is just one band

        try:
            red_path = window_root / "layers" / "landsat" / "B4" / "image.png"
            with red_path.open("rb") as f:
                red = Image.open(f)
                red.load()
            green_path = window_root / "layers" / "landsat" / "B3" / "image.png"
            with green_path.open("rb") as f:
                green = Image.open(f)
                green.load()
            blue_path = window_root / "layers" / "landsat" / "B2" / "image.png"
            with blue_path.open("rb") as f:
                blue = Image.open(f)
                blue.load()
        except Exception as e:
            print(f"Error loading images for {window_root}: {e}")
            continue

        image = np.stack([red, green, blue], axis=2)
        print(window_root)
        output_fname = window_root / "layers" / "label" / "data.geojson"

        with output_fname.open("r") as f:
            for feat in json.load(f)["features"]:
                shp = shapely.geometry.shape(feat["geometry"])
                shp = shapely.transform(
                    shp,
                    lambda coords: (coords - [window_bounds[0], window_bounds[1]]) / 2,
                )
                shp = shapely.clip_by_rect(shp, 0, 0, image.shape[1], image.shape[0])
                bounds = [int(value) for value in shp.bounds]
                image[bounds[1] : bounds[3], bounds[0] : bounds[0] + 1, :] = [255, 0, 0]
                image[bounds[1] : bounds[3], bounds[2] - 1 : bounds[2], :] = [255, 0, 0]
                image[bounds[1] : bounds[1] + 1, bounds[0] : bounds[2], :] = [255, 0, 0]
                image[bounds[3] - 1 : bounds[3], bounds[0] : bounds[2], :] = [255, 0, 0]

        Image.fromarray(image).save(os.path.join(out_dir, f"{window_name}.png"))
