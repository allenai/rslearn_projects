"""Visualize the outputs from `model predict`."""

import json
import os
import sys

import numpy as np
import rasterio
import shapely
from PIL import Image
from upath import UPath

if __name__ == "__main__":
    ds_path = UPath(sys.argv[1])
    out_dir = sys.argv[2]

    metadata_fnames = ds_path.glob("windows/default/*/metadata.json")
    window_roots = [fname.parent for fname in metadata_fnames]

    for window_root in window_roots:
        window_name = window_root.name
        with (window_root / "metadata.json").open() as f:
            metadata = json.load(f)
        window_bounds = metadata["bounds"]

        red = rasterio.open(
            window_root / "layers" / "landsat" / "B4" / "geotiff.tif"
        ).read(1)
        green = rasterio.open(
            window_root / "layers" / "landsat" / "B3" / "geotiff.tif"
        ).read(1)
        blue = rasterio.open(
            window_root / "layers" / "landsat" / "B2" / "geotiff.tif"
        ).read(1)
        image = np.stack([red, green, blue], axis=2)
        output_fname = window_root / "layers" / "output" / "data.geojson"
        with output_fname.open() as f:
            for feat in json.load(f)["features"]:
                shp = shapely.geometry.shape(feat["geometry"])
                shp = shapely.transform(
                    shp,
                    lambda coords: (coords - [window_bounds[0], window_bounds[1]]) / 2,
                )
                shp = shapely.clip_by_rect(shp, 0, 0, image.shape[1], image.shape[0])
                bounds = [int(value) for value in shp.bounds]
                image[bounds[1] : bounds[3], bounds[0] : bounds[0] + 2, :] = [255, 0, 0]
                image[bounds[1] : bounds[3], bounds[2] - 2 : bounds[2], :] = [255, 0, 0]
                image[bounds[1] : bounds[1] + 2, bounds[0] : bounds[2], :] = [255, 0, 0]
                image[bounds[3] - 2 : bounds[3], bounds[0] : bounds[2], :] = [255, 0, 0]
        Image.fromarray(image).save(os.path.join(out_dir, f"{window_name}.png"))
