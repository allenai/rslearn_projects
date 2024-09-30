"""Create WebMercator version of the wind turbine dataset as rslearn dataset.

This is just used to make sure performance is the same when training in rslearn versus
training in multisat.

The projection is not set correctly since it's just for testing.
"""

import argparse
import json
import multiprocessing
import os
import shutil

import shapely
import tqdm
from rasterio.crs import CRS
from rslearn.dataset import Window
from rslearn.utils import Feature, Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

BANDS = ["tci", "b05", "b06", "b07", "b08", "b11", "b12"]

# How many images model will be trained with, so can't have examples with fewer than
# this many images.
REQUIRED_IMAGES = 4


def process_example(
    ds_path: UPath,
    label_dir: str,
    image_dir: str,
    tile: tuple[int, int],
    image_ids: list[str],
    split: str,
):
    projection = Projection(CRS.from_epsg(3857), 10, -10)

    window_name = f"{tile[0]}_{tile[1]}"
    group = "default"
    window_root = ds_path / "windows" / group / window_name
    window = Window(
        path=window_root,
        group=group,
        name=window_name,
        projection=projection,
        bounds=[0, 0, 512, 512],
        time_range=None,
        options={"split": split},
    )
    window.save()

    # Image layers.
    for idx, image_id in enumerate(image_ids):
        if idx == 0:
            layer_name = "sentinel2"
        else:
            layer_name = f"sentinel2.{idx}"

        for band in BANDS:
            src_fname = os.path.join(
                image_dir, image_id, band, f"{tile[0]}_{tile[1]}.png"
            )
            if band == "tci":
                dst_band_name = "R_G_B"
            else:
                dst_band_name = band
            dst_fname = (
                window_root / "layers" / layer_name / dst_band_name / "image.png"
            )
            dst_fname.parent.mkdir(parents=True, exist_ok=True)
            with open(src_fname, "rb") as src:
                with dst_fname.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        (window_root / "layers" / layer_name / "completed").touch()

    # Label layer.
    features = []
    with open(os.path.join(label_dir, fname)) as f:
        for x1, y1, x2, y2, category in json.load(f):
            geom = STGeometry(projection, shapely.box(x1, y1, x2, y2), None)
            props = dict(category=category)
            features.append(Feature(geom, props))
    layer_dir = window_root / "layers" / "label"
    GeojsonVectorFormat().encode_vector(layer_dir, projection, features)
    (layer_dir / "completed").touch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_dir",
        help="The multisat label directory",
        type=str,
        default="/multisat/labels/renewable_infra_point_naip_supervision",
    )
    parser.add_argument(
        "--image_dir",
        help="The multisat image directory",
        type=str,
        default="/multisat/mosaic/turbine_naip_supervision/",
    )
    parser.add_argument(
        "--ds_path",
        help="The path to write output rslearn dataset",
        type=str,
        default="gs://rslearn-eai/datasets/wind_turbine/webmercator_dataset/20240927/",
    )
    parser.add_argument(
        "--train_split",
        help="The JSON file containing train split",
        type=str,
        default="/multisat/mosaic/splits/turbine_naip_supervision/train.json",
    )
    parser.add_argument(
        "--workers",
        help="Number of parallel workers",
        type=int,
        default=32,
    )
    args = parser.parse_args()

    # Create map from tile to image IDs that have it available.
    tile_to_image_ids = {}
    for image_id in os.listdir(args.image_dir):
        for fname in os.listdir(os.path.join(args.image_dir, image_id, BANDS[0])):
            # Make sure the other bands exist.
            bands_exist = True
            for band in BANDS:
                if os.path.exists(os.path.join(args.image_dir, image_id, band, fname)):
                    continue
                bands_exist = False
                break
            if not bands_exist:
                continue

            parts = fname.split(".")[0].split("_")
            tile = (int(parts[0]), int(parts[1]))
            if tile not in tile_to_image_ids:
                tile_to_image_ids[tile] = []
            tile_to_image_ids[tile].append(image_id)

    # Identify tiles in train split.
    train_split = set()
    with open(args.train_split) as f:
        for col, row in json.load(f):
            train_split.add((col, row))

    ds_path = UPath(args.ds_path)

    jobs = []
    for fname in os.listdir(args.label_dir):
        parts = fname.split(".")[0].split("_")
        tile = (int(parts[0]), int(parts[1]))
        image_ids = tile_to_image_ids.get(tile, [])
        if len(image_ids) < REQUIRED_IMAGES:
            continue

        if tile in train_split:
            split = "train"
        else:
            split = "val"

        jobs.append(
            dict(
                ds_path=ds_path,
                label_dir=args.label_dir,
                image_dir=args.image_dir,
                tile=tile,
                image_ids=image_ids,
                split=split,
            )
        )

    p = multiprocessing.Pool(args.workers)
    outputs = star_imap_unordered(p, process_example, jobs)
    for _ in tqdm.tqdm(outputs, total=len(jobs)):
        pass
    p.close()
