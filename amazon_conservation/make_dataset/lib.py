import io
import os

import multisat.util
import numpy as np
import rasterio.features
import requests
import skimage.io

sentinel2_url = "https://se-tile-api.allen.ai/image_mosaic/sentinel2/[LABEL]/tci/[ZOOM]/[COL]/[ROW].png"
chip_size = 512


def get_sentinel2_callback(label):
    def callback(tile):
        cur_url = sentinel2_url
        cur_url = cur_url.replace("[LABEL]", label)
        cur_url = cur_url.replace("[ZOOM]", "13")
        cur_url = cur_url.replace("[COL]", str(tile[0]))
        cur_url = cur_url.replace("[ROW]", str(tile[1]))

        response = requests.get(cur_url)
        if response.status_code != 200:
            print(f"got status_code={response.status_code} url={cur_url}")
            if response.status_code == 404:
                return np.zeros((chip_size, chip_size, 3))
            raise Exception(f"bad status code {response.status_code}")

        buf = io.BytesIO(response.content)
        im = skimage.io.imread(buf)
        return im

    return callback


def fetch_images(job):
    cur_img_dir, wanted_labels, start, crop_size = job
    for band_name, label in wanted_labels:
        out_fname = os.path.join(cur_img_dir, f"{band_name}.png")
        if os.path.exists(out_fname):
            continue
        callback = get_sentinel2_callback(label)
        im = multisat.util.load_window_callback(
            callback, start[0], start[1], crop_size, crop_size
        )
        skimage.io.imsave(out_fname, im, check_contrast=False)


def fetch_image_group(job):
    cur_img_dir, label_options, needed, prefix, start, crop_size, shp = job
    images = []
    for label in label_options:
        callback = get_sentinel2_callback(label)
        im = multisat.util.load_window_callback(
            callback, start[0], start[1], crop_size, crop_size
        )
        images.append(im)

    # Sort images by badness = missing*5 + cloudy.
    images.sort(
        key=lambda im: np.count_nonzero(im.max(axis=2) <= 1) * 5
        + np.count_nonzero(im.min(axis=2) >= 230)
    )
    selected = images[0:needed]

    # Include mask image if shp is set.
    if shp:
        mask_im = rasterio.features.rasterize(
            [(shp, 255)],
            out_shape=(crop_size, crop_size),
        )

    for idx, im in enumerate(selected):
        out_dir = os.path.join(cur_img_dir, f"image_{idx}")
        os.makedirs(out_dir, exist_ok=True)
        out_fname = os.path.join(out_dir, f"{prefix}.png")
        skimage.io.imsave(out_fname, im, check_contrast=False)

        if shp:
            skimage.io.imsave(
                os.path.join(out_dir, "mask.png"),
                mask_im,
            )
