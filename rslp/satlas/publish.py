"""Publish Satlas outputs."""

import json
import os
import shutil
import subprocess  # nosec
import tempfile
import zipfile
from typing import Any

import boto3
import boto3.s3
from upath import UPath

from rslp.log_utils import get_logger
from rslp.satlas.bkt import make_bkt

from .predict_pipeline import Application

logger = get_logger(__name__)

# Number of timesteps to re-publish.
# Smoothing for points changes all of the outputs but we only upload outputs for this
# many of the most recent timesteps.
NUM_RECOMPUTE = 6

# Name on Cloudflare R2 for each application.
APP_NAME_ON_R2 = {
    Application.MARINE_INFRA: "marine",
}

APP_TIPPECANOE_LAYERS = {
    Application.MARINE_INFRA: "marine",
}

SHP_EXTENSIONS = [
    ".shp",
    ".dbf",
    ".prj",
    ".shx",
]

BKT_TILE_PATH = "output_mosaic/"


def get_cloudflare_r2_bucket() -> Any:
    """Returns the Cloudflare R2 bucket where outputs are published."""
    s3 = boto3.resource(
        "s3",
        endpoint_url=os.environ["SATLAS_R2_ENDPOINT"],
        aws_access_key_id=os.environ["SATLAS_R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["SATLAS_R2_SECRET_ACCESS_KEY"],
    )
    bucket = s3.Bucket(os.environ["SATLAS_R2_BUCKET_NAME"])
    return bucket


def make_shapefile_zip(fname: str) -> str:
    """Create zip file of the shapefile and its supporting files.

    If filename is "x" (for x.shp and supporting files) then output is "x.shp.zip".

    Args:
        fname: fname without .shp extension

    Returns:
        the local filename of the resulting zip file.
    """
    zip_fname = fname + ".shp.zip"
    basename = os.path.basename(fname)
    with zipfile.ZipFile(zip_fname, "w") as z:
        for ext in SHP_EXTENSIONS:
            z.write(fname + ext, arcname=basename + ext)
    return zip_fname


def update_index(bucket: Any, prefix: str) -> None:
    """Update index file on Cloudflare R2.

    The index file just has list of filenames, last modified time, and md5.

    There is one index for each application folder.

    Args:
        bucket: the Cloudflare R2 bucket.
        prefix: the folder's prefix in the bucket.
    """
    index_lines = []
    for obj in bucket.objects.filter(Prefix=prefix):
        if obj.key.endswith("/index.txt"):
            continue
        line = "{},{},{}".format(
            obj.key, obj.last_modified, obj.e_tag.split("-")[0].replace('"', "")
        )
        index_lines.append(line)
    index_lines.append("")
    index_data = "\n".join(index_lines)
    bucket.put_object(
        Body=index_data.encode(),
        Key=prefix + "index.txt",
    )


def publish_points(
    application: Application,
    smoothed_path: str,
    version: str,
    workers: int = 32,
) -> None:
    """Publish Satlas point outputs.

    The points are added to two locations: GeoJSONs are added to Cloudflare R2, while
    tippecanoe is used to generate vector tiles that are uploaded to GCS for use by the
    satlas.allen.ai website.

    Args:
        application: the application.
        smoothed_path: folder containing smoothed predictions (including
            history.geojson file).
        version: current model version for use to distinguish different outputs on GCS.
        workers: number of worker processes.
    """
    smoothed_upath = UPath(smoothed_path)

    # First upload files to R2.
    bucket = get_cloudflare_r2_bucket()
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Upload history.
        logger.info("upload history")
        local_hist_fname = os.path.join(tmp_dir, "history.geojson")
        with (smoothed_upath / "history.geojson").open("rb") as src:
            with open(local_hist_fname, "wb") as dst:
                shutil.copyfileobj(src, dst)
        app_name_on_r2 = APP_NAME_ON_R2[application]
        bucket.upload_file(local_hist_fname, f"outputs/{app_name_on_r2}/marine.geojson")

        # Upload the latest outputs too.
        available_fnames: list[UPath] = []
        for fname in smoothed_upath.iterdir():
            if fname.name == "history.geojson":
                continue
            available_fnames.append(fname)
        available_fnames.sort(key=lambda fname: fname.name)
        for fname in available_fnames[-NUM_RECOMPUTE:]:
            logger.info("upload %s", str(fname))
            local_geojson_fname = os.path.join(tmp_dir, "data.geojson")

            with fname.open("rb") as src:
                with open(local_geojson_fname, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            fname_prefix = fname.name.split(".")[0]

            bucket.upload_file(
                local_geojson_fname,
                f"outputs/{app_name_on_r2}/{fname_prefix}.geojson",
            )
            if fname == available_fnames[-1]:
                bucket.upload_file(
                    local_geojson_fname,
                    f"outputs/{app_name_on_r2}/latest.geojson",
                )

        update_index(bucket, f"outputs/{app_name_on_r2}/")

    # Generate the tippecanoe tiles.
    # We set tippecanoe layer via property of each feature.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tippecanoe_layer = APP_TIPPECANOE_LAYERS[application]
        with (smoothed_upath / "history.geojson").open("rb") as f:
            fc = json.load(f)
        for feat in fc["features"]:
            feat["tippecanoe"] = {"layer": tippecanoe_layer}
        local_geojson_fname = os.path.join(tmp_dir, "history.geojson")
        with open(local_geojson_fname, "w") as f:
            json.dump(fc, f)

        local_tile_dir = os.path.join(tmp_dir, "tiles")
        logger.info("run tippecanoe on history in local tmp dir %s", local_tile_dir)
        subprocess.check_call(
            [
                "tippecanoe",
                "-z13",
                "-r1",
                "--cluster-densest-as-needed",
                "--no-tile-compression",
                "-e",
                local_tile_dir,
                local_geojson_fname,
            ]
        )  # nosec

        tile_dst_path = f"{BKT_TILE_PATH}{version}/history/{{zoom}}/0/0.bkt"
        logger.info("make bkt at %s", tile_dst_path)
        make_bkt(src_dir=local_tile_dir, dst_path=tile_dst_path)
