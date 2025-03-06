"""Produce the tiles that will appear in the web app."""

import json
import multiprocessing
import os
import subprocess  # nosec
import tempfile
from typing import Any

import shapely
import tqdm
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from rslp.forest_loss_driver.const import (
    DEFAULT_TILE_PATH,
    GEOJSON_FNAME,
    GROUP,
    READY_FOR_SERVING_FNAME,
    WINDOWS_FNAME,
)
from rslp.utils.fs import copy_file

from .index_windows import OUTPUT_GEOJSON_SUFFIX

DEFAULT_NUM_WORKERS = 32


def get_geojson_feature(index: int, window_root: UPath) -> dict[str, Any]:
    """Get the GeoJSON feature corresponding to the prediction for this window.

    The feature uses the polygon from the original forest loss event, with a property
    indicating the driver category prediction.

    Args:
        index: the index that this appears in good_windows.json. This will be another
            property on the feature.
        window_root: the window path.
    """
    # Get the polygon from the info.json.
    with (window_root / "info.json").open() as f:
        info = json.load(f)
    shp = shapely.from_wkt(info["wkt"])
    geom_dict = json.loads(shapely.to_geojson(shp))

    # Get the predicted category.
    with (window_root / OUTPUT_GEOJSON_SUFFIX).open() as f:
        output_data = json.load(f)
    category = output_data["features"][0]["properties"]["new_label"]

    properties = dict(
        index=index,
        date=info["date"],
        category=category,
        window_name=window_root.name,
    )

    # We include a tippecanoe dict which tells tippecanoe to put features pertaining to
    # each driver category in a separate layer (named by the category) in the vector
    # tiles.
    return {
        "type": "Feature",
        "properties": properties,
        "geometry": geom_dict,
        "tippecanoe": dict(
            layer=category,
        ),
    }


class MakeTilesArgs:
    """Arguments for make_tiles function."""

    def __init__(
        self,
        ds_root: str,
        workers: int = DEFAULT_NUM_WORKERS,
        tile_path: str = DEFAULT_TILE_PATH,
    ):
        """Arguments for make_tiles.

        Args:
            ds_root: the inference dataset root path.
            workers: number of workers to use.
            tile_path: the base path to store tiles. It will be extended with the
                folder name of the inference dataset.
        """
        self.ds_root = ds_root
        self.workers = workers
        self.tile_path = tile_path

    def get_tile_dir(self) -> UPath:
        """Get the directory to store tiles for this specific dataset."""
        return UPath(self.tile_path) / UPath(self.ds_root).name


def make_tiles(args: MakeTilesArgs) -> None:
    """Produce the vector tiles that will appear in the web app.

    Args:
        args: the MakeTilesArgs parameters.
    """
    ds_path = UPath(args.ds_root)
    dst_dir = args.get_tile_dir()

    # Create the GeoJSON file that we will apply tippecanoe on to create the vector
    # tiles. To do so, we enumerate the windows in good_windows.json, and extract a
    # GeoJSON feature corresponding to each window.
    jobs = []
    with (ds_path / WINDOWS_FNAME).open() as f:
        for index, window_name in enumerate(json.load(f)):
            jobs.append(
                dict(
                    index=index + 1,
                    window_root=ds_path / "windows" / GROUP / window_name,
                )
            )

    p = multiprocessing.Pool(args.workers)
    features = list(
        tqdm.tqdm(star_imap_unordered(p, get_geojson_feature, jobs), total=len(jobs))
    )
    fc = {
        "type": "FeatureCollection",
        "features": features,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_fname = os.path.join(tmp_dir, GEOJSON_FNAME)
        with open(local_fname, "w") as f:
            json.dump(fc, f)

        # Save the GeoJSON with the tiles so it can be downloaded.
        copy_file(UPath(local_fname), dst_dir / GEOJSON_FNAME)

        # Apply tippecanoe to convert the GeoJSON into a set of vector tiles.
        local_tile_dir = os.path.join(tmp_dir, "tiles")
        subprocess.call(
            [
                "tippecanoe",
                # Choose the maximum zoom level automatically.
                "-zg",
                # Write to directory.
                "-e",
                local_tile_dir,
                # Need no compression in the tile files to work with Leaflet.js.
                "--no-tile-compression",
                # Drop the smaller polygons in coarser zoom levels.
                "--drop-smallest-as-needed",
                local_fname,
            ]
        )  # nosec

        # Copy to GCS from which we serve the tiles.
        src_fnames = UPath(local_tile_dir).glob("*/*/*.pbf")
        copy_jobs = []
        for src_fname in src_fnames:
            dst_fname = (
                dst_dir
                / src_fname.parents[1].name
                / src_fname.parents[0].name
                / src_fname.name
            )
            copy_jobs.append(
                dict(
                    src_fname=src_fname,
                    dst_fname=dst_fname,
                )
            )

        outputs = star_imap_unordered(p, copy_file, copy_jobs)
        for _ in tqdm.tqdm(outputs, total=len(copy_jobs)):
            pass

    # Create an extra file to mark the tiles ready to serve from web app.
    (dst_dir / READY_FOR_SERVING_FNAME).touch()

    p.close()
