"""Prediction pipeline for Satlas models."""

import json
import os
import shutil
from datetime import datetime
from enum import Enum
from typing import Any

from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources.copernicus import load_sentinel2_tile_index
from rslearn.dataset import Window
from rslearn.utils.geometry import PixelBounds, Projection
from upath import UPath

from rslp.log_utils import get_logger
from rslp.utils.rslearn import materialize_dataset, run_model_predict

DATASET_CONFIG_FNAME = "data/satlas/{application}/config.json"
MODEL_CONFIG_FNAME = "data/satlas/{application}/config.yaml"
SENTINEL2_LAYER = "sentinel2"
PATCH_SIZE = 2048

# Add padding to the time range specified by the user for prediction since some
# applications use images from up to this many days outside of that time range.
RTREE_TIME_PAD_DAYS = 30

# Layers not to use when seeing which patches are valid.
VALIDITY_EXCLUDE_LAYERS = ["mask", "output", "label"]

logger = get_logger(__name__)


class Application(Enum):
    """Specifies the various Satlas applications."""

    SOLAR_FARM = "solar_farm"
    WIND_TURBINE = "wind_turbine"
    MARINE_INFRA = "marine_infra"
    TREE_COVER = "tree_cover"


APP_IS_RASTER = {
    Application.SOLAR_FARM: True,
    Application.WIND_TURBINE: False,
    Application.MARINE_INFRA: False,
    Application.SOLAR_FARM: True,
}


def get_output_fname(
    application: Application, out_path: str, projection: Projection, bounds: PixelBounds
) -> UPath:
    """Get output filename to use for this application and task.

    Args:
        application: the application.
        out_path: the output path.
        projection: the projection of this task.
        bounds: the bounds of this task.

    Returns:
        the output filename.
    """
    if APP_IS_RASTER[application]:
        out_fname = (
            UPath(out_path) / f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}.tif"
        )
    else:
        out_fname = (
            UPath(out_path) / f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}.geojson"
        )
    return out_fname


def predict_pipeline(
    application: Application,
    projection_json: str,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    out_path: str,
    scratch_path: str,
) -> None:
    """Compute outputs of a Satlas model on this tile.

    The tile is one part of a UTM zone.

    Args:
        application: the application for which to compute outputs.
        projection_json: JSON-encoded projection, normally a UTM zone with 10 m/pixel
            resolution.
        bounds: pixel coordinates within the projection on which to compute outputs.
        time_range: time range to apply model on.
        out_path: directory to write the outputs. It will either be a GeoTIFF or
            GeoJSON, named based on the bounds.
        scratch_path: where to store the dataset.
    """
    dataset_config_fname = DATASET_CONFIG_FNAME.format(application=application.value)
    model_config_fname = MODEL_CONFIG_FNAME.format(application=application.value)

    # Check if the output was already computed.
    projection = Projection.deserialize(json.loads(projection_json))
    out_fname = get_output_fname(application, out_path, projection, bounds)
    if out_fname.exists():
        logger.info(f"output file {out_fname} already exists")
        return

    # Initialize an rslearn dataset.
    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True)
    with open(dataset_config_fname) as f:
        ds_cfg = json.load(f)

    # Set the time range to use for the rtree.
    # And also make sure the rtree will be cached based on the out_path.
    index_cache_dir = ds_path / "index_cache_dir"
    index_cache_dir.mkdir()
    image_layer_names = []
    for layer_name, layer_cfg in ds_cfg["layers"].items():
        if "data_source" not in layer_cfg:
            continue
        layer_source_cfg = layer_cfg["data_source"]
        if not layer_source_cfg["name"].endswith("gcp_public_data.Sentinel2"):
            continue
        layer_source_cfg["index_cache_dir"] = str(index_cache_dir)
        # layer_source_cfg["rtree_cache_dir"] = str(UPath(out_path) / "index")
        # layer_source_cfg["use_rtree_index"] = True
        # layer_source_cfg["rtree_time_range"] = [
        #    (time_range[0] - timedelta(days=RTREE_TIME_PAD_DAYS)).isoformat(),
        #    (time_range[1] + timedelta(days=RTREE_TIME_PAD_DAYS)).isoformat(),
        # ]
        image_layer_names.append(layer_name)

    with (ds_path / "config.json").open("w") as f:
        json.dump(ds_cfg, f)

    # Create windows corresponding to the specified projection and bounds.
    # Each window is PATCH_SIZE x PATCH_SIZE, we create multiple of smaller patch size
    # than the bounds instead of one big window for better parallelism and memory
    # usage.
    # It also helps with creating the mosaic -- depending on the dataset configuration,
    # if there are portions of large window that are not fully covered by scenes, then
    # only one mosaic layer would be created. (TODO: actually that seems like an issue
    # with the match_candidate_items_to_window logic. Maybe we should build all the
    # mosaics simultaneously instead of discarding scenes that don't match with the
    # current mosaic.)
    # Note that bounds must be multiple of patch size.
    for value in bounds:
        assert value % PATCH_SIZE == 0
    group = "predict"
    tile_to_window = {}
    for tile_col in range(bounds[0] // PATCH_SIZE, bounds[2] // PATCH_SIZE):
        for tile_row in range(bounds[1] // PATCH_SIZE, bounds[3] // PATCH_SIZE):
            window_name = f"{tile_col}_{tile_row}"
            window_bounds = (
                tile_col * PATCH_SIZE,
                tile_row * PATCH_SIZE,
                (tile_col + 1) * PATCH_SIZE,
                (tile_row + 1) * PATCH_SIZE,
            )
            window_path = ds_path / "windows" / group / window_name
            window = Window(
                path=window_path,
                group=group,
                name=window_name,
                projection=projection,
                bounds=window_bounds,
                time_range=time_range,
            )

            # Skip if the window is too close to 0 longitude.
            # Or if it crosses it.
            epsilon = 1e-4
            wgs84_geom = window.get_geometry().to_projection(WGS84_PROJECTION)
            wgs84_bounds = wgs84_geom.shp.bounds
            if wgs84_bounds[0] <= -180 + epsilon or wgs84_bounds[2] >= 180 - epsilon:
                logger.debug(
                    "skipping window at column %d row %d because it is out of bounds (wgs84_bounds=%s)",
                    tile_col,
                    tile_row,
                    wgs84_bounds,
                )
                continue
            if wgs84_bounds[0] < -90 and wgs84_bounds[2] > 90:
                logger.debug(
                    "skipping window at column %d row %d because it seems to cross 0 longitude (wgs84_bounds=%s)",
                    tile_col,
                    tile_row,
                    wgs84_bounds,
                )
                continue

            window.save()
            tile_to_window[(tile_col, tile_row)] = window

    # Before preparing, cache the Sentinel-2 tile index.
    # This way it is only downloaded once here instead of many times during prepare.
    # We could set use_initial_prepare_job=True in materialize_dataset call, but then
    # it could take a minute or more longer than needed.
    load_sentinel2_tile_index(index_cache_dir)

    # Populate the windows.
    logger.info("materialize dataset")
    materialize_dataset(ds_path, group=group, prepare_workers=128)

    # Run the model, only if at least one window has some data.
    completed_fnames = ds_path.glob(
        f"windows/{group}/*/layers/{image_layer_names[0]}/completed"
    )
    if len(list(completed_fnames)) == 0:
        logger.info("skipping prediction since no windows seem to have data")
    else:
        run_model_predict(model_config_fname, ds_path)

    if APP_IS_RASTER[application]:
        """src_fname = window_path / "layers" / "output" / "output" / "geotiff.tif"

        with src_fname.open("rb") as src:
            with out_fname.open("wb") as dst:
                shutil.copyfileobj(src, dst)"""
        raise NotImplementedError

    else:
        # Merge the features across the windows.
        # Here we also add valid patches attribute indicating which windows (patches)
        # were non-zero. This is used to distinguish a point not being detected because
        # it wasn't there vs not being detected just because there was no image
        # available there.
        fc = None
        valid_patches = []
        for window in tile_to_window.values():
            window_output_fname = window.path / "layers" / "output" / "data.geojson"

            if not window_output_fname.exists():
                continue

            with window_output_fname.open() as f:
                cur_fc = json.load(f)

            if fc is None:
                fc = cur_fc
            else:
                fc["features"].extend(cur_fc["features"])

            valid_patches.append(
                (window.bounds[0] // PATCH_SIZE, window.bounds[1] // PATCH_SIZE)
            )

        if fc is None:
            # So there was no image here.
            # We still want to write an empty GeoJSON so the job is marked completed.
            fc = {
                "type": "FeatureCollection",
                "features": [],
            }

        """
        # Add a list specifying which patches are valid vs invalid to the GeoJSON.
        # Valid means that none of the input layers are completely zero at the patch.
        # This is so that when we smooth the predictions over time, we can distinguish
        # a point not being detected because it wasn't there vs not being detected just
        # because there was no image available there.
        check_images = window_path.glob("layers/*/B02_B03_B04_B08/geotiff.tif")
        valid_patches = set()
        for check_image in check_images:
            path_parts = check_image.path.split("/")
            if path_parts[-3] in VALIDITY_EXCLUDE_LAYERS:
                continue

            with check_image.open("rb") as f:
                with rasterio.open(f) as raster:
                    valid_mask = raster.read().max(axis=0) > 0

            for tile_col in range(bounds[0] // PATCH_SIZE, bounds[2] // PATCH_SIZE):
                for tile_row in range(bounds[1] // PATCH_SIZE, bounds[3] // PATCH_SIZE):
                    cur_patch_id = (tile_col, tile_row)
                    cur_offset = (tile_col * PATCH_SIZE, tile_row * PATCH_SIZE)

                    if cur_patch_id in valid_patches:
                        continue

                    # Read from the window that contains this patch.
                    window = tile_to_window[cur_patch_id]


                    patch_valid = np.zeros((VALIDITY_PATCH_SIZE, VALIDITY_PATCH_SIZE))
                    copy_spatial_array(valid_mask, patch_valid, bounds[0:2], cur_offset)
                    if valid_mask.max() is False:
                        continue

                    valid_patches.add(cur_patch_id)
        """

        if "properties" not in fc:
            fc["properties"] = {}
        fc["properties"]["valid_patches"] = {
            str(projection.crs): list(valid_patches),
        }

        # The object detector predicts bounding boxes but we want to make all features
        # just points.
        for feat in fc["features"]:
            assert feat["geometry"]["type"] == "Polygon"
            coords = feat["geometry"]["coordinates"][0]
            xs = [coord[0] for coord in coords]
            ys = [coord[1] for coord in coords]
            feat["geometry"] = {
                "type": "Point",
                "coordinates": [
                    (min(xs) + max(xs)) / 2,
                    (min(ys) + max(ys)) / 2,
                ],
            }

        with out_fname.open("w") as f:
            json.dump(fc, f)


class PredictTaskArgs:
    """Represents one prediction task among a set that shares application and paths."""

    def __init__(
        self,
        projection_json: dict[str, Any],
        bounds: PixelBounds,
        time_range: tuple[datetime, datetime],
    ):
        """Create a new PredictTaskArgs.

        Args:
            projection_json: serialized projection.
            bounds: the bounds of this task.
            time_range: the time range of this task.
        """
        self.projection_json = projection_json
        self.bounds = bounds
        self.time_range = time_range

    def serialize(self) -> dict[str, Any]:
        """Serialize the task to a dictionary.

        Returns:
            JSON-encodable dictionary.
        """
        return dict(
            projection_json=self.projection_json,
            bounds=json.dumps(self.bounds),
            time_range=json.dumps(
                (self.time_range[0].isoformat(), self.time_range[1].isoformat())
            ),
        )


def predict_multi(
    application: Application,
    out_path: str,
    scratch_path: str,
    tasks: list[PredictTaskArgs],
) -> None:
    """Run multiple prediction tasks.

    Args:
        application: the application.
        out_path: directory to write outputs.
        scratch_path: local directory to use for scratch space.
        tasks: list of tasks to execute.
    """
    if os.path.exists(scratch_path):
        shutil.rmtree(scratch_path)

    for task in tasks:
        predict_pipeline(
            application=application,
            projection_json=json.dumps(task.projection_json),
            bounds=task.bounds,
            time_range=task.time_range,
            out_path=out_path,
            scratch_path=scratch_path,
        )
        if os.path.exists(scratch_path):
            shutil.rmtree(scratch_path)
