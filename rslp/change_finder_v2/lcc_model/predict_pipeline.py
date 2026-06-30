"""Scaled prediction pipeline for the change_finder_v2 LCC model.

This applies the land cover change model over a tile (one part of a UTM zone) by
creating PATCH_SIZE windows, materializing Sentinel-2 imagery from the OlmoEarth
Datasets source, running the model, and polygonizing the resulting change raster
into a per-tile GeoJSON. Optionally the merged 49-band raster is also written.

Unlike the satlas pipeline, there is no rtree index: the OlmoEarth Datasets source
queries its API per window, and all imagery is derived from a single reference
timestamp T via the layer time_offset/duration in config_predict.json.

This module is tile-size-agnostic: it accepts any ``bounds`` whose extents are
multiples of PATCH_SIZE. The fixed 32768x32768 tiling lives only in write_jobs.py.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import rasterio
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import PixelBounds, Projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from rslp.log_utils import get_logger
from rslp.utils.rslearn import (
    ApplyWindowsArgs,
    IngestArgs,
    MaterializeArgs,
    MaterializePipelineArgs,
    PrepareArgs,
    materialize_dataset,
    run_model_predict,
)

from .postprocess import OUTPUT_BANDS, OUTPUT_LAYER, collect_features

logger = get_logger(__name__)

DATASET_CONFIG_FNAME = "data/change_finder_v2/lcc_model/config_predict.json"
MODEL_CONFIG_FNAME = (
    "data/change_finder_v2/lcc_model/config_pass20_v1_2_predict.yaml"
)

# Per-window size. The tile size (passed via bounds) must be a multiple of this.
PATCH_SIZE = 2048
RESOLUTION = 10

PREDICTION_GROUP = "predict"

# Default postprocessing parameters (match postprocess.py CLI defaults).
DEFAULT_THRESHOLD = 128
DEFAULT_MIN_PIXELS = 10
DEFAULT_POSTPROCESS_WORKERS = 32

CHANGE_FINDER_MATERIALIZE_PIPELINE_ARGS = MaterializePipelineArgs(
    disabled_layers=[],
    # Use initial job for prepare since it involves caching steps that should only be
    # performed once.
    prepare_args=PrepareArgs(
        apply_windows_args=ApplyWindowsArgs(
            group=PREDICTION_GROUP, workers=32, use_initial_job=True
        )
    ),
    # The OlmoEarth Datasets source sets ingest=false, so this step is a no-op, but we
    # keep it for parity with the standard materialize pipeline.
    ingest_args=IngestArgs(
        ignore_errors=False,
        apply_windows_args=ApplyWindowsArgs(
            group=PREDICTION_GROUP, workers=32, use_initial_job=False
        ),
    ),
    materialize_args=MaterializeArgs(
        ignore_errors=False,
        retry_max_attempts=3,
        retry_backoff=timedelta(seconds=5),
        apply_windows_args=ApplyWindowsArgs(
            group=PREDICTION_GROUP, workers=128, use_initial_job=False
        ),
    ),
)


def get_output_fname(
    out_path: str, projection: Projection, bounds: PixelBounds, raster: bool = False
) -> UPath:
    """Get the output filename to use for this task.

    Args:
        out_path: the output directory.
        projection: the projection of this task.
        bounds: the bounds of this task.
        raster: whether this is the merged raster output (.tif) rather than the
            polygonized GeoJSON (.geojson).

    Returns:
        the output filename.
    """
    suffix = "tif" if raster else "geojson"
    return UPath(out_path) / f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}.{suffix}"


def merge_and_upload_raster(
    projection: Projection,
    bounds: PixelBounds,
    windows: list[Window],
    out_fname: UPath,
) -> None:
    """Mosaic each window's 49-band output_change raster into a tile-level GeoTIFF.

    Windows without a prediction (missing input imagery) are left as zeros.

    Args:
        projection: the UTM projection that we are working in.
        bounds: the overall bounds of this task.
        windows: the windows that were used for prediction.
        out_fname: the filename to write the merged result.
    """
    num_bands = len(OUTPUT_BANDS)
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    prediction = np.zeros((num_bands, height, width), dtype=np.uint8)

    for window in windows:
        if window.projection != projection:
            raise ValueError(
                "expected projection of window to match the task projection"
            )
        tif_path = window.get_raster_dir(OUTPUT_LAYER, OUTPUT_BANDS) / "geotiff.tif"
        if not tif_path.exists():
            # Required input layers must have been missing, so no prediction was made.
            continue

        with tif_path.open("rb") as f:
            with rasterio.open(f) as src:
                arr = src.read()

        col_offset = window.bounds[0] - bounds[0]
        row_offset = window.bounds[1] - bounds[1]
        prediction[
            :,
            row_offset : row_offset + PATCH_SIZE,
            col_offset : col_offset + PATCH_SIZE,
        ] = arr

    GeotiffRasterFormat().encode_raster(
        out_fname.parent,
        projection,
        bounds,
        RasterArray(chw_array=prediction),
        fname=out_fname.name,
    )


def predict_pipeline(
    projection_json: str,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    out_path: str,
    scratch_path: str,
    write_raster: bool = False,
    threshold: int = DEFAULT_THRESHOLD,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    workers: int = DEFAULT_POSTPROCESS_WORKERS,
) -> None:
    """Compute LCC model outputs over one tile.

    Args:
        projection_json: JSON-encoded projection, normally a UTM zone with 10 m/pixel
            resolution.
        bounds: pixel coordinates within the projection on which to compute outputs.
            Each value must be a multiple of PATCH_SIZE.
        time_range: the reference timestamp as (T, T). The layer time_offset/duration
            derive the quarterly lookback and frequent lookforward from this.
        out_path: directory to write the outputs (per-tile GeoJSON, optionally a
            per-tile GeoTIFF named based on the bounds).
        scratch_path: where to store the temporary rslearn dataset.
        write_raster: also write the merged 49-band raster GeoTIFF.
        threshold: binary change probability threshold (0-255) for polygonization.
        min_pixels: minimum connected-component size for polygonization.
        workers: parallel workers for polygonization.
    """
    projection = Projection.deserialize(json.loads(projection_json))

    out_fname = get_output_fname(out_path, projection, bounds)
    raster_fname = get_output_fname(out_path, projection, bounds, raster=True)
    if out_fname.exists() and (not write_raster or raster_fname.exists()):
        logger.info(f"output file {out_fname} already exists")
        return

    # Initialize an rslearn dataset in scratch from the predict dataset config.
    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True)
    shutil.copyfile(DATASET_CONFIG_FNAME, ds_path / "config.json")

    # Create PATCH_SIZE windows covering the tile bounds.
    for value in bounds:
        assert value % PATCH_SIZE == 0
    tile_to_window: dict[tuple[int, int], Window] = {}
    dataset = Dataset(ds_path)
    for tile_col in range(bounds[0] // PATCH_SIZE, bounds[2] // PATCH_SIZE):
        for tile_row in range(bounds[1] // PATCH_SIZE, bounds[3] // PATCH_SIZE):
            window_bounds = (
                tile_col * PATCH_SIZE,
                tile_row * PATCH_SIZE,
                (tile_col + 1) * PATCH_SIZE,
                (tile_row + 1) * PATCH_SIZE,
            )
            window = Window(
                storage=dataset.storage,
                group=PREDICTION_GROUP,
                name=f"{tile_col}_{tile_row}",
                projection=projection,
                bounds=window_bounds,
                time_range=time_range,
            )

            # Skip windows too close to or crossing 0/180 longitude.
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

    # Materialize imagery for the windows.
    logger.info("materialize dataset")
    materialize_dataset(
        ds_path, materialize_pipeline_args=CHANGE_FINDER_MATERIALIZE_PIPELINE_ARGS
    )

    # Run the model only if at least one window has materialized imagery.
    completed_fnames = list(
        ds_path.glob(
            f"windows/{PREDICTION_GROUP}/*/layers/sentinel2_quarterly/completed"
        )
    )
    if len(completed_fnames) == 0:
        logger.info("skipping prediction since no windows seem to have data")
    else:
        run_model_predict(MODEL_CONFIG_FNAME, ds_path)

    # Polygonize the change raster into a per-tile GeoJSON.
    features = collect_features(
        dataset_path=str(ds_path),
        threshold=threshold,
        min_pixels=min_pixels,
        workers=workers,
    )
    fc = {"type": "FeatureCollection", "features": features}
    with out_fname.open("w") as f:
        json.dump(fc, f)
    logger.info("wrote %d features to %s", len(features), out_fname)

    # Optionally also write the merged 49-band raster.
    if write_raster:
        merge_and_upload_raster(
            projection, bounds, list(tile_to_window.values()), raster_fname
        )
        logger.info("wrote merged raster to %s", raster_fname)


class PredictTaskArgs:
    """Represents one prediction task among a set that shares paths."""

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
            time_range: the time range (reference timestamp) of this task.
        """
        self.projection_json = projection_json
        self.bounds = bounds
        self.time_range = time_range

    def serialize(self) -> dict[str, Any]:
        """Serialize the task to a JSON-encodable dictionary."""
        return dict(
            projection_json=self.projection_json,
            bounds=list(self.bounds),
            time_range=[
                self.time_range[0].isoformat(),
                self.time_range[1].isoformat(),
            ],
        )


def predict_multi(
    out_path: str,
    scratch_path: str,
    tasks: str,
    write_raster: bool = False,
    threshold: int = DEFAULT_THRESHOLD,
) -> None:
    """Run multiple prediction tasks sequentially.

    Args:
        out_path: directory to write outputs.
        scratch_path: local directory to use for scratch space.
        tasks: JSON-encoded list of serialized PredictTaskArgs dicts (see
            PredictTaskArgs.serialize). Passed as a plain string and parsed here to
            avoid jsonargparse re-parsing the nested timestamp/bounds values.
        write_raster: also write the merged 49-band raster GeoTIFF per tile.
        threshold: binary change probability threshold (0-255) for polygonization.
    """
    os.makedirs(scratch_path, exist_ok=True)
    for task in json.loads(tasks):
        bounds = tuple(task["bounds"])
        time_range = (
            datetime.fromisoformat(task["time_range"][0]),
            datetime.fromisoformat(task["time_range"][1]),
        )
        with tempfile.TemporaryDirectory(dir=scratch_path) as tmp_dir:
            logger.info(f"running task {task} in temporary directory {tmp_dir}")
            predict_pipeline(
                projection_json=json.dumps(task["projection_json"]),
                bounds=bounds,
                time_range=time_range,
                out_path=out_path,
                scratch_path=os.path.join(tmp_dir, "scratch"),
                write_raster=write_raster,
                threshold=threshold,
            )
