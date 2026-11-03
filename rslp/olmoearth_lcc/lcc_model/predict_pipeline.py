"""Scaled prediction pipeline for the OlmoEarth LCC model.

This applies the land cover change model over a tile (one part of a UTM zone) by
creating PATCH_SIZE windows, materializing Sentinel-2 imagery from the OlmoEarth
Datasets source, running the model, and polygonizing the resulting change raster
into a per-tile GeoJSON. Optionally the merged output_change raster (full uint16,
write_raster) and/or the compact uint8 summary raster (write_summary_raster) are
also written, and/or each window is written into a GeoZarr store
(zarr_store_path, see geozarr.py). A small per-tile ``_meta.json`` records the
reference timestamp and the Sentinel-2 item groups (period time ranges and scene
names) that each window's input layers were built from. It is written last, so it
also serves as the tile's completion marker in GeoZarr mode.

Windows outside the UTM zone's canonical wedge are skipped (see tiling.py), since
neighboring zones cover them.

Unlike the satlas pipeline, there is no rtree index: the OlmoEarth Datasets source
queries its API per window, and all imagery is derived from a single reference
timestamp T via the layer time_offset/duration in config_predict.json.

This module is tile-size-agnostic: it accepts any ``bounds`` whose extents are
multiples of PATCH_SIZE. The fixed 32768x32768 tiling is defined in tiling.py and
used by write_jobs.py.
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
from rslearn.utils.mp import make_pool_and_star_imap_unordered
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

from . import geozarr
from .postprocess import (
    OUTPUT_BANDS,
    OUTPUT_LAYER,
    SUMMARY_BANDS,
    collect_features,
    geozarr_window_arrays,
    summary_window_array,
)
from .tiling import PATCH_SIZE, window_in_zone_wedge

logger = get_logger(__name__)

DATASET_CONFIG_FNAME = "data/olmoearth_lcc/lcc_model/config_predict.json"
MODEL_CONFIG_FNAME = (
    "data/olmoearth_lcc/lcc_model/config_bp_abs_qdrop_bpcat_1pass_predict.yaml"
)

PREDICTION_GROUP = "predict"

# Default postprocessing parameters (match postprocess.py CLI defaults).
DEFAULT_MIN_PIXELS = 10
DEFAULT_POSTPROCESS_WORKERS = 32

# Parallel workers for writing windows to the GeoZarr store. Each holds one full
# window output (about 480 MB) in memory.
DEFAULT_GEOZARR_WORKERS = 16

CHANGE_FINDER_MATERIALIZE_PIPELINE_ARGS = MaterializePipelineArgs(
    disabled_layers=[],
    # Use initial job for prepare since it involves caching steps that should only be
    # performed once.
    prepare_args=PrepareArgs(
        retry_max_attempts=10,
        retry_backoff=timedelta(seconds=5),
        apply_windows_args=ApplyWindowsArgs(
            group=PREDICTION_GROUP, workers=32, use_initial_job=True
        ),
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
        retry_max_attempts=10,
        retry_backoff=timedelta(seconds=5),
        apply_windows_args=ApplyWindowsArgs(
            group=PREDICTION_GROUP, workers=128, use_initial_job=False
        ),
    ),
)


def get_output_fname(
    out_path: str,
    projection: Projection,
    bounds: PixelBounds,
    raster: bool = False,
    summary: bool = False,
    meta: bool = False,
) -> UPath:
    """Get the output filename to use for this task.

    Args:
        out_path: the output directory.
        projection: the projection of this task.
        bounds: the bounds of this task.
        raster: whether this is the merged raster output (.tif) rather than the
            polygonized GeoJSON (.geojson).
        summary: whether this is the uint8 summary raster (implies raster);
            named with a _summary suffix so it can coexist with the full
            raster in the same output directory.
        meta: whether this is the per-tile JSON metadata sidecar (_meta.json).

    Returns:
        the output filename.
    """
    if meta:
        suffix = "_meta.json"
    elif summary:
        suffix = "_summary.tif"
    elif raster:
        suffix = ".tif"
    else:
        suffix = ".geojson"
    return UPath(out_path) / f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}{suffix}"


def merge_and_upload_raster(
    projection: Projection,
    bounds: PixelBounds,
    windows: list[Window],
    full_fname: UPath | None,
    summary_fname: UPath | None,
) -> None:
    """Mosaic each window's output_change raster into tile-level GeoTIFF(s).

    Windows without a prediction (missing input imagery) are left as zeros.

    Args:
        projection: the UTM projection that we are working in.
        bounds: the overall bounds of this task.
        windows: the windows that were used for prediction.
        full_fname: write the full merged raster (all OUTPUT_BANDS, uint16)
            here, or None to skip it.
        summary_fname: write the uint8 summary raster (SUMMARY_BANDS) here,
            or None to skip it.
    """
    height = bounds[3] - bounds[1]
    width = bounds[2] - bounds[0]
    full = None
    if full_fname is not None:
        full = np.zeros((len(OUTPUT_BANDS), height, width), dtype=np.uint16)
    summary = None
    if summary_fname is not None:
        summary = np.zeros((len(SUMMARY_BANDS), height, width), dtype=np.uint8)

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
        window_slice = (
            slice(None),
            slice(row_offset, row_offset + PATCH_SIZE),
            slice(col_offset, col_offset + PATCH_SIZE),
        )
        if full is not None:
            full[window_slice] = arr
        if summary is not None:
            summary[window_slice] = summary_window_array(arr)

    for prediction, out_fname in [(full, full_fname), (summary, summary_fname)]:
        if prediction is None:
            continue
        GeotiffRasterFormat().encode_raster(
            out_fname.parent,
            projection,
            bounds,
            RasterArray(chw_array=prediction),
            fname=out_fname.name,
        )
        logger.info("wrote merged raster to %s", out_fname)


def write_meta(
    out_fname: UPath,
    projection: Projection,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    windows: list[Window],
) -> None:
    """Write the per-tile metadata sidecar.

    For each window, records every input layer's item groups: the request time
    range of the group (the 90-day or 7-day period for these period-mosaic layers)
    and the names and capture time ranges of the scenes in it. This lets later
    steps (e.g. creating annotation windows from predictions) reproduce or check
    the exact images the model saw.

    Args:
        out_fname: where to write the JSON.
        projection: the tile projection.
        bounds: the tile bounds.
        time_range: the reference timestamp as (T, T).
        windows: the prediction windows of this tile.
    """
    window_metas = []
    for window in windows:
        layers = {}
        for layer_name, layer_data in window.load_layer_datas().items():
            group_time_ranges = layer_data.group_time_ranges or [None] * len(
                layer_data.serialized_item_groups
            )
            groups = []
            for group, group_time_range in zip(
                layer_data.serialized_item_groups, group_time_ranges
            ):
                groups.append(
                    dict(
                        time_range=(
                            [t.isoformat() for t in group_time_range]
                            if group_time_range is not None
                            else None
                        ),
                        items=[
                            dict(
                                name=item["name"],
                                time_range=item["geometry"].get("time_range"),
                            )
                            for item in group
                        ],
                    )
                )
            layers[layer_name] = dict(
                materialized=window.is_layer_completed(layer_name), groups=groups
            )
        window_metas.append(
            dict(name=window.name, bounds=list(window.bounds), layers=layers)
        )

    meta = dict(
        projection=projection.serialize(),
        bounds=list(bounds),
        time_range=[t.isoformat() for t in time_range],
        windows=window_metas,
    )
    with out_fname.open("w") as f:
        json.dump(meta, f)
    logger.info("wrote metadata to %s", out_fname)


def _write_geozarr_window(
    tif_path: UPath, store_path: str, projection_json: str, bounds: PixelBounds
) -> bool:
    """Encode one window's output_change raster and write it to the GeoZarr store.

    Args:
        tif_path: the window's output_change GeoTIFF.
        store_path: the GeoZarr store.
        projection_json: the JSON-encoded window projection.
        bounds: the window bounds.

    Returns:
        whether the window had a prediction to write.
    """
    if not tif_path.exists():
        # Required input layers must have been missing, so no prediction was made.
        return False
    with tif_path.open("rb") as f:
        with rasterio.open(f) as src:
            arr = src.read()
    change, change_dates = geozarr_window_arrays(arr)
    geozarr.write_window(
        store_path,
        Projection.deserialize(json.loads(projection_json)),
        bounds,
        change,
        change_dates,
    )
    return True


def write_geozarr(
    store_path: str, windows: list[Window], workers: int = DEFAULT_GEOZARR_WORKERS
) -> None:
    """Write each window's predictions into the GeoZarr store.

    Each window is one shard of the store's arrays, so windows can be written in
    parallel. Windows without a prediction are not written (they read as nodata).

    Args:
        store_path: the GeoZarr store, created beforehand by geozarr.init_store.
        windows: the windows that were used for prediction.
        workers: number of parallel worker processes.
    """
    kwargs_list = [
        dict(
            tif_path=window.get_raster_dir(OUTPUT_LAYER, OUTPUT_BANDS) / "geotiff.tif",
            store_path=store_path,
            projection_json=json.dumps(window.projection.serialize()),
            bounds=window.bounds,
        )
        for window in windows
    ]
    num_written = 0
    with make_pool_and_star_imap_unordered(
        workers, _write_geozarr_window, kwargs_list
    ) as results:
        for written in results:
            num_written += int(written)
    logger.info(
        "wrote %d of %d windows to GeoZarr store %s",
        num_written,
        len(windows),
        store_path,
    )


def predict_pipeline(
    projection_json: str,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    out_path: str,
    scratch_path: str,
    write_raster: bool = False,
    write_summary_raster: bool = False,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    workers: int = DEFAULT_POSTPROCESS_WORKERS,
    zarr_store_path: str | None = None,
    write_geojson: bool = True,
    geozarr_workers: int = DEFAULT_GEOZARR_WORKERS,
) -> None:
    """Compute LCC model outputs over one tile.

    Args:
        projection_json: JSON-encoded projection, normally a northern UTM zone
            (EPSG:326NN, spanning both hemispheres) with 10 m/pixel resolution.
        bounds: pixel coordinates within the projection on which to compute outputs.
            Each value must be a multiple of PATCH_SIZE.
        time_range: the reference timestamp as (T, T). The layer time_offset/duration
            derive the quarterly lookback and frequent lookforward from this.
        out_path: directory to write the outputs (per-tile GeoJSON and _meta.json,
            optionally a per-tile GeoTIFF named based on the bounds).
        scratch_path: where to store the temporary rslearn dataset.
        write_raster: also write the full merged output_change raster GeoTIFF
            (all OUTPUT_BANDS, uint16; includes the per-category scores).
        write_summary_raster: also write the compact uint8 summary raster
            (SUMMARY_BANDS), which the olmoearth_lcc_viewer consumes directly.
        min_pixels: minimum connected-component size for polygonization.
        workers: parallel workers for polygonization.
        zarr_store_path: also write each window into this GeoZarr store, which must
            have been created for this reference timestamp (geozarr.init_store).
            The projection must then be EPSG:326NN. The tile is considered complete
            once its _meta.json exists.
        write_geojson: whether to write the per-tile GeoJSON.
        geozarr_workers: parallel workers for writing to the GeoZarr store.
    """
    if zarr_store_path is None and not write_geojson:
        raise ValueError("write_geojson=false requires zarr_store_path")
    projection = Projection.deserialize(json.loads(projection_json))

    out_fname = get_output_fname(out_path, projection, bounds)
    raster_fname = get_output_fname(out_path, projection, bounds, raster=True)
    summary_fname = get_output_fname(out_path, projection, bounds, summary=True)
    meta_fname = get_output_fname(out_path, projection, bounds, meta=True)
    if zarr_store_path is not None:
        if meta_fname.exists():
            logger.info(f"tile marker {meta_fname} already exists")
            return
    elif (
        out_fname.exists()
        and (not write_raster or raster_fname.exists())
        and (not write_summary_raster or summary_fname.exists())
    ):
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

            # Skip windows belonging to a neighboring UTM zone.
            if not window_in_zone_wedge(projection, window_bounds):
                logger.debug(
                    "skipping window at column %d row %d because it is outside the zone wedge",
                    tile_col,
                    tile_row,
                )
                continue

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

    windows = list(tile_to_window.values())

    if zarr_store_path is not None:
        write_geozarr(zarr_store_path, windows, workers=geozarr_workers)

    # Polygonize the change raster into a per-tile GeoJSON.
    if write_geojson:
        _, features = collect_features(
            dataset_path=str(ds_path),
            min_pixels=min_pixels,
            workers=workers,
        )
        fc = {"type": "FeatureCollection", "features": features}
        with out_fname.open("w") as f:
            json.dump(fc, f)
        logger.info("wrote %d features to %s", len(features), out_fname)

    # Optionally also write the merged output_change raster(s).
    if write_raster or write_summary_raster:
        merge_and_upload_raster(
            projection,
            bounds,
            windows,
            raster_fname if write_raster else None,
            summary_fname if write_summary_raster else None,
        )

    # Written last so that its existence marks the tile as complete.
    write_meta(meta_fname, projection, bounds, time_range, windows)


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
    write_summary_raster: bool = False,
    zarr_store_path: str | None = None,
    write_geojson: bool = True,
) -> None:
    """Run multiple prediction tasks sequentially.

    Args:
        out_path: directory to write outputs.
        scratch_path: local directory to use for scratch space.
        tasks: JSON-encoded list of serialized PredictTaskArgs dicts (see
            PredictTaskArgs.serialize). Passed as a plain string and parsed here to
            avoid jsonargparse re-parsing the nested timestamp/bounds values.
        write_raster: also write the full merged output_change raster GeoTIFF per
            tile.
        write_summary_raster: also write the compact uint8 summary raster per tile.
        zarr_store_path: also write each tile into this GeoZarr store.
        write_geojson: whether to write the per-tile GeoJSON.
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
                write_summary_raster=write_summary_raster,
                zarr_store_path=zarr_store_path,
                write_geojson=write_geojson,
            )
