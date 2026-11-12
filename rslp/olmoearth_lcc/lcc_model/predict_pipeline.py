"""Scaled prediction pipeline for the OlmoEarth LCC model.

This applies the land cover change model over a tile (one part of a UTM zone) by
creating PATCH_SIZE windows, materializing Sentinel-2 imagery from the OlmoEarth
Datasets source, running the model, and writing the outputs. Each tile has two
stages:

- prepare_tile: create the windows and materialize their imagery.
- predict_tile: run the model with LCCTileWriter (prediction_writer.py), which writes
  each window into the GeoZarr store (zarr_store_path, see geozarr.py) in background
  threads, then write the tile's levels of a pyramid store if requested
  (zarr_pyramid_path, see geozarr_pyramid.py) and the per-tile ``_meta.json``.

The worker (worker.py) overlaps the two: while one tile is predicted on the GPU, the
next tile is prepared in a background process (BackgroundPrepare). The
``_meta.json`` records the reference timestamp, per-stage timings, and the
Sentinel-2 item groups (period time ranges and scene names) that each window's input
layers were built from. It is written last, so it also serves as the tile's
completion marker.

Windows outside the UTM zone's canonical wedge are skipped (see tiling.py), since
neighboring zones cover them. Windows with no land on a 512-pixel lattice (per
global_land_mask) are skipped too, including coastal water that Sentinel-2 images.

Unlike the satlas pipeline, there is no rtree index: the OlmoEarth Datasets source
queries its API per window, and all imagery is derived from a single reference
timestamp T via the layer time_offset/duration in config_predict.json.

This module is tile-size-agnostic: it accepts any ``bounds`` whose extents are
multiples of PATCH_SIZE (multiples of TILE_SIZE with a pyramid store). The fixed
16384x16384 tiling is defined in tiling.py and used by write_jobs.py.
"""

import json
import multiprocessing
import shutil
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing.connection import Connection
from typing import Any

import yaml
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import PixelBounds, Projection
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

from . import geozarr, geozarr_pyramid
from .fast_inference import prepare_fast_inference
from .output_bands import OUTPUT_LAYER
from .prediction_writer import DEFAULT_WRITER_WORKERS, LCCTileWriter
from .tiling import (
    PATCH_SIZE,
    TILE_SIZE,
    check_bounds_aligned,
    window_has_land,
    window_in_zone_wedge,
)

logger = get_logger(__name__)

DATASET_CONFIG_FNAME = "data/olmoearth_lcc/lcc_model/config_predict.json"
MODEL_CONFIG_FNAME = (
    "data/olmoearth_lcc/lcc_model/config_bp_abs_bpcat_1pass_predict.yaml"
)

PREDICTION_GROUP = "predict"

# Parallel workers for the materialize step. This runs alongside prediction of the
# previous tile (dataloader workers and writer threads), so it is lower than the 128
# used when materialization ran alone.
DEFAULT_MATERIALIZE_WORKERS = 64


def get_materialize_pipeline_args(
    materialize_workers: int = DEFAULT_MATERIALIZE_WORKERS,
) -> MaterializePipelineArgs:
    """Get the materialize pipeline arguments for the prediction windows.

    Args:
        materialize_workers: parallel workers for the materialize step.

    Returns:
        the arguments for materialize_dataset.
    """
    return MaterializePipelineArgs(
        disabled_layers=[],
        # Use initial job for prepare since it involves caching steps that should
        # only be performed once.
        prepare_args=PrepareArgs(
            retry_max_attempts=10,
            retry_backoff=timedelta(seconds=5),
            apply_windows_args=ApplyWindowsArgs(
                group=PREDICTION_GROUP, workers=32, use_initial_job=True
            ),
        ),
        # The OlmoEarth Datasets source sets ingest=false, so this step is a no-op,
        # but we keep it for parity with the standard materialize pipeline.
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
                group=PREDICTION_GROUP,
                workers=materialize_workers,
                use_initial_job=False,
            ),
        ),
    )


def get_meta_fname(out_path: str, projection: Projection, bounds: PixelBounds) -> UPath:
    """Get the filename of a tile's ``_meta.json`` (its completion marker).

    Args:
        out_path: the output directory.
        projection: the projection of the tile.
        bounds: the bounds of the tile.

    Returns:
        the metadata filename.
    """
    return UPath(out_path) / f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}_meta.json"


def write_meta(
    out_fname: UPath,
    projection: Projection,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    windows: list[Window],
    timings: dict[str, float] | None = None,
    fast_inference: bool = False,
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
        timings: optional seconds spent in each stage of the pipeline.
        fast_inference: whether the predictions used fast inference (see
            fast_inference.py).
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
        fast_inference=fast_inference,
        windows=window_metas,
    )
    if timings is not None:
        meta["timings"] = timings
    with out_fname.open("w") as f:
        json.dump(meta, f)
    logger.info("wrote metadata to %s", out_fname)


def check_tile(
    projection: Projection, bounds: PixelBounds, zarr_pyramid_path: str | None
) -> None:
    """Check that a tile can be written to the GeoZarr (and pyramid) store.

    Args:
        projection: the tile projection.
        bounds: the tile bounds.
        zarr_pyramid_path: the pyramid store, if any.
    """
    geozarr.check_zone_projection(projection)
    check_bounds_aligned(bounds, PATCH_SIZE if zarr_pyramid_path is None else TILE_SIZE)


@dataclass
class PreparedTile:
    """A tile whose windows were created and materialized by prepare_tile."""

    projection: Projection
    bounds: PixelBounds
    # The reference timestamp as (T, T).
    time_range: tuple[datetime, datetime]
    # The rslearn dataset holding the tile's windows.
    ds_path: str
    # Whether at least one window has materialized imagery to predict on.
    has_data: bool
    # Seconds spent in each preparation stage.
    timings: dict[str, float]


def prepare_tile(
    projection: Projection,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    scratch_path: str,
    materialize_workers: int = DEFAULT_MATERIALIZE_WORKERS,
) -> PreparedTile:
    """Create the tile's windows in a scratch dataset and materialize their imagery.

    Args:
        projection: the tile projection.
        bounds: the tile bounds, multiples of PATCH_SIZE.
        time_range: the reference timestamp as (T, T).
        scratch_path: where to create the rslearn dataset.
        materialize_workers: parallel workers for the materialize step.

    Returns:
        the prepared tile.
    """
    timings: dict[str, float] = {}
    stage_start = time.monotonic()

    # Initialize an rslearn dataset in scratch from the predict dataset config.
    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True)
    shutil.copyfile(DATASET_CONFIG_FNAME, ds_path / "config.json")

    # Create PATCH_SIZE windows covering the tile bounds.
    check_bounds_aligned(bounds, PATCH_SIZE)
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

            # Skip windows that are entirely water.
            if not window_has_land(projection, window_bounds):
                logger.debug(
                    "skipping window at column %d row %d because it has no land",
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

    now = time.monotonic()
    timings["create_windows"] = now - stage_start
    stage_start = now

    # Materialize imagery for the windows.
    logger.info("materialize dataset %s", ds_path)
    materialize_dataset(
        ds_path,
        materialize_pipeline_args=get_materialize_pipeline_args(materialize_workers),
    )
    timings["materialize"] = time.monotonic() - stage_start

    completed_fnames = list(
        ds_path.glob(
            f"windows/{PREDICTION_GROUP}/*/layers/sentinel2_quarterly/completed"
        )
    )
    return PreparedTile(
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        ds_path=str(ds_path),
        has_data=len(completed_fnames) > 0,
        timings=timings,
    )


def get_writer_config(
    zarr_store_path: str,
    pyramid: bool,
    writer_workers: int,
    model_config: str = MODEL_CONFIG_FNAME,
) -> dict[str, Any]:
    """Get the trainer callback config for the LCCTileWriter.

    The callback replaces the RslearnWriter of the model config, whose merger it
    copies.

    Args:
        zarr_store_path: the GeoZarr store to write windows into.
        pyramid: whether the writer computes the level-1 pyramid reductions.
        writer_workers: writer threads of the LCCTileWriter.
        model_config: the model config with the RslearnWriter callback.

    Returns:
        the callback config (class_path and init_args).
    """
    with open(model_config) as f:
        callbacks = yaml.safe_load(f)["trainer"]["callbacks"]
    (rslearn_writer,) = (
        callback
        for callback in callbacks
        if callback["class_path"] == "rslearn.train.prediction_writer.RslearnWriter"
    )
    init_args: dict[str, Any] = dict(
        output_layer=OUTPUT_LAYER,
        zarr_store_path=zarr_store_path,
        pyramid=pyramid,
        workers=writer_workers,
    )
    if "merger" in rslearn_writer["init_args"]:
        init_args["merger"] = rslearn_writer["init_args"]["merger"]
    return dict(
        class_path=f"{LCCTileWriter.__module__}.{LCCTileWriter.__name__}",
        init_args=init_args,
    )


def predict_tile(
    prepared: PreparedTile,
    out_path: str,
    zarr_store_path: str,
    start_time: float,
    timings: dict[str, float],
    zarr_pyramid_path: str | None = None,
    writer_workers: int = DEFAULT_WRITER_WORKERS,
    model_extra_args: list[str] = [],
) -> None:
    """Run the model over a prepared tile and write its outputs.

    Args:
        prepared: the tile prepared by prepare_tile.
        out_path: directory to write the per-tile _meta.json.
        zarr_store_path: the GeoZarr store to write windows into.
        start_time: time.monotonic() when this process started working on the
            tile, for the total time in the timings.
        timings: the timings so far, which this adds to and records in _meta.json.
        zarr_pyramid_path: also write the tile's pyramid levels into this store.
        writer_workers: writer threads of the LCCTileWriter.
        model_extra_args: extra arguments for rslearn model predict (see
            fast_inference.py); whether there are any is recorded in _meta.json.
    """
    projection, bounds = prepared.projection, prepared.bounds
    stage_start = time.monotonic()

    def end_stage(name: str) -> None:
        nonlocal stage_start
        now = time.monotonic()
        timings[name] = now - stage_start
        stage_start = now

    windows = sorted(
        Dataset(UPath(prepared.ds_path)).load_windows(groups=[PREDICTION_GROUP]),
        key=lambda window: window.name,
    )

    # Run the model only if at least one window has materialized imagery. The
    # writer finishes writing all windows before prediction returns.
    writer: LCCTileWriter | None = None
    if not prepared.has_data:
        logger.info("skipping prediction since no windows seem to have data")
    else:
        writer_config = get_writer_config(
            zarr_store_path=zarr_store_path,
            pyramid=zarr_pyramid_path is not None,
            writer_workers=writer_workers,
        )
        cli = run_model_predict(
            MODEL_CONFIG_FNAME,
            UPath(prepared.ds_path),
            extra_args=model_extra_args,
            callbacks=[writer_config],
        )
        writer = next(
            callback
            for callback in cli.trainer.callbacks
            if isinstance(callback, LCCTileWriter)
        )
    end_stage("predict")
    if writer is not None:
        timings["write_wait"] = writer.write_wait
    results = writer.results if writer is not None else []

    # Tiles without predictions read as nodata in the pyramid store already.
    if zarr_pyramid_path is not None and results:
        level1_blocks = []
        for result in results:
            assert result.level1 is not None
            level1_blocks.append((result.bounds, *result.level1))
        geozarr_pyramid.write_tile_pyramid(
            zarr_pyramid_path, projection, bounds, level1_blocks
        )
        end_stage("write_pyramid")

    # Written last so that its existence marks the tile as complete. The total
    # excludes writing the metadata itself.
    timings["total"] = time.monotonic() - start_time
    write_meta(
        get_meta_fname(out_path, projection, bounds),
        projection,
        bounds,
        prepared.time_range,
        windows,
        timings=timings,
        fast_inference=bool(model_extra_args),
    )


def predict_pipeline(
    projection_json: str,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    out_path: str,
    scratch_path: str,
    zarr_store_path: str,
    zarr_pyramid_path: str | None = None,
    fast_inference: bool = True,
    materialize_workers: int = DEFAULT_MATERIALIZE_WORKERS,
    writer_workers: int = DEFAULT_WRITER_WORKERS,
) -> None:
    """Compute LCC model outputs over one tile.

    Args:
        projection_json: JSON-encoded projection, a northern UTM zone (EPSG:326NN,
            spanning both hemispheres) with 10 m/pixel resolution.
        bounds: pixel coordinates within the projection on which to compute outputs.
            Each value must be a multiple of PATCH_SIZE (TILE_SIZE with
            zarr_pyramid_path).
        time_range: the reference timestamp as (T, T). The layer time_offset/duration
            derive the quarterly lookback and frequent lookforward from this.
        out_path: directory to write the per-tile _meta.json (named based on the
            bounds).
        scratch_path: where to store the temporary rslearn dataset.
        zarr_store_path: write each window into this GeoZarr store, which must
            have been created beforehand (geozarr.init_store). The tile is
            considered complete once its _meta.json exists.
        zarr_pyramid_path: also write the tile's pyramid levels into this pyramid
            store (geozarr_pyramid.init_pyramid_store).
        fast_inference: compile the encoder and use cuDNN attention (see
            fast_inference.py). Requires torch>=2.8 and a CUDA GPU. Set false to
            run the model eagerly.
        materialize_workers: parallel workers for the materialize step.
        writer_workers: writer threads for the GeoZarr outputs.
    """
    projection = Projection.deserialize(json.loads(projection_json))
    check_tile(projection, bounds, zarr_pyramid_path)
    meta_fname = get_meta_fname(out_path, projection, bounds)
    if meta_fname.exists():
        logger.info(f"tile marker {meta_fname} already exists")
        return

    # Prepare fast inference before materializing so that it fails early.
    model_extra_args = prepare_fast_inference() if fast_inference else []

    start_time = time.monotonic()
    prepared = prepare_tile(
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        scratch_path=scratch_path,
        materialize_workers=materialize_workers,
    )
    predict_tile(
        prepared,
        out_path=out_path,
        zarr_store_path=zarr_store_path,
        start_time=start_time,
        timings=dict(prepared.timings),
        zarr_pyramid_path=zarr_pyramid_path,
        writer_workers=writer_workers,
        model_extra_args=model_extra_args,
    )


def _prepare_tile_in_child(conn: Connection, kwargs: dict[str, Any]) -> None:
    """Run prepare_tile in a background process and send the result to the parent."""
    try:
        conn.send(("ok", prepare_tile(**kwargs)))
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class BackgroundPrepare:
    """prepare_tile running in a separate process.

    The process uses spawn so it does not fork a CUDA-initialized parent, and it is
    not a daemon since materialize_dataset creates its own worker pools.
    """

    def __init__(self, kwargs: dict[str, Any]):
        """Start preparing a tile.

        Args:
            kwargs: the prepare_tile arguments.
        """
        ctx = multiprocessing.get_context("spawn")
        self._conn, child_conn = ctx.Pipe(duplex=False)
        self._process = ctx.Process(
            target=_prepare_tile_in_child, args=(child_conn, kwargs), daemon=False
        )
        self._process.start()
        child_conn.close()

    def result(self) -> PreparedTile:
        """Wait for the tile to be prepared.

        Returns:
            the prepared tile.

        Raises:
            RuntimeError: if preparing the tile failed.
        """
        try:
            status, payload = self._conn.recv()
        except EOFError:
            status, payload = "error", "background process exited without a result"
        self._process.join()
        if status != "ok":
            raise RuntimeError(f"preparing tile failed:\n{payload}")
        return payload

    def terminate(self) -> None:
        """Stop the background process if it is still running."""
        if self._process.is_alive():
            self._process.terminate()
        self._process.join()
