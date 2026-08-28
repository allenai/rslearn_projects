"""Scaled inference pipeline for global OlmoEarth embeddings.

This computes 10 m/pixel, 128-dimensional, int8-quantized OlmoEarth embeddings over
one tile (a part of a UTM zone) by creating PATCH_SIZE windows, materializing
Sentinel-2 (and optionally Sentinel-1) mosaics from the OlmoEarth Datasets source,
running the model, and writing each window's embeddings into the GeoZarr store
(see zarr_store.py). A per-tile marker file is written to completed_path once the
tile is done, recording which crops were written and which were skipped.

Windows that don't intersect the zone's canonical wedge or that are entirely ocean
are skipped (see tiling.py). Embedding pixels where all Sentinel-2 mosaics are empty
are set to the nodata value (-128).

This module is tile-size-agnostic: it accepts any ``bounds`` whose extents are
multiples of PATCH_SIZE. The fixed 32768x32768 tiling lives only in write_jobs.py.

Note that different input variants (see EmbeddingInputs), checkpoints, and model
settings (patch_size/window_size/overlap_size) produce different embeddings, so each
combination must use its own store and completed_path.
"""

import json
import multiprocessing
import shutil
import tempfile
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import yaml
from rasterio.enums import Resampling
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import GeotiffRasterFormat
from shapely import box as shapely_box
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

from .model import NODATA_VALUE
from .tiling import get_zone_wedge, list_kept_crops
from .zarr_store import write_window_region

logger = get_logger(__name__)


class EmbeddingInputs(Enum):
    """Which input modalities the embeddings are computed from."""

    S2 = "s2"
    S2_S1 = "s2_s1"
    S2_LANDSAT = "s2_landsat"
    # The distilled release candidate: same inputs as S2_LANDSAT but the model's
    # 128-dim student head rather than its 768-dim teacher. A separate variant because
    # the variant name is what separates the two stores -- the student is selected by
    # an environment variable, which leaves no trace in the output path, so running
    # both against one checkpoint under one variant would silently mix them.
    S2_LANDSAT_DISTILLED = "s2_landsat_distilled"


DATASET_CONFIG_FNAME = "data/large_scale_embeddings/{inputs}.json"
MODEL_CONFIG_FNAME = "data/large_scale_embeddings/{inputs}.yaml"

# Per-window size. The tile size (passed via bounds) must be a multiple of this.
PATCH_SIZE = 2048
RESOLUTION = 10

PREDICTION_GROUP = "predict"

SENTINEL2_LAYER = "sentinel2_l2a"
OUTPUT_LAYER = "output"
# Width of the model's embedding output, and so the band count of the GeoZarr array.
# The band *names* now come from the dataset config rather than being derived here, but
# init_store still needs the count up front: the array's shape is fixed at creation,
# before any dataset is materialized.
EMBEDDING_DIM = 128

# These pool sizes are the long-standing default and are known to work. Scaling them
# down to the job's window count was tried once and reverted, but on bad evidence (a
# mismeasured elapsed time), so treat that as untested rather than disproven. If you
# revisit it, note that materialize parallelizes over window x item-group units --
# each window pulls 12 monthly mosaics, so a 12-window job is ~144 units, not 12 --
# so sizing a pool by window count alone would under-parallelize. Measure completion
# rate over several job durations before concluding anything.
MATERIALIZE_PIPELINE_ARGS = MaterializePipelineArgs(
    disabled_layers=[],
    # Use initial job for prepare since it involves caching steps that should only be
    # performed once.
    prepare_args=PrepareArgs(
        retry_max_attempts=10,
        retry_backoff=timedelta(seconds=10),
        apply_windows_args=ApplyWindowsArgs(
            group=PREDICTION_GROUP, workers=32, use_initial_job=True
        ),
    ),
    # The OlmoEarth Datasets source sets ingest=false, so this step is a no-op, but we
    # keep it for parity with the standard materialize pipeline.
    ingest_args=IngestArgs(
        ignore_errors=False,
        retry_max_attempts=10,
        retry_backoff=timedelta(seconds=10),
        apply_windows_args=ApplyWindowsArgs(
            group=PREDICTION_GROUP, workers=32, use_initial_job=False
        ),
    ),
    materialize_args=MaterializeArgs(
        ignore_errors=False,
        retry_max_attempts=10,
        retry_backoff=timedelta(seconds=10),
        apply_windows_args=ApplyWindowsArgs(
            group=PREDICTION_GROUP, workers=128, use_initial_job=False
        ),
    ),
)


def _get_model_extra_args(
    model_config_fname: str,
    checkpoint_path: str,
    patch_size: int,
    window_size: int,
    overlap_size: int,
    compile_model: bool,
    batch_size: int | None,
) -> list[str]:
    """Get the extra arguments to pass to rslearn model predict.

    These override the defaults in the model config file. The encoder and callbacks
    are list-valued so individual entries cannot be overridden via jsonargparse dotted
    keys; instead, the corresponding lists are loaded from the model config, updated,
    and passed whole as JSON.

    Args:
        model_config_fname: the model configuration file.
        checkpoint_path: the OlmoEarth checkpoint to compute embeddings with.
        patch_size: the encoder patch size (yields one embedding per patch_size
            pixels).
        window_size: the size of the crops the model operates on.
        overlap_size: overlap in pixels between adjacent crops.
        compile_model: whether to compile the encoder transformer blocks.
        batch_size: crops per batch, or None to keep the config's value.

            This is the GPU-memory knob. Crops are tiny, so the config batches many of
            them for throughput, but a tile carrying every monthly input has far more
            channels per crop and the same batch no longer fits. Batching only groups
            independent crops, so a smaller batch changes footprint and speed, never
            the embeddings.

    Returns:
        list of arguments to pass to rslearn model predict.
    """
    with open(model_config_fname) as f:
        model_config = yaml.safe_load(f)

    # Set the checkpoint path, patch size, and compilation flag on the OlmoEarth
    # encoder (the first and only encoder entry).
    encoder = model_config["model"]["init_args"]["model"]["init_args"]["encoder"]
    encoder[0]["init_args"]["checkpoint_path"] = checkpoint_path
    encoder[0]["init_args"]["patch_size"] = patch_size
    encoder[0]["init_args"]["compile_model"] = compile_model

    # Set the merger options on the RslearnWriter callback (the first and only
    # callback entry). The merger operates at the output resolution, which is
    # 1/patch_size of the input resolution.
    callbacks = model_config["trainer"]["callbacks"]
    merger = callbacks[0]["init_args"]["merger"]
    merger["init_args"]["downsample_factor"] = patch_size
    merger["init_args"]["overlap_pixels"] = overlap_size // patch_size

    return [
        "--model.init_args.model.init_args.encoder",
        json.dumps(encoder),
        "--trainer.callbacks",
        json.dumps(callbacks),
        "--data.init_args.default_config.crop_size",
        str(window_size),
        "--data.init_args.predict_config.overlap_pixels",
        str(overlap_size),
        *(
            ["--data.init_args.batch_size", str(batch_size)]
            if batch_size is not None
            else []
        ),
    ]


def get_output_fname(
    out_path: str, projection: Projection, bounds: PixelBounds
) -> UPath:
    """Get the output GeoTIFF filename for one PATCH_SIZE crop.

    Args:
        out_path: the output directory.
        projection: the projection of the crop.
        bounds: the pixel bounds of the crop.

    Returns:
        the output filename.
    """
    return UPath(out_path) / f"{projection.crs!s}_{bounds[0]}_{bounds[1]}.tif"


def get_marker_fname(
    completed_path: str, projection: Projection, bounds: PixelBounds
) -> UPath:
    """Get the per-tile completion marker filename.

    Args:
        completed_path: the directory for completion markers.
        projection: the projection of the tile.
        bounds: the pixel bounds of the tile.

    Returns:
        the marker filename.
    """
    return UPath(completed_path) / f"{projection.crs!s}_{bounds[0]}_{bounds[1]}.json"


def _crop_crosses_bad_longitude(projection: Projection, bounds: PixelBounds) -> bool:
    """Check whether a crop is too close to or crossing 0/180 longitude.

    Mosaics for such crops are unreliable (items on the other side of the
    antimeridian may be matched), so we skip them like the other scaled inference
    pipelines do.

    Args:
        projection: the UTM projection.
        bounds: the pixel bounds of the crop.

    Returns:
        whether the crop should be skipped.
    """
    epsilon = 1e-4
    wgs84_geom = STGeometry(projection, shapely_box(*bounds), None).to_projection(
        WGS84_PROJECTION
    )
    wgs84_bounds = wgs84_geom.shp.bounds
    if wgs84_bounds[0] <= -180 + epsilon or wgs84_bounds[2] >= 180 - epsilon:
        return True
    if wgs84_bounds[0] < -90 and wgs84_bounds[2] > 90:
        return True
    return False


def _read_window_embeddings(
    dataset: Dataset, window: Window, patch_size: int
) -> np.ndarray:
    """Read the window's int8 embedding raster and mask invalid pixels to nodata.

    Reads the merged embedding output from the scratch dataset and sets pixels where
    all Sentinel-2 mosaics are empty to NODATA_VALUE.

    Args:
        dataset: the scratch dataset, used to look up band names from its config rather
            than hardcoding them here.
        window: the window to read.
        patch_size: the encoder patch size. The embedding raster is at 1/patch_size
            of the window resolution.

    Returns:
        the int8 embedding array of shape (band, height, width).
    """
    # The embedding raster is at 1/patch_size of the window resolution.
    out_projection = Projection(
        window.projection.crs,
        window.projection.x_resolution * patch_size,
        window.projection.y_resolution * patch_size,
    )
    out_bounds = (
        window.bounds[0] // patch_size,
        window.bounds[1] // patch_size,
        window.bounds[2] // patch_size,
        window.bounds[3] // patch_size,
    )
    raster = window.data.read_raster(
        OUTPUT_LAYER,
        dataset.layers[OUTPUT_LAYER].band_sets[0].bands,
        GeotiffRasterFormat(),
        projection=out_projection,
        bounds=out_bounds,
        resampling=Resampling.nearest,
    )
    embeddings = raster.get_chw_array().copy()

    # A pixel is valid if any band is nonzero in any of the Sentinel-2 mosaics.
    valid = np.zeros(
        (
            window.bounds[3] - window.bounds[1],
            window.bounds[2] - window.bounds[0],
        ),
        dtype=bool,
    )
    for layer_name, group_idx in window.list_completed_layers():
        if layer_name != SENTINEL2_LAYER:
            continue
        s2_array = window.data.read_raster(
            SENTINEL2_LAYER,
            dataset.layers[SENTINEL2_LAYER].band_sets[0].bands,
            GeotiffRasterFormat(),
            group_idx=group_idx,
            resampling=Resampling.nearest,
        ).get_chw_array()
        valid |= (s2_array != 0).any(axis=0)
    # Downsample the validity mask to the output resolution: an output pixel is
    # valid if any input pixel in its patch is valid.
    if patch_size > 1:
        valid = valid.reshape(
            valid.shape[0] // patch_size,
            patch_size,
            valid.shape[1] // patch_size,
            patch_size,
        ).any(axis=(1, 3))
    embeddings[:, ~valid] = NODATA_VALUE
    return embeddings


def _write_debug_geotiff(
    window: Window, embeddings: np.ndarray, debug_geotiff_path: str, patch_size: int
) -> None:
    """Write a window's embeddings to an uncompressed GeoTIFF for debugging.

    Args:
        window: the window being written.
        embeddings: the int8 embedding array of shape (band, height, width).
        debug_geotiff_path: the directory to write the GeoTIFF to.
        patch_size: the encoder patch size; the embeddings are at 1/patch_size of the
            window resolution, so the GeoTIFF is georeferenced at that resolution.
    """
    # The embedding raster is at 1/patch_size of the window resolution.
    out_projection = Projection(
        window.projection.crs,
        window.projection.x_resolution * patch_size,
        window.projection.y_resolution * patch_size,
    )
    out_bounds = (
        window.bounds[0] // patch_size,
        window.bounds[1] // patch_size,
        window.bounds[2] // patch_size,
        window.bounds[3] // patch_size,
    )
    out_fname = get_output_fname(debug_geotiff_path, window.projection, window.bounds)
    raster_format = GeotiffRasterFormat(
        always_enable_tiling=True,
        block_size=512,
        geotiff_options={"compress": "none"},
    )
    raster_format.encode_raster(
        out_fname.parent,
        out_projection,
        out_bounds,
        RasterArray(
            chw_array=embeddings,
            metadata=RasterMetadata(nodata_value=NODATA_VALUE),
        ),
        fname=out_fname.name,
    )


def _write_window_by_name(
    ds_path: UPath,
    window_name: str,
    store_path: str,
    time_index: int,
    patch_size: int,
    debug_geotiff_path: str | None,
) -> None:
    """Load one window from the scratch dataset and write its embeddings to the store.

    This is the multiprocessing worker for the write step; the window is reloaded by
    name so that only picklable arguments cross the process boundary. The window must
    be in its zone's northern CRS (EPSG:326NN) so its bounds match the store grid.

    Args:
        ds_path: the scratch dataset path.
        window_name: the name of the window to write.
        store_path: the GeoZarr store path.
        time_index: the index into the store's time axis for this reference year.
        patch_size: the encoder patch size.
        debug_geotiff_path: if set, also write an uncompressed GeoTIFF here.
    """
    dataset = Dataset(ds_path)
    windows = dataset.load_windows(groups=[PREDICTION_GROUP], names=[window_name])
    if len(windows) != 1:
        raise ValueError(
            f"expected one window named {window_name} but got {len(windows)}"
        )
    window = windows[0]
    embeddings = _read_window_embeddings(dataset, window, patch_size)
    zone_number = window.projection.crs.to_epsg() % 100
    write_window_region(
        store_path=store_path,
        zone_number=zone_number,
        window_bounds=window.bounds,
        time_index=time_index,
        embeddings=embeddings,
        patch_size=patch_size,
    )
    if debug_geotiff_path is not None:
        _write_debug_geotiff(window, embeddings, debug_geotiff_path, patch_size)


def predict_pipeline(
    inputs: EmbeddingInputs,
    projection_json: str,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    store_path: str,
    completed_path: str,
    checkpoint_path: str,
    time_index: int,
    patch_size: int = 1,
    window_size: int = 16,
    overlap_size: int = 4,
    compile_model: bool = True,
    batch_size: int | None = None,
    scratch_path: str | None = None,
    upload_workers: int = 16,
    debug_geotiff_path: str | None = None,
) -> None:
    """Compute quantized OlmoEarth embeddings over one tile.

    Args:
        inputs: which input variant to use. Different variants produce different
            embeddings so they must use different stores.
        projection_json: JSON-encoded projection, normally the zone's northern UTM CRS
            (EPSG:326NN) with 10 m/pixel resolution.
        bounds: pixel coordinates within the projection on which to compute outputs.
            Each value must be a multiple of PATCH_SIZE.
        time_range: the reference timestamp as (T, T). The layer time_offset/duration
            derive the twelve monthly mosaics over the following year from this.
        store_path: the GeoZarr store to write embeddings into (must be initialized
            by init_store first).
        completed_path: directory to write per-tile completion markers.
        checkpoint_path: the OlmoEarth checkpoint to compute embeddings with, e.g.
            /weka/dfive-default/helios/checkpoints/gabrielt/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform/step667200.
            Different checkpoints produce different embeddings so they must use
            different store and completed_path (same for patch_size, window_size, and
            overlap_size below).
        time_index: the index into the store's time axis for this reference year.
        patch_size: the encoder patch size; yields one 128-dimensional embedding per
            patch_size x patch_size pixels (so the output rasters are at 1/patch_size
            of the window resolution).
        window_size: the size of the crops the model operates on (much bigger than 16
            fails with the 12 monthly inputs at patch_size=1 due to GPU memory
            constraints).
        overlap_size: overlap in pixels between adjacent crops, to mitigate embedding
            seams at crop boundaries.
        compile_model: whether to compile the encoder transformer blocks.
        batch_size: crops per batch, or None to keep the config's value. Lower it
            for a tile whose full monthly input stack will not fit in GPU memory;
            batching groups independent crops, so this changes footprint and
            speed, never the embeddings.
        scratch_path: optional directory to store the scratch rslearn dataset in
            directly, and keep it afterward (useful for debugging). By default, a
            temporary directory is used and deleted when the tile is done.
        upload_workers: number of worker processes for writing the per-crop
            embeddings.
        debug_geotiff_path: if set, also write an uncompressed GeoTIFF per crop here
            (for debugging small runs).
    """
    if PATCH_SIZE % patch_size != 0:
        raise ValueError(f"patch_size must divide {PATCH_SIZE}, got {patch_size}")
    if window_size % patch_size != 0:
        raise ValueError(
            f"window_size ({window_size}) must be a multiple of patch_size "
            f"({patch_size})"
        )
    if overlap_size % patch_size != 0:
        raise ValueError(
            f"overlap_size ({overlap_size}) must be a multiple of patch_size "
            f"({patch_size})"
        )

    projection = Projection.deserialize(json.loads(projection_json))

    marker_fname = get_marker_fname(completed_path, projection, bounds)
    if marker_fname.exists():
        logger.info(f"marker file {marker_fname} already exists")
        return

    if scratch_path is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _process_tile(
                inputs=inputs,
                projection=projection,
                bounds=bounds,
                time_range=time_range,
                store_path=store_path,
                marker_fname=marker_fname,
                ds_path=UPath(tmp_dir) / "dataset",
                checkpoint_path=checkpoint_path,
                time_index=time_index,
                patch_size=patch_size,
                window_size=window_size,
                overlap_size=overlap_size,
                compile_model=compile_model,
                batch_size=batch_size,
                upload_workers=upload_workers,
                debug_geotiff_path=debug_geotiff_path,
            )
    else:
        _process_tile(
            inputs=inputs,
            projection=projection,
            bounds=bounds,
            time_range=time_range,
            store_path=store_path,
            marker_fname=marker_fname,
            ds_path=UPath(scratch_path),
            checkpoint_path=checkpoint_path,
            time_index=time_index,
            patch_size=patch_size,
            window_size=window_size,
            overlap_size=overlap_size,
            compile_model=compile_model,
            batch_size=batch_size,
            upload_workers=upload_workers,
            debug_geotiff_path=debug_geotiff_path,
        )


def _process_tile(
    inputs: EmbeddingInputs,
    projection: Projection,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    store_path: str,
    marker_fname: UPath,
    ds_path: UPath,
    checkpoint_path: str,
    time_index: int,
    patch_size: int,
    window_size: int,
    overlap_size: int,
    compile_model: bool,
    batch_size: int | None,
    upload_workers: int,
    debug_geotiff_path: str | None,
) -> None:
    """Process one tile using the given scratch dataset path.

    See predict_pipeline for details.

    Args:
        inputs: which input variant to use.
        projection: the projection of the tile.
        bounds: the pixel bounds of the tile.
        time_range: the reference timestamp as (T, T).
        store_path: the GeoZarr store to write embeddings into.
        marker_fname: the per-tile completion marker filename to write.
        ds_path: where to create the temporary rslearn dataset.
        checkpoint_path: the OlmoEarth checkpoint to compute embeddings with.
        time_index: the index into the store's time axis for this reference year.
        patch_size: the encoder patch size.
        window_size: the size of the crops the model operates on.
        overlap_size: overlap in pixels between adjacent crops.
        compile_model: whether to compile the encoder transformer blocks.
        batch_size: crops per batch, or None to keep the config's value. Lower it
            for a tile whose full monthly input stack will not fit in GPU memory;
            batching groups independent crops, so this changes footprint and
            speed, never the embeddings.
        upload_workers: number of worker processes for writing the per-crop
            embeddings.
        debug_geotiff_path: if set, also write an uncompressed GeoTIFF per crop here.
    """
    # Initialize an rslearn dataset in scratch from the predict dataset config.
    dataset_config_fname = DATASET_CONFIG_FNAME.format(inputs=inputs.value)
    model_config_fname = MODEL_CONFIG_FNAME.format(inputs=inputs.value)
    ds_path.mkdir(parents=True)
    shutil.copyfile(dataset_config_fname, ds_path / "config.json")

    # Determine which PATCH_SIZE crops to process (see tiling.py), and additionally
    # skip crops too close to 0/180 longitude.
    wedge = get_zone_wedge(projection.crs, projection.x_resolution)
    kept_crops = list_kept_crops(projection, bounds, PATCH_SIZE, wedge=wedge)

    dataset = Dataset(ds_path)
    windows: list[Window] = []
    skipped_longitude: list[list[int]] = []
    for crop_bounds in kept_crops:
        if _crop_crosses_bad_longitude(projection, crop_bounds):
            logger.debug(
                "skipping crop at %s because it is too close to 0/180 longitude",
                crop_bounds,
            )
            skipped_longitude.append([crop_bounds[0], crop_bounds[1]])
            continue
        window = Window(
            storage=dataset.storage,
            group=PREDICTION_GROUP,
            name=f"{crop_bounds[0] // PATCH_SIZE}_{crop_bounds[1] // PATCH_SIZE}",
            projection=projection,
            bounds=crop_bounds,
            time_range=time_range,
            data_factory=dataset.window_data_storage_factory,
        )
        window.save()
        windows.append(window)

    written: list[list[int]] = []
    skipped_no_data: list[list[int]] = []

    if len(windows) > 0:
        # Materialize imagery for the windows.
        logger.info("materialize dataset")
        materialize_dataset(
            ds_path, materialize_pipeline_args=MATERIALIZE_PIPELINE_ARGS
        )

        # Run the model only if at least one window has materialized imagery.
        if not any(window.is_layer_completed(SENTINEL2_LAYER) for window in windows):
            logger.info("skipping prediction since no windows seem to have data")
        else:
            run_model_predict(
                model_config_fname,
                ds_path,
                extra_args=_get_model_extra_args(
                    model_config_fname=model_config_fname,
                    checkpoint_path=checkpoint_path,
                    patch_size=patch_size,
                    window_size=window_size,
                    overlap_size=overlap_size,
                    compile_model=compile_model,
                    batch_size=batch_size,
                ),
            )

        # Write each window's embeddings to the store. The writes are handled by a
        # pool of worker processes since converting and writing the rasters is slow.
        # We use the forkserver context because the CUDA context initialized by
        # run_model_predict above cannot be safely forked.
        upload_kwargs: list[dict] = []
        for window in windows:
            crop_offset = [window.bounds[0], window.bounds[1]]
            if not window.is_layer_completed(OUTPUT_LAYER):
                # Required input layers must have been missing, so no prediction was
                # made for this window.
                skipped_no_data.append(crop_offset)
                continue
            upload_kwargs.append(
                dict(
                    ds_path=ds_path,
                    window_name=window.name,
                    store_path=store_path,
                    time_index=time_index,
                    patch_size=patch_size,
                    debug_geotiff_path=debug_geotiff_path,
                )
            )
            written.append(crop_offset)
        if len(upload_kwargs) > 0:
            pool = multiprocessing.get_context("forkserver").Pool(upload_workers)
            try:
                for _ in star_imap_unordered(
                    pool, _write_window_by_name, upload_kwargs
                ):
                    pass
            finally:
                pool.close()
                pool.join()
        logger.info(
            "wrote %d crops (%d skipped due to missing data)",
            len(written),
            len(skipped_no_data),
        )
    else:
        logger.info("no crops to process for this tile")

    # Write the per-tile completion marker.
    marker = {
        "projection": projection.serialize(),
        "bounds": list(bounds),
        "time_range": [time_range[0].isoformat(), time_range[1].isoformat()],
        "time_index": time_index,
        "written": written,
        "skipped_no_data": skipped_no_data,
        "skipped_longitude": skipped_longitude,
        "num_filtered_crops": (bounds[2] - bounds[0])
        * (bounds[3] - bounds[1])
        // (PATCH_SIZE * PATCH_SIZE)
        - len(kept_crops),
    }
    marker_fname.parent.mkdir(parents=True, exist_ok=True)
    with marker_fname.open("w") as f:
        json.dump(marker, f)
    logger.info("wrote marker file %s", marker_fname)
