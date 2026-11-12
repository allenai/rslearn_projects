"""Compute quantized OlmoEarth embeddings over one block of a UTM zone.

A block is split into PATCH_SIZE windows. Imagery is materialized for each window into a
scratch rslearn dataset, the model runs over it, and each window's embeddings are
written into the GeoZarr store as one shard (see zarr_store.py). The block then gets a
completion marker, and a provenance file naming the source scenes behind each window.

Pixels where every Sentinel-2 mosaic is empty are written as nodata (-128).
"""

import json
import multiprocessing
import os
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
from rslearn.utils.raster_format import GeotiffRasterFormat
from shapely import box as shapely_box
from upath import UPath

from rslp.common.worker import PrefetchSpec
from rslp.large_scale_embeddings.model import NODATA_VALUE
from rslp.large_scale_embeddings.tiling import get_zone_wedge, list_kept_crops
from rslp.large_scale_embeddings.zarr_store import write_window_region
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

logger = get_logger(__name__)


class EmbeddingInputs(Enum):
    """Which input modalities the embeddings are computed from."""

    # Sentinel-2, plus Sentinel-1 and Landsat 8/9 where available, through the
    # model's 128-dim distilled head.
    S2_S1_LANDSAT_DISTILLED = "s2_s1_landsat_distilled"


DATASET_CONFIG_FNAME = "data/large_scale_embeddings/{inputs}.json"
MODEL_CONFIG_FNAME = "data/large_scale_embeddings/{inputs}.yaml"

# Window size in pixels. Block bounds must be multiples of this.
PATCH_SIZE = 2048
RESOLUTION = 10

PREDICTION_GROUP = "predict"

# Provenance is written to a sibling of the marker directory with this prefix swapped.
COMPLETED_DIR_PREFIX = "completed"
PROVENANCE_DIR_PREFIX = "provenance"

SENTINEL2_LAYER = "sentinel2_l2a"
OUTPUT_LAYER = "output"

# Written once a scratch dataset is materialized, so a later run goes straight to
# inference.
MATERIALIZED_SENTINEL = "materialized"

# Lets a worker materialize the next block in a subprocess while the GPU runs the
# current one (see rslp.common.worker).
PREFETCH: PrefetchSpec = {"workflow": "materialize", "scratch_arg": "--scratch_path"}
# Band count of the store's embedding array, which is fixed when the store is created.
EMBEDDING_DIM = 128

MATERIALIZE_PIPELINE_ARGS = MaterializePipelineArgs(
    disabled_layers=[],
    prepare_args=PrepareArgs(
        retry_max_attempts=10,
        retry_backoff=timedelta(seconds=10),
        apply_windows_args=ApplyWindowsArgs(
            group=PREDICTION_GROUP, workers=32, use_initial_job=True
        ),
    ),
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

# The encoder's mutually exclusive ways to say where the weights are.
CHECKPOINT_ARGS = ("model_id", "model_path", "checkpoint_path")

# Identifies the code that wrote a block, recorded in its marker and provenance.
CODE_VERSION_ENV_VAR = "RSLP_CODE_VERSION"


class AllZeroEmbeddingError(Exception):
    """A pixel with input data has an all-zero embedding, the signature of a NaN."""


def _checkpoint_arg(checkpoint_path: str) -> str:
    """Which encoder argument to pass `checkpoint_path` as.

    Args:
        checkpoint_path: a released bundle (with weights.pth) or a training checkpoint.

    Returns:
        "model_path" for a released bundle, "checkpoint_path" otherwise.
    """
    return (
        "model_path"
        if (UPath(checkpoint_path) / "weights.pth").exists()
        else "checkpoint_path"
    )


def _get_model_extra_args(
    model_config_fname: str,
    checkpoint_path: str,
    patch_size: int,
    location_query_patch_size: int,
    window_size: int,
    overlap_size: int,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> list[str]:
    """Get the arguments to pass to rslearn model predict, overriding the config.

    The encoder and callbacks are lists, which jsonargparse cannot override entry by
    entry, so they are loaded from the config, updated, and passed whole as JSON.

    Args:
        model_config_fname: the model configuration file.
        checkpoint_path: the OlmoEarth checkpoint to compute embeddings with.
        patch_size: the encoder's token patch size.
        location_query_patch_size: input pixels per output embedding.
        window_size: the size of the crops the model operates on.
        overlap_size: overlap in pixels between adjacent crops.
        batch_size: crops per batch, or None to keep the config's value.
        num_workers: data loader processes, or None to keep the config's value.

    Returns:
        list of arguments to pass to rslearn model predict.
    """
    with open(model_config_fname) as f:
        model_config = yaml.safe_load(f)

    encoder = model_config["model"]["init_args"]["model"]["init_args"]["encoder"]
    encoder_args = encoder[0]["init_args"]
    # The encoder accepts exactly one of these, so the others must be absent.
    for name in CHECKPOINT_ARGS:
        encoder_args.pop(name, None)
    encoder_args[_checkpoint_arg(checkpoint_path)] = checkpoint_path
    encoder_args["patch_size"] = patch_size
    encoder_args["forward_kwargs"]["location_query_patch_size"] = (
        location_query_patch_size
    )

    # The merger works at the output resolution.
    callbacks = model_config["trainer"]["callbacks"]
    merger = callbacks[0]["init_args"]["merger"]
    merger["init_args"]["downsample_factor"] = location_query_patch_size
    merger["init_args"]["overlap_pixels"] = overlap_size // location_query_patch_size

    args = [
        "--model.init_args.model.init_args.encoder",
        json.dumps(encoder),
        "--trainer.callbacks",
        json.dumps(callbacks),
        "--data.init_args.default_config.crop_size",
        str(window_size),
        "--data.init_args.predict_config.overlap_pixels",
        str(overlap_size),
    ]
    if batch_size is not None:
        args += ["--data.init_args.batch_size", str(batch_size)]
    if num_workers is not None:
        args += ["--data.init_args.num_workers", str(num_workers)]
    return args


def _window_span(
    dataset: Dataset, time_range: tuple[datetime, datetime]
) -> tuple[datetime, datetime]:
    """Widen a reference timestamp (T, T) to the span the dataset's layers request.

    The model builds its period grid from the window's time range, so a zero-length
    range would give it no periods.

    Args:
        dataset: the scratch dataset.
        time_range: the reference timestamp as (T, T).

    Returns:
        (T, the latest end any layer requests), or `time_range` if no layer has one.
    """
    ends = [
        requested[1]
        for layer in dataset.layers.values()
        if (source := getattr(layer, "data_source", None)) is not None
        and (requested := source.get_request_time_range(time_range)) is not None
    ]
    if not ends:
        return time_range
    return (time_range[0], max(max(ends), time_range[1]))


def get_marker_fname(
    completed_path: str, projection: Projection, bounds: PixelBounds
) -> UPath:
    """Get the per-block completion marker filename.

    Args:
        completed_path: the directory for completion markers.
        projection: the projection of the block.
        bounds: the pixel bounds of the block.

    Returns:
        the marker filename.
    """
    return UPath(completed_path) / f"{projection.crs!s}_{bounds[0]}_{bounds[1]}.json"


def get_provenance_fname(marker_fname: UPath) -> UPath:
    """Get the provenance filename for a block's marker filename.

    Args:
        marker_fname: the block's completion marker filename.

    Returns:
        the provenance filename, with the same basename in a sibling directory.
    """
    completed_upath = marker_fname.parent
    name = completed_upath.name
    if name.startswith(COMPLETED_DIR_PREFIX):
        name = PROVENANCE_DIR_PREFIX + name[len(COMPLETED_DIR_PREFIX) :]
    else:
        name = f"{name}_{PROVENANCE_DIR_PREFIX}"
    return completed_upath.parent / name / marker_fname.name


def _collect_provenance(windows: list[Window]) -> dict[str, dict]:
    """Record the source scenes composited into each window's mosaics.

    Args:
        windows: the windows of this block, after materialize.

    Returns:
        a dict from window name to layer name to the per-mosaic item groups.
    """
    provenance: dict[str, dict] = {}
    for window in windows:
        layers: dict[str, list[dict]] = {}
        for layer_name, layer_data in window.load_layer_datas().items():
            if layer_name == OUTPUT_LAYER:
                continue
            group_time_ranges = layer_data.group_time_ranges or [None] * len(
                layer_data.serialized_item_groups
            )
            layers[layer_name] = [
                {
                    "time_range": (
                        [
                            group_time_range[0].isoformat(),
                            group_time_range[1].isoformat(),
                        ]
                        if group_time_range is not None
                        else None
                    ),
                    "items": items,
                }
                for items, group_time_range in zip(
                    layer_data.serialized_item_groups, group_time_ranges, strict=True
                )
            ]
        provenance[window.name] = layers
    return provenance


def _crop_crosses_bad_longitude(projection: Projection, bounds: PixelBounds) -> bool:
    """Check whether a crop touches or crosses the antimeridian.

    Mosaics for such crops are unreliable, so they are skipped.

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
    touches_antimeridian = (
        wgs84_bounds[0] <= -180 + epsilon or wgs84_bounds[2] >= 180 - epsilon
    )
    # A polygon crossing the antimeridian has bounds spanning most of the globe.
    spans_the_seam = wgs84_bounds[0] < -90 and wgs84_bounds[2] > 90
    return touches_antimeridian or spans_the_seam


def _core_bounds(window: Window, context_pixels: int) -> PixelBounds:
    """The window's bounds without its context margin, i.e. the crop it writes."""
    return (
        window.bounds[0] + context_pixels,
        window.bounds[1] + context_pixels,
        window.bounds[2] - context_pixels,
        window.bounds[3] - context_pixels,
    )


def _read_window_embeddings(
    dataset: Dataset,
    window: Window,
    bounds: PixelBounds,
    location_query_patch_size: int,
) -> np.ndarray:
    """Read a window's int8 embeddings, with pixels lacking Sentinel-2 set to nodata.

    Args:
        dataset: the scratch dataset.
        window: the window to read.
        bounds: the part of the window to read, at the input resolution.
        location_query_patch_size: input pixels per output embedding.

    Returns:
        the int8 embedding array of shape (band, height, width).

    Raises:
        AllZeroEmbeddingError: if any pixel with input data is all-zero.
    """
    scale = location_query_patch_size
    out_projection = Projection(
        window.projection.crs,
        window.projection.x_resolution * scale,
        window.projection.y_resolution * scale,
    )
    out_bounds = (
        bounds[0] // scale,
        bounds[1] // scale,
        bounds[2] // scale,
        bounds[3] // scale,
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

    # A pixel is valid if any band is nonzero in any Sentinel-2 mosaic.
    valid = np.zeros((bounds[3] - bounds[1], bounds[2] - bounds[0]), dtype=bool)
    for layer_name, group_idx in window.list_completed_layers():
        if layer_name != SENTINEL2_LAYER:
            continue
        s2_array = window.data.read_raster(
            SENTINEL2_LAYER,
            dataset.layers[SENTINEL2_LAYER].band_sets[0].bands,
            GeotiffRasterFormat(),
            bounds=bounds,
            group_idx=group_idx,
            resampling=Resampling.nearest,
        ).get_chw_array()
        valid |= (s2_array != 0).any(axis=0)
    # An output pixel is valid if any input pixel in its patch is.
    if scale > 1:
        valid = valid.reshape(
            valid.shape[0] // scale, scale, valid.shape[1] // scale, scale
        ).any(axis=(1, 3))
    return _mask_invalid_pixels(embeddings, valid)


def _mask_invalid_pixels(embeddings: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Set pixels outside the valid mask to NODATA_VALUE, and reject all-zero vectors.

    Args:
        embeddings: int8 array of shape (band, height, width), modified in place.
        valid: bool array of shape (height, width), True where the inputs had data.

    Returns:
        the masked embeddings array.

    Raises:
        AllZeroEmbeddingError: if any valid pixel holds an all-zero vector.
    """
    all_zero = valid & ~embeddings.any(axis=0)
    if all_zero.any():
        raise AllZeroEmbeddingError(
            f"{int(all_zero.sum())} of {int(valid.sum())} valid pixels have "
            "all-zero embeddings"
        )
    embeddings[:, ~valid] = NODATA_VALUE
    return embeddings


def _write_window_by_name(
    ds_path: UPath,
    window_name: str,
    store_path: str,
    time_index: int,
    location_query_patch_size: int,
    context_pixels: int,
) -> None:
    """Load one window from the scratch dataset and write its embeddings to the store.

    Runs in a worker process, so the window is reloaded by name.

    Args:
        ds_path: the scratch dataset path.
        window_name: the name of the window to write.
        store_path: the GeoZarr store path.
        time_index: the index into the store's time axis for this reference year.
        location_query_patch_size: input pixels per output embedding.
        context_pixels: the margin the window carries on each side, read as model
            input but not written.
    """
    dataset = Dataset(ds_path)
    windows = dataset.load_windows(groups=[PREDICTION_GROUP], names=[window_name])
    if len(windows) != 1:
        raise ValueError(
            f"expected one window named {window_name} but got {len(windows)}"
        )
    window = windows[0]
    epsg = window.projection.crs.to_epsg()
    if epsg // 100 != 326:
        raise ValueError(f"window {window_name} is in EPSG:{epsg}, expected 326NN")
    bounds = _core_bounds(window, context_pixels)
    embeddings = _read_window_embeddings(
        dataset, window, bounds, location_query_patch_size
    )
    write_window_region(
        store_path=store_path,
        zone_number=epsg % 100,
        window_bounds=bounds,
        time_index=time_index,
        embeddings=embeddings,
        patch_size=location_query_patch_size,
    )


def predict_pipeline(
    inputs: EmbeddingInputs,
    projection_json: str,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    store_path: str,
    completed_path: str,
    checkpoint_path: str,
    time_index: int,
    patch_size: int = 2,
    location_query_patch_size: int = 1,
    window_size: int = 480,
    overlap_size: int = 32,
    context_pixels: int = 48,
    batch_size: int | None = None,
    num_workers: int | None = None,
    scratch_path: str | None = None,
    upload_workers: int = 16,
) -> None:
    """Compute quantized OlmoEarth embeddings over one block.

    Embeddings depend on the inputs, checkpoint, patch sizes, window_size,
    overlap_size and context_pixels, so a run must keep all of them fixed per store.

    Args:
        inputs: which input variant to use.
        projection_json: JSON-encoded projection, the zone's northern UTM CRS
            (EPSG:326NN) at 10 m/pixel.
        bounds: pixel bounds of the block, each a multiple of PATCH_SIZE.
        time_range: the reference timestamp as (T, T). The layers derive the monthly
            mosaics over the following year from it.
        store_path: the GeoZarr store to write into, created by init_store.
        completed_path: directory to write the block's completion marker to.
        checkpoint_path: the OlmoEarth checkpoint or released model bundle.
        time_index: the index into the store's time axis for this reference year.
        patch_size: the encoder's token patch size.
        location_query_patch_size: input pixels per output embedding.
        window_size: the size of the crops the model operates on.
        overlap_size: overlap in pixels between adjacent crops.
        context_pixels: imagery margin added on each side of every window and given
            to the model, but not written.
        batch_size: crops per batch, or None to keep the config's value.
        num_workers: data loader processes, or None to keep the config's value.
        scratch_path: directory for the scratch rslearn dataset, kept afterward. By
            default a temporary directory is used and deleted.
        upload_workers: processes writing windows to the store. If scratch_path was
            already filled by the materialize workflow, inference runs on it directly.
    """
    _run_block(materialize_only=False, **locals())


def materialize_pipeline(
    *,
    inputs: EmbeddingInputs,
    projection_json: str,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    store_path: str,
    completed_path: str,
    checkpoint_path: str,
    time_index: int,
    scratch_path: str,
    patch_size: int = 2,
    location_query_patch_size: int = 1,
    window_size: int = 480,
    overlap_size: int = 32,
    context_pixels: int = 48,
    batch_size: int | None = None,
    num_workers: int | None = None,
    upload_workers: int = 16,
) -> None:
    """Only materialize one block's imagery into scratch_path.

    A later predict call with the same arguments runs inference on it. Takes the
    same arguments as predict_pipeline so one queue entry can drive both.
    """
    _run_block(materialize_only=True, **locals())


def _run_block(
    inputs: EmbeddingInputs,
    projection_json: str,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    store_path: str,
    completed_path: str,
    checkpoint_path: str,
    time_index: int,
    patch_size: int,
    location_query_patch_size: int,
    window_size: int,
    overlap_size: int,
    context_pixels: int,
    batch_size: int | None,
    num_workers: int | None,
    scratch_path: str | None,
    upload_workers: int,
    materialize_only: bool,
) -> None:
    """Validate the arguments and process the block, see predict_pipeline."""
    for name, value in (
        ("window_size", window_size),
        ("overlap_size", overlap_size),
    ):
        if value % location_query_patch_size != 0:
            raise ValueError(
                f"{name} ({value}) must be a multiple of location_query_patch_size "
                f"({location_query_patch_size})"
            )
    if window_size % patch_size != 0:
        raise ValueError(
            f"window_size ({window_size}) must be a multiple of patch_size "
            f"({patch_size})"
        )
    if PATCH_SIZE % location_query_patch_size != 0:
        raise ValueError(
            f"location_query_patch_size must divide {PATCH_SIZE}, got "
            f"{location_query_patch_size}"
        )
    if context_pixels < 0 or context_pixels % max(
        patch_size, location_query_patch_size
    ):
        raise ValueError(
            f"context_pixels ({context_pixels}) must be a non-negative multiple of "
            f"patch_size and location_query_patch_size"
        )

    projection = Projection.deserialize(json.loads(projection_json))
    marker_fname = get_marker_fname(completed_path, projection, bounds)
    if marker_fname.exists():
        logger.info(f"marker file {marker_fname} already exists")
        return

    process_kwargs = dict(
        inputs=inputs,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        store_path=store_path,
        marker_fname=marker_fname,
        checkpoint_path=checkpoint_path,
        time_index=time_index,
        patch_size=patch_size,
        location_query_patch_size=location_query_patch_size,
        window_size=window_size,
        overlap_size=overlap_size,
        context_pixels=context_pixels,
        batch_size=batch_size,
        num_workers=num_workers,
        upload_workers=upload_workers,
        materialize_only=materialize_only,
    )
    if scratch_path is None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _process_block(ds_path=UPath(tmp_dir) / "dataset", **process_kwargs)
    else:
        _process_block(ds_path=UPath(scratch_path), **process_kwargs)


def _process_block(
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
    location_query_patch_size: int,
    window_size: int,
    overlap_size: int,
    context_pixels: int,
    batch_size: int | None,
    num_workers: int | None,
    upload_workers: int,
    materialize_only: bool,
) -> None:
    """Process one block using the given scratch dataset path.

    See predict_pipeline for the arguments; marker_fname is the block's completion
    marker and ds_path is where to create the scratch dataset.
    """
    dataset_config_fname = DATASET_CONFIG_FNAME.format(inputs=inputs.value)
    model_config_fname = MODEL_CONFIG_FNAME.format(inputs=inputs.value)
    materialized = (ds_path / MATERIALIZED_SENTINEL).exists()
    if not materialized:
        ds_path.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(dataset_config_fname, ds_path / "config.json")

    wedge = get_zone_wedge(projection.crs, projection.x_resolution)
    kept_crops = list_kept_crops(projection, bounds, PATCH_SIZE, wedge=wedge)

    dataset = Dataset(ds_path)
    window_time_range = _window_span(dataset, time_range)
    windows: list[Window] = []
    skipped_longitude: list[list[int]] = []
    for crop_bounds in kept_crops:
        # Materialized and run over the padded bounds, but named and written by the
        # crop.
        window_bounds = (
            crop_bounds[0] - context_pixels,
            crop_bounds[1] - context_pixels,
            crop_bounds[2] + context_pixels,
            crop_bounds[3] + context_pixels,
        )
        if _crop_crosses_bad_longitude(projection, window_bounds):
            skipped_longitude.append([crop_bounds[0], crop_bounds[1]])
            continue
        window = Window(
            storage=dataset.storage,
            group=PREDICTION_GROUP,
            name=f"{crop_bounds[0] // PATCH_SIZE}_{crop_bounds[1] // PATCH_SIZE}",
            projection=projection,
            bounds=window_bounds,
            time_range=window_time_range,
            data_factory=dataset.window_data_storage_factory,
        )
        if not materialized:
            window.save()
        windows.append(window)

    if materialized:
        # A scratch prepared with another margin would be cropped wrongly on write.
        saved = {
            w.name: tuple(w.bounds)
            for w in dataset.load_windows(groups=[PREDICTION_GROUP])
        }
        for window in windows:
            if saved.get(window.name) != tuple(window.bounds):
                raise ValueError(
                    f"prefetched window {window.name} has bounds "
                    f"{saved.get(window.name)} but context_pixels {context_pixels} "
                    f"needs {tuple(window.bounds)}"
                )
    elif len(windows) > 0:
        logger.info("materialize dataset")
        materialize_dataset(
            ds_path, materialize_pipeline_args=MATERIALIZE_PIPELINE_ARGS
        )
    if materialize_only:
        (ds_path / MATERIALIZED_SENTINEL).touch()
        logger.info("materialized %d windows into %s", len(windows), ds_path)
        return

    written: list[list[int]] = []
    skipped_no_data: list[list[int]] = []
    if len(windows) > 0:
        if not any(window.is_layer_completed(SENTINEL2_LAYER) for window in windows):
            logger.info("skipping prediction since no windows have data")
        else:
            run_model_predict(
                model_config_fname,
                ds_path,
                extra_args=_get_model_extra_args(
                    model_config_fname=model_config_fname,
                    checkpoint_path=checkpoint_path,
                    patch_size=patch_size,
                    location_query_patch_size=location_query_patch_size,
                    window_size=window_size,
                    overlap_size=overlap_size,
                    batch_size=batch_size,
                    num_workers=num_workers,
                ),
            )

        upload_kwargs: list[dict] = []
        for window in windows:
            crop_offset = list(_core_bounds(window, context_pixels)[:2])
            # No output means a required input layer was missing.
            if not window.is_layer_completed(OUTPUT_LAYER):
                skipped_no_data.append(crop_offset)
                continue
            upload_kwargs.append(
                {
                    "ds_path": ds_path,
                    "window_name": window.name,
                    "store_path": store_path,
                    "time_index": time_index,
                    "location_query_patch_size": location_query_patch_size,
                    "context_pixels": context_pixels,
                }
            )
            written.append(crop_offset)
        if len(upload_kwargs) > 0:
            # forkserver, since the CUDA context from prediction cannot be forked.
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
        logger.info("no crops to process for this block")

    # Provenance first, so a failure leaves the block unmarked and retried.
    code_version = os.environ.get(CODE_VERSION_ENV_VAR)
    provenance_fname = get_provenance_fname(marker_fname)
    provenance_fname.parent.mkdir(parents=True, exist_ok=True)
    with provenance_fname.open("w") as f:
        json.dump(
            {
                "projection": projection.serialize(),
                "bounds": list(bounds),
                "time_range": [time_range[0].isoformat(), time_range[1].isoformat()],
                "time_index": time_index,
                "context_pixels": context_pixels,
                "code_version": code_version,
                "window_bounds": {
                    window.name: list(window.bounds) for window in windows
                },
                "windows": _collect_provenance(windows),
            },
            f,
        )
    logger.info("wrote provenance file %s", provenance_fname)

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
        "window_size": window_size,
        "context_pixels": context_pixels,
        "code_version": code_version,
    }
    marker_fname.parent.mkdir(parents=True, exist_ok=True)
    with marker_fname.open("w") as f:
        json.dump(marker, f)
    logger.info("wrote marker file %s", marker_fname)
