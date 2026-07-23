"""Scaled inference pipeline for global OlmoEarth embeddings.

This computes 10 m/pixel, 128-dimensional, int8-quantized OlmoEarth embeddings over
one tile (a part of a UTM zone) by creating PATCH_SIZE windows, materializing
Sentinel-2 (and optionally Sentinel-1) mosaics from the OlmoEarth Datasets source,
running the model, and uploading one GeoTIFF per window to out_path. A per-tile
marker file is written to completed_path once the tile is done, recording which
crops were written and which were skipped.

Windows that don't intersect the zone's canonical wedge or that are entirely ocean
are skipped (see tiling.py). Embedding pixels where all Sentinel-2 mosaics are empty
are set to the nodata value (-128).

This module is tile-size-agnostic: it accepts any ``bounds`` whose extents are
multiples of PATCH_SIZE. The fixed 32768x32768 tiling lives only in write_jobs.py.

Note that different input variants (see EmbeddingInputs) produce different
embeddings, so each variant must use its own out_path and completed_path.
"""

import json
import multiprocessing
import shutil
import tempfile
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
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

logger = get_logger(__name__)


class EmbeddingInputs(Enum):
    """Which input modalities the embeddings are computed from."""

    S2 = "s2"
    S2_S1 = "s2_s1"


DATASET_CONFIG_FNAME = "data/large_scale_embeddings/{inputs}.json"
MODEL_CONFIG_FNAME = "data/large_scale_embeddings/{inputs}.yaml"

# Per-window size. The tile size (passed via bounds) must be a multiple of this.
PATCH_SIZE = 2048
RESOLUTION = 10

PREDICTION_GROUP = "predict"

SENTINEL2_LAYER = "sentinel2_l2a"
# Band order in the dataset config band set (used to read materialized mosaics when
# computing the validity mask).
SENTINEL2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]

OUTPUT_LAYER = "output"
EMBEDDING_DIM = 128
# The dataset configs use num_bands, for which rslearn generates band names B0, B1,
# etc.
OUTPUT_BANDS = [f"B{band_idx}" for band_idx in range(EMBEDDING_DIM)]

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
    return UPath(out_path) / f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}.tif"


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
    return UPath(completed_path) / f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}.json"


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


def _upload_window_output(
    window: Window,
    projection: Projection,
    out_fname: UPath,
) -> None:
    """Upload one window's embedding raster to the output path.

    Reads the int8 embedding raster from the scratch dataset, sets pixels where all
    Sentinel-2 mosaics are empty to NODATA_VALUE, and writes a tiled (uncompressed)
    GeoTIFF to the output path.

    Args:
        window: the window to upload.
        projection: the UTM projection (must match the window projection).
        out_fname: the output filename.
    """
    raster = window.data.read_raster(
        OUTPUT_LAYER,
        OUTPUT_BANDS,
        GeotiffRasterFormat(),
        resampling=Resampling.nearest,
    )
    embeddings = raster.get_chw_array().copy()

    # A pixel is valid if any band is nonzero in any of the Sentinel-2 mosaics.
    valid = np.zeros(embeddings.shape[1:], dtype=bool)
    for layer_name, group_idx in window.list_completed_layers():
        if layer_name != SENTINEL2_LAYER:
            continue
        s2_array = window.data.read_raster(
            SENTINEL2_LAYER,
            SENTINEL2_BANDS,
            GeotiffRasterFormat(),
            group_idx=group_idx,
            resampling=Resampling.nearest,
        ).get_chw_array()
        valid |= (s2_array != 0).any(axis=0)
    embeddings[:, ~valid] = NODATA_VALUE

    raster_format = GeotiffRasterFormat(
        always_enable_tiling=True,
        block_size=512,
        geotiff_options={"compress": "none"},
    )
    raster_format.encode_raster(
        out_fname.parent,
        projection,
        window.bounds,
        RasterArray(
            chw_array=embeddings,
            metadata=RasterMetadata(nodata_value=NODATA_VALUE),
        ),
        fname=out_fname.name,
    )


def _upload_window_by_name(
    ds_path: UPath,
    window_name: str,
    out_path: str,
) -> None:
    """Load one window from the scratch dataset and upload its embedding raster.

    This is the multiprocessing worker for the upload step; the window is reloaded by
    name so that only picklable arguments cross the process boundary.

    Args:
        ds_path: the scratch dataset path.
        window_name: the name of the window to upload.
        out_path: the output directory.
    """
    dataset = Dataset(ds_path)
    windows = dataset.load_windows(groups=[PREDICTION_GROUP], names=[window_name])
    if len(windows) != 1:
        raise ValueError(
            f"expected one window named {window_name} but got {len(windows)}"
        )
    window = windows[0]
    out_fname = get_output_fname(out_path, window.projection, window.bounds)
    _upload_window_output(window, window.projection, out_fname)


def predict_pipeline(
    inputs: EmbeddingInputs,
    projection_json: str,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    out_path: str,
    completed_path: str,
    scratch_path: str | None = None,
    upload_workers: int = 16,
) -> None:
    """Compute quantized OlmoEarth embeddings over one tile.

    Args:
        inputs: which input variant to use. Different variants produce different
            embeddings so they must use different out_path/completed_path.
        projection_json: JSON-encoded projection, normally a UTM zone with 10 m/pixel
            resolution.
        bounds: pixel coordinates within the projection on which to compute outputs.
            Each value must be a multiple of PATCH_SIZE.
        time_range: the reference timestamp as (T, T). The layer time_offset/duration
            derive the twelve monthly mosaics over the following year from this.
        out_path: directory to write one embedding GeoTIFF per PATCH_SIZE crop.
        completed_path: directory to write per-tile completion markers.
        scratch_path: optional directory to store the scratch rslearn dataset in
            directly, and keep it afterward (useful for debugging). By default, a
            temporary directory is used and deleted when the tile is done.
        upload_workers: number of worker processes for uploading the per-crop
            embedding GeoTIFFs.
    """
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
                out_path=out_path,
                marker_fname=marker_fname,
                ds_path=UPath(tmp_dir) / "dataset",
                upload_workers=upload_workers,
            )
    else:
        _process_tile(
            inputs=inputs,
            projection=projection,
            bounds=bounds,
            time_range=time_range,
            out_path=out_path,
            marker_fname=marker_fname,
            ds_path=UPath(scratch_path),
            upload_workers=upload_workers,
        )


def _process_tile(
    inputs: EmbeddingInputs,
    projection: Projection,
    bounds: PixelBounds,
    time_range: tuple[datetime, datetime],
    out_path: str,
    marker_fname: UPath,
    ds_path: UPath,
    upload_workers: int,
) -> None:
    """Process one tile using the given scratch dataset path.

    See predict_pipeline for details.

    Args:
        inputs: which input variant to use.
        projection: the projection of the tile.
        bounds: the pixel bounds of the tile.
        time_range: the reference timestamp as (T, T).
        out_path: directory to write one embedding GeoTIFF per PATCH_SIZE crop.
        marker_fname: the per-tile completion marker filename to write.
        ds_path: where to create the temporary rslearn dataset.
        upload_workers: number of worker processes for uploading the per-crop
            embedding GeoTIFFs.
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
        completed_fnames = list(
            ds_path.glob(
                f"windows/{PREDICTION_GROUP}/*/layers/{SENTINEL2_LAYER}/completed"
            )
        )
        if len(completed_fnames) == 0:
            logger.info("skipping prediction since no windows seem to have data")
        else:
            run_model_predict(model_config_fname, ds_path)

        # Upload each window's embedding raster. The uploads are handled by a pool
        # of worker processes since converting and uploading the rasters is slow. We
        # use the forkserver context because the CUDA context initialized by
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
                dict(ds_path=ds_path, window_name=window.name, out_path=out_path)
            )
            written.append(crop_offset)
        if len(upload_kwargs) > 0:
            pool = multiprocessing.get_context("forkserver").Pool(upload_workers)
            try:
                for _ in star_imap_unordered(
                    pool, _upload_window_by_name, upload_kwargs
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
