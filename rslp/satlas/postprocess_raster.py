"""Postprocessing outputs from Satlas raster models."""

import json
import multiprocessing
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import rasterio.features
import shapely
import torch
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from upath import UPath

import rslp.common.worker
from rslp.log_utils import get_logger

from .predict_pipeline import Application, projection_and_bounds_from_fname

logger = get_logger(__name__)

# Factor by which to downsample before applying Viterbi algorithm.
# This is because it is expensive to do at the full resolution.
DOWNSAMPLE_FACTOR = 1

# How many historical timesteps to use for smoothing.
NUM_HISTORICAL_TIMESTEPS = 12

# Minimum area in pixels to consider it to be valid polygon.
MIN_AREA = 8


@dataclass
class RasterSmoothConfig:
    """Configuration for smoothing."""

    num_classes: int
    # The transition probabilities for HMM used for smoothing.
    transition_probs: npt.NDArray
    # The emission (observation) probabilities for HMM used for smoothing.
    emission_probs: npt.NDArray
    # Initial state probabilities.
    initial_probs: npt.NDArray
    # Whether to enable sparse smoothing. In this case, we only apply smoothing on
    # pixels that are neither invalid (0) nor background class (1). This should remain
    # disabled for applications like tree cover that don't really have a background
    # class.
    sparse: bool = False


APP_RASTER_SMOOTH_CONFIGS = {
    Application.SOLAR_FARM: RasterSmoothConfig(
        num_classes=3,
        transition_probs=np.array(
            [
                [1, 0, 0],
                [0, 0.9, 0.1],
                [0, 0.02, 0.98],
            ],
            dtype=np.float32,
        ),
        emission_probs=np.array(
            [
                [1, 0, 0],
                [0.05, 0.75, 0.2],
                [0.05, 0.2, 0.75],
            ],
            dtype=np.float32,
        ),
        initial_probs=np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32),
        sparse=True,
    )
}


def downsample(im: torch.Tensor, factor: int) -> torch.Tensor:
    """Downsample the segmentation mask by the factor.

    Each output pixel corresponds to factor x factor pixels in the input image.

    The value of the output pixel is the mode of the corresponding input pixel values.

    Args:
        im: the HxW image to downsample.
        factor: the factor to downsample by.

    Returns:
        the downsampled image, (H/factor) x (W/factor).
    """
    if factor == 1:
        return im

    num_classes = im.max() + 1

    # Accumulate the per-class frequencies at each lower-res pixel.
    cls_counts = torch.zeros(
        (num_classes, im.shape[0] // factor, im.shape[1] // factor), dtype=torch.float32
    )
    for cls_id in range(num_classes):
        # We use pytorch avg_pool2d to compute the count of each class since it is
        # reasonably fast. The result is not actual counts but counts divided by
        # factor*factor.
        cur_counts = im == cls_id
        cur_counts = torch.nn.functional.avg_pool2d(
            cur_counts.float()[None, :, :], kernel_size=factor
        )[0, :, :]
        cls_counts[cls_id] = cur_counts

    # Now we use argmax to find the mode.
    im = cls_counts.argmax(dim=0)

    return im.byte()


def download_and_downsample(
    tile_path: UPath, projection: Projection, bounds: PixelBounds
) -> torch.Tensor:
    """Download and downsample the image stored at the specified location.

    Args:
        tile_path: the GeoTIFF filename to read.
        projection: the projection to read under.
        bounds: the bounds to read.

    Returns:
        HW tensor containing segmentation mask.
    """
    logger.info(f"Downloading image from {tile_path}")
    np_array = GeotiffRasterFormat().decode_raster(
        tile_path.parent, projection, bounds, fname=tile_path.name
    )[0, :, :]
    return downsample(torch.as_tensor(np_array), factor=DOWNSAMPLE_FACTOR)


def upload_smoothed_raster(
    dst_path: UPath, projection: Projection, bounds: PixelBounds, array: npt.NDArray
) -> None:
    """Upload the smoothed raster.

    Args:
        dst_path: path to save to.
        projection: the projection.
        bounds: the bounds.
        array: the smoothed raster data.
    """
    logger.info(f"Uploading image to {dst_path}")
    GeotiffRasterFormat().encode_raster(
        dst_path.parent, projection, bounds, array, fname=dst_path.name
    )


def smooth_rasters(
    application: Application,
    label: str,
    predict_path: str,
    smoothed_path: str,
    projection_json: str,
    bounds: PixelBounds,
    workers: int = 8,
) -> None:
    """Apply temporal smoothing for rasters at the specified tile.

    Args:
        application: the application to run smoothing for. It must have a configuration
            set in APP_RASTER_SMOOTH_CONFIGS.
        label: the YYYY-MM of the latest timestep to apply temporal smoothing.
        predict_path: the folder containing the predictions across different timesteps.
            We will check it to identify the historical timesteps to use.
        smoothed_path: where to write smoothed data.
        projection_json: the JSON-encoded projection of the tile to process.
        bounds: the bounds of the tile to process.
        workers: number of worker processes for downloading and uploading.
    """
    projection = Projection.deserialize(json.loads(projection_json))
    smooth_config = APP_RASTER_SMOOTH_CONFIGS[application]
    num_classes = smooth_config.num_classes
    transition_probs = torch.tensor(smooth_config.transition_probs)
    emission_probs = torch.tensor(smooth_config.emission_probs)
    initial_probs = torch.tensor(smooth_config.initial_probs)

    # Make sure the output doesn't already exist.
    tile_fname = f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}.tif"
    smoothed_upath = UPath(smoothed_path)
    out_fname = smoothed_upath / label / tile_fname
    if out_fname.exists():
        return

    # Get the historical timesteps that we should use for smoothing.
    # We only consider timesteps that are older than the given timestep.
    predict_upath = UPath(predict_path)
    candidate_timesteps = [
        dir_name.name
        for dir_name in predict_upath.iterdir()
        if dir_name.name <= label and (dir_name / tile_fname).exists()
    ]
    candidate_timesteps.sort()
    timesteps = candidate_timesteps[-NUM_HISTORICAL_TIMESTEPS:]
    if timesteps[-1] != label:
        raise ValueError(
            f"Expected most recent timestep to match {label}, does {predict_upath / label} not exist?"
        )

    # Download the input tiles in parallel while downsampling.
    download_args = [
        (
            predict_upath / ts / tile_fname,
            projection,
            bounds,
        )
        for ts in timesteps
    ]
    p = multiprocessing.Pool(workers)
    image_list: list[torch.Tensor] = p.starmap(
        download_and_downsample, download_args, chunksize=1
    )
    # Stack to form THW tensor.
    images = torch.stack(image_list, dim=0)
    # We manually delete because the memory usage is actually significant.
    del image_list

    if smooth_config.sparse:
        # For sparse smoothing, we select pixels where the prediction is >=2 at some
        # point along the time series (we use >=2 since we assume 0=invalid and
        # 1=background). We flatten the pixels and save it back to images, but keep
        # sel_indexes around to support unflatenning later.
        # sel_indexes (N, 2) contains the row and column for each non-empty pixel.
        logger.info("Selecting non-invalid/background pixels")
        orig_images_shape = images.shape
        sel_indexes = (images.amax(dim=0) >= 2).nonzero()
        sparse_images = images[:, sel_indexes[:, 0], sel_indexes[:, 1]]
        # We create a fake images here for the sparse pixels where we use height
        # dimension for the number of pixels and width dimension is 1.
        images = sparse_images[:, :, None]
        logger.info(f"Extracted {sel_indexes.shape[0]} sparse indices")

    # Apply viterbi over the images.
    # probs contains the state probabilities computed so far, while pointers contains
    # all the backpointers to reconstruct the state sequence later.
    logger.info("Applying Viterbi forward pass")
    probs = torch.zeros(
        (images.shape[1], images.shape[2], num_classes), dtype=torch.float32
    )
    probs[:, :, :] = initial_probs
    pointers = torch.zeros(
        (len(images), images[0].shape[0], images[0].shape[1], num_classes),
        dtype=torch.uint8,
    )
    for im_idx, im in enumerate(images):
        logger.info(f"Viterbi step {im_idx}")
        # Compute a matrix obs[i, j, state] indicating the probability of the current observation im[i, j] for that state.
        # obs[i, j] = emission_probs[:, im[i, j]].
        obs = (
            emission_probs.index_select(dim=1, index=im.flatten().int())
            .reshape(emission_probs.shape[0], im.shape[0], im.shape[1])
            .permute(1, 2, 0)
        )

        # Compute a matrix trans[i, j, prev_state, state] indicating probability for being in prev_state and then transitioning to state.
        # trans[i, j, prev_state, state] = probs[i, j, prev_state] * transition_probs[prev_state, state]
        trans = probs[:, :, :, None] * transition_probs

        probs = trans.amax(dim=2) * obs
        pointers[im_idx, :, :, :] = trans.argmax(dim=2)

    # Follow pointers.
    # cur_output contains the states for the current timestep, while we collect the
    # reversed sequence of states into outputs (and unreverse at the end).
    logger.info("Applying Viterbi backward pass (following pointers)")
    cur_output = probs[:, :, :].argmax(dim=2)
    outputs: list | torch.Tensor = [cur_output.byte()]
    for im_idx in range(images.shape[0] - 1, 0, -1):
        # Set cur_output[i, j] = pointers[im_idx, i, j, cur_output[i, j]].
        cur_output = pointers[im_idx].int()[
            torch.arange(cur_output.shape[0])[:, None],
            torch.arange(cur_output.shape[1]),
            cur_output,
        ]
        outputs.append(cur_output.byte())
    outputs = list(reversed(outputs))

    del probs
    del pointers

    if smooth_config.sparse:
        # Now we unflatten the sparse pixels.
        # First stack the outputs and drop singleton fake width dimension.
        sparse_outputs = torch.stack(outputs, dim=0)[:, :, 0]
        # Now collect it into the actual 3D array using the sel_indexes
        outputs = torch.zeros(orig_images_shape, dtype=sparse_outputs.dtype)
        outputs[:, sel_indexes[:, 0], sel_indexes[:, 1]] = sparse_outputs

    # Compute the projection and bounds for the output, which is at a lower resolution.
    downsampled_projection = Projection(
        projection.crs,
        projection.x_resolution * DOWNSAMPLE_FACTOR,
        projection.y_resolution * DOWNSAMPLE_FACTOR,
    )
    downsampled_bounds = (
        bounds[0] // DOWNSAMPLE_FACTOR,
        bounds[1] // DOWNSAMPLE_FACTOR,
        bounds[2] // DOWNSAMPLE_FACTOR,
        bounds[3] // DOWNSAMPLE_FACTOR,
    )

    # Upload results in parallel.
    logger.info("Uploading outputs")
    upload_args = [
        (
            smoothed_upath / ts / tile_fname,
            downsampled_projection,
            downsampled_bounds,
            cur_output.numpy()[None, :, :],
        )
        for cur_output, ts in zip(outputs, timesteps)
    ]
    p.starmap(upload_smoothed_raster, upload_args)
    p.close()
    logger.info("Done smoothing rasters")


def write_smooth_rasters_jobs(
    application: Application,
    label: str,
    predict_path: str,
    smoothed_path: str,
    queue_name: str,
) -> None:
    """Write jobs to run smooth_rasters.

    Args:
        application: see smooth_rasters.
        label: see smooth_rasters.
        predict_path: see smooth_rasters.
        smoothed_path: see smooth_rasters.
        queue_name: the Beaker queue to write the jobs to.
    """
    predict_upath = UPath(predict_path)
    smoothed_upath = UPath(smoothed_path)

    # Get completed filename set from smoothed dir.
    completed_fnames = set()
    for fname in (smoothed_upath / label).iterdir():
        completed_fnames.add(fname.name)

    # Now create one job for each filename in the prediction dir.
    jobs: list[list[str]] = []
    for fname in (predict_upath / label).iterdir():
        if not fname.name.endswith(".tif"):
            continue
        if fname.name in completed_fnames:
            continue

        projection, bounds = projection_and_bounds_from_fname(fname.name)
        jobs.append(
            [
                "--application",
                application.value.upper(),
                "--label",
                label,
                "--predict_path",
                predict_path,
                "--smoothed_path",
                smoothed_path,
                "--projection",
                json.dumps(projection.serialize()),
                "--bounds",
                json.dumps(bounds),
            ]
        )

    rslp.common.worker.write_jobs(queue_name, "satlas", "smooth_rasters", jobs)


def extract_polygons(
    label: str,
    smoothed_path: str,
    vectorized_path: str,
    projection_json: str,
    bounds: PixelBounds,
) -> None:
    """Vectorize the smoothed raster outputs into polygons.

    Args:
        application: the application.
        label: YYYY-MM representation of the time range used for this prediction run.
        smoothed_path: path where smoothed GeoTIFFs have been written.
        vectorized_path: folder to write GeoJSONs containing vectorized features.
        workers: number of worker processes.
        projection_json: the JSON-encoded projection of the tile to process.
        bounds: the bounds of the tile to process.
    """
    # Make sure job isn't already done.
    projection = Projection.deserialize(json.loads(projection_json))
    tile_prefix = f"{str(projection.crs)}_{bounds[0]}_{bounds[1]}"
    out_fname = UPath(vectorized_path) / label / f"{tile_prefix}.geojson"
    if out_fname.exists():
        return

    # Get the segmentation mask after temporal smoothing.
    raster_format = GeotiffRasterFormat()
    downsampled_projection = Projection(
        projection.crs,
        projection.x_resolution * DOWNSAMPLE_FACTOR,
        projection.y_resolution * DOWNSAMPLE_FACTOR,
    )
    downsampled_bounds = (
        bounds[0] // DOWNSAMPLE_FACTOR,
        bounds[1] // DOWNSAMPLE_FACTOR,
        bounds[2] // DOWNSAMPLE_FACTOR,
        bounds[3] // DOWNSAMPLE_FACTOR,
    )
    in_fname = UPath(smoothed_path) / label / f"{tile_prefix}.tif"
    logger.info("Downloading smoothed raster from %s", in_fname)
    array = raster_format.decode_raster(
        in_fname.parent, downsampled_projection, downsampled_bounds, fname=in_fname.name
    )

    # Set value 1 (assumed to be background) to 0 for simplicity.
    array[array == 1] = 0

    # Vectorize the array.
    logger.info("Extracting polygons")
    shapes = rasterio.features.shapes(array)
    features = []
    for shp, value in shapes:
        if value == 0:
            continue

        # Simplify the polygon and offset it by the bounds.
        shp = shapely.geometry.shape(shp)
        if shp.area < MIN_AREA:
            continue
        shp = shp.simplify(tolerance=2)
        shp = shapely.transform(
            shp,
            lambda coords: coords
            + np.array([downsampled_bounds[0], downsampled_bounds[1]]),
        )
        geom = STGeometry(downsampled_projection, shp, None)
        features.append(Feature(geom, {"value": value}))

    # Write output features.
    logger.info("Writing %d output features to %s", len(features), out_fname)
    vector_format = GeojsonVectorFormat(coordinate_mode=GeojsonCoordinateMode.WGS84)
    vector_format.encode_to_file(out_fname, features)


def write_extract_polygons_jobs(
    label: str,
    smoothed_path: str,
    vectorized_path: str,
    queue_name: str,
) -> None:
    """Write jobs to run extract_polygons.

    Args:
        label: see extract_polygons.
        smoothed_path: see extract_polygons.
        vectorized_path: see extract_polygons.
        queue_name: the Beaker queue to write the jobs to.
    """
    smoothed_upath = UPath(smoothed_path)
    vectorized_upath = UPath(vectorized_path)

    # Get completed filename set from vectorized dir.
    completed_fnames = set()
    for fname in (vectorized_upath / label).iterdir():
        completed_fnames.add(fname.name.split(".")[0])

    # Now create one job for each filename in the prediction dir.
    jobs: list[list[str]] = []
    for fname in (smoothed_upath / label).iterdir():
        if not fname.name.endswith(".tif"):
            continue
        if fname.name.split(".")[0] in completed_fnames:
            continue

        projection, bounds = projection_and_bounds_from_fname(fname.name)
        jobs.append(
            [
                "--label",
                label,
                "--smoothed_path",
                smoothed_path,
                "--vectorized_path",
                vectorized_path,
                "--projection",
                json.dumps(projection.serialize()),
                "--bounds",
                json.dumps(bounds),
            ]
        )

    rslp.common.worker.write_jobs(queue_name, "satlas", "extract_polygons", jobs)
