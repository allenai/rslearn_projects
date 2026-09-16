"""NISAR vessel prediction pipeline."""

import json
import math
import tempfile
from dataclasses import dataclass

import numpy as np
import shapely
from PIL import Image
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.log_utils import get_logger
from rslp.nisar_vessels.config import (
    INFRA_DISTANCE_THRESHOLD_KM,
    MARINE_INFRA_PATH,
    NUM_DATA_LOADER_WORKERS,
    NUM_MATERIALIZE_WORKERS,
    PREDICT_CROP_SIZE,
    PREDICT_OVERLAP_PIXELS,
    SCENE_TILE_OVERLAP,
    SCENE_TILE_SIZE,
)
from rslp.nisar_vessels.hdf5 import GranuleGrid, granule_to_geotiff
from rslp.nisar_vessels.prom_metrics import TimerOperations, time_operation
from rslp.utils.filter import NearInfraFilter
from rslp.utils.rslearn import (
    ApplyWindowsArgs,
    IngestArgs,
    MaterializeArgs,
    MaterializePipelineArgs,
    PrepareArgs,
    materialize_dataset,
    run_model_predict,
)
from rslp.vessels import VesselDetection, VesselDetectionSource

logger = get_logger(__name__)

NISAR_LAYER_NAME = "nisar"
OUTPUT_LAYER_NAME = "output"
WINDOW_GROUP = "detector_predict"

DATASET_CONFIG = "data/nisar_vessels/config_predict.json"
DETECT_MODEL_CONFIG = "data/nisar_vessels/config_satlas.yaml"

# Resolution of the windows the detector runs on. GCOV granules are posted at 10 m or
# 20 m, and the training windows were all built at 10 m/pixel, so everything is
# resampled to 10 m here too.
RESOLUTION = 10

# The GCOV diagonal covariance terms the detector reads, in the order the model expects
# them. HHHH and HVHV are the HH and HV backscatter intensities, which every dual-pol
# H-transmit (POLE "DH") acquisition carries.
BAND_NAMES = ["HHHH", "HVHV"]

# Band name to the key it appears under in VesselDetection.crop_fnames. The sat service
# reads the crops back out of the response by these names.
CROP_KEYS = {"HHHH": "hh", "HVHV": "hv"}

# Crop side length in pixels, i.e. 1.28 km at 10 m.
CROP_WINDOW_SIZE = 128

# Display range for the crops, covering where vessels and open water actually sit. GCOV
# carries gamma-0 in linear power, so it is converted to decibels first, as in training.
CROP_DECIBEL_RANGE = (-35.0, 5.0)

# Matches the epsilon in rslearn's Sentinel1ToDecibels, so a zero or filled pixel lands
# at -60 dB rather than negative infinity.
DECIBEL_EPSILON = 1e-6

# Detections this far apart, in pixels, are treated as the same vessel seen from two
# overlapping scene tiles. Matches the distance threshold the model config's merger uses
# to combine detections across the crops within a single window.
DEDUPE_DISTANCE_PIXELS = 10


@dataclass(frozen=True)
class PredictionTask:
    """A task to predict vessels in one NISAR scene.

    Args:
        h5_path: local path of the NISAR HDF5 granule to detect vessels in. This is
            the only way the sidecar can be given imagery; it has no data source of its
            own to look a granule up with.
        scene_id: the granule name. Defaults to the filename of h5_path, so a caller
            that only has a path does not have to supply one.
        json_path: optional path to write the JSON of vessel detections.
        crop_path: optional path to write the vessel crop images.
        geojson_path: optional path to write GeoJSON of detections.
    """

    h5_path: str
    scene_id: str | None = None
    json_path: str | None = None
    crop_path: str | None = None
    geojson_path: str | None = None

    def get_scene_id(self) -> str:
        """Get the granule name for this task.

        Returns:
            the scene ID the caller gave, otherwise the granule filename without its
            extension. The sat service names the downloaded file after the granule, so
            the stem is the granule name.
        """
        if self.scene_id is not None:
            return self.scene_id
        return UPath(self.h5_path).stem


@dataclass(frozen=True)
class SceneData:
    """Where in the world one scene sits, and where its imagery was staged.

    Args:
        scene_id: the granule name.
        projection: the projection the scene's windows are created in.
        bounds: the whole scene's bounds in that projection.
        geotiff_path: the GeoTIFF the granule was converted to. Crops are read back from
            here rather than from a tile, so a detection near a tile seam still gets a
            full crop.
    """

    scene_id: str
    projection: Projection
    bounds: PixelBounds
    geotiff_path: str


def setup_dataset(ds_path: UPath, tasks: list[PredictionTask]) -> list[SceneData]:
    """Initialize an rslearn dataset for prediction from the granules in the tasks.

    Each granule is converted to a GeoTIFF up front, since GDAL cannot georeference the
    HDF5 datasets on its own. From there the layer is an ordinary LocalFiles raster.

    Args:
        ds_path: the dataset path to write to.
        tasks: the prediction tasks to process.

    Returns:
        a list of SceneData corresponding to the list of tasks.
    """
    with open(DATASET_CONFIG) as f:
        ds_cfg = json.load(f)

    # src_dir only ends up holding LocalFiles' summary.json, since the GeoTIFFs are
    # named by absolute URI in the item specs.
    src_dir = ds_path / "source_dir"
    src_dir.mkdir(parents=True, exist_ok=True)
    geotiff_dir = ds_path / "geotiffs"
    geotiff_dir.mkdir(parents=True, exist_ok=True)

    item_specs = []
    scene_datas: list[SceneData] = []
    for task in tasks:
        scene_id = task.get_scene_id()
        geotiff_path = geotiff_dir / f"{scene_id}.tif"
        grid = granule_to_geotiff(task.h5_path, BAND_NAMES, str(geotiff_path))

        item_specs.append(
            {
                "fnames": [geotiff_path.absolute().as_uri()],
                "bands": [BAND_NAMES],
                "name": scene_id,
            }
        )
        scene_datas.append(_get_scene_data(scene_id, grid, str(geotiff_path)))

    layer_cfg = ds_cfg["layers"][NISAR_LAYER_NAME]["data_source"]["init_args"]
    layer_cfg["raster_item_specs"] = item_specs
    layer_cfg["src_dir"] = src_dir.name
    with (ds_path / "config.json").open("w") as f:
        json.dump(ds_cfg, f)

    return scene_datas


def _get_scene_data(scene_id: str, grid: GranuleGrid, geotiff_path: str) -> SceneData:
    """Place a granule's footprint on the grid the detector runs on.

    GCOV is already geocoded, but not necessarily in the UTM/UPS zone the training
    windows used, so the footprint is re-derived the same way training did it: pick the
    zone from the scene centroid and take the bounds there.

    Args:
        scene_id: the granule name.
        grid: the grid the granule's bands are sampled on.
        geotiff_path: the GeoTIFF the granule was converted to.

    Returns:
        the SceneData for this granule.
    """
    src_geom = STGeometry(
        # Resolution 1 makes "pixel" coordinates equal to CRS coordinates, which is what
        # the grid's bounds are expressed in.
        Projection(grid.crs, 1, 1),
        shapely.box(*grid.bounds),
        None,
    )
    wgs84_geom = src_geom.to_projection(WGS84_PROJECTION)
    projection = get_utm_ups_projection(
        wgs84_geom.shp.centroid.x,
        wgs84_geom.shp.centroid.y,
        RESOLUTION,
        -RESOLUTION,
    )
    minx, miny, maxx, maxy = src_geom.to_projection(projection).shp.bounds
    return SceneData(
        scene_id=scene_id,
        projection=projection,
        bounds=(
            math.floor(minx),
            math.floor(miny),
            math.ceil(maxx),
            math.ceil(maxy),
        ),
        geotiff_path=geotiff_path,
    )


def tile_scene_bounds(
    bounds: PixelBounds, tile_size: int, overlap: int
) -> list[PixelBounds]:
    """Split a scene's bounds into overlapping tiles of at most tile_size.

    Args:
        bounds: the whole scene's bounds.
        tile_size: the side length of each tile in pixels.
        overlap: how much adjacent tiles share, so a vessel on a seam falls fully inside
            at least one of them.

    Returns:
        the tile bounds, row-major.
    """
    minx, miny, maxx, maxy = bounds
    return [
        (x, y, min(x + tile_size, maxx), min(y + tile_size, maxy))
        for y in _tile_starts(miny, maxy, tile_size, overlap)
        for x in _tile_starts(minx, maxx, tile_size, overlap)
    ]


def _tile_starts(low: int, high: int, tile_size: int, overlap: int) -> list[int]:
    """Get the tile start offsets covering one axis.

    The last tile is flush against the far edge rather than clipped, so every tile is
    full size and the final one simply overlaps its neighbour more than the rest.

    Args:
        low: the first coordinate on this axis.
        high: one past the last coordinate on this axis.
        tile_size: the side length of each tile in pixels.
        overlap: how much adjacent tiles share.

    Returns:
        the start offset of each tile.
    """
    if high - low <= tile_size:
        return [low]
    stride = tile_size - overlap
    starts = list(range(low, high - tile_size, stride))
    starts.append(high - tile_size)
    return starts


def materialize_scenes(
    ds_path: UPath, scene_datas: list[SceneData]
) -> list[tuple[int, Window]]:
    """Create a window per scene tile and materialize the NISAR imagery into each.

    Scenes are split into tiles rather than materialized whole: rslearn builds a
    window's raster as a single in-memory array, so one window per granule makes peak
    memory scale with the granule, which is unbounded. Tiling caps it at the tile size
    no matter how large the scene is.

    Args:
        ds_path: the dataset path, already configured by :func:`setup_dataset`.
        scene_datas: the SceneDatas to create windows for.

    Returns:
        (scene index, window) for every tile, so detections can be traced back to the
        scene they came from.

    Raises:
        ValueError: if a window's NISAR layer could not be materialized.
    """
    dataset = Dataset(ds_path)
    tiles: list[tuple[int, Window]] = []
    for scene_idx, scene_data in enumerate(scene_datas):
        tile_bounds = tile_scene_bounds(
            scene_data.bounds, SCENE_TILE_SIZE, SCENE_TILE_OVERLAP
        )
        logger.info(
            f"Split scene {scene_data.scene_id} into {len(tile_bounds)} tile(s) "
            f"of up to {SCENE_TILE_SIZE}px"
        )
        for tile_idx, bounds in enumerate(tile_bounds):
            window = Window(
                storage=dataset.storage,
                group=WINDOW_GROUP,
                name=f"{scene_idx}_{tile_idx}",
                projection=scene_data.projection,
                bounds=bounds,
                # A granule is a single acquisition and the detector needs no other
                # imagery, so there is nothing for a time range to select between.
                time_range=None,
                data_factory=dataset.window_data_storage_factory,
            )
            window.save()
            tiles.append((scene_idx, window))

    logger.info("Materialize dataset for NISAR vessel detection")
    apply_windows_args = ApplyWindowsArgs(
        group=WINDOW_GROUP, workers=NUM_MATERIALIZE_WORKERS
    )
    materialize_pipeline_args = MaterializePipelineArgs(
        disabled_layers=[],
        prepare_args=PrepareArgs(apply_windows_args=apply_windows_args),
        ingest_args=IngestArgs(
            ignore_errors=False, apply_windows_args=apply_windows_args
        ),
        materialize_args=MaterializeArgs(
            ignore_errors=False, apply_windows_args=apply_windows_args
        ),
    )
    with time_operation(TimerOperations.MaterializeDataset):
        materialize_dataset(ds_path, materialize_pipeline_args)

    for scene_idx, window in tiles:
        if not window.is_layer_completed(NISAR_LAYER_NAME):
            raise ValueError(
                f"window {window.name} does not have the NISAR layer completed for "
                f"scene {scene_datas[scene_idx].scene_id}"
            )

    return tiles


def get_vessel_detections(
    ds_path: UPath,
    scene_datas: list[SceneData],
    score_threshold: float,
) -> list[VesselDetection]:
    """Apply the vessel detector.

    The caller is responsible for setting up the dataset configuration that will obtain
    the NISAR images.

    Args:
        ds_path: the dataset path that will be populated with the tile windows to apply
            the detector to.
        scene_datas: the SceneDatas to apply the detector on.
        score_threshold: override the detector's configured score threshold for this
            run, so callers can raise or lower the cutoff without editing the config.

    Returns:
        the detections, recording the scene each came from in their task_idx metadata.
    """
    tiles = materialize_scenes(ds_path, scene_datas)

    with time_operation(TimerOperations.RunModelPredict):
        extra_args = [
            "--data.init_args.num_workers",
            str(NUM_DATA_LOADER_WORKERS),
            "--data.init_args.predict_config.crop_size",
            str(PREDICT_CROP_SIZE),
            "--data.init_args.predict_config.overlap_pixels",
            str(PREDICT_OVERLAP_PIXELS),
            "--data.init_args.task.init_args.tasks.detect.init_args.score_threshold",
            str(score_threshold),
        ]
        run_model_predict(
            DETECT_MODEL_CONFIG, ds_path, groups=[WINDOW_GROUP], extra_args=extra_args
        )

    detections: list[VesselDetection] = []
    for scene_idx, window in tiles:
        features = window.data.read_vector(OUTPUT_LAYER_NAME, GeojsonVectorFormat())
        for feature in features:
            geometry = feature.geometry
            detections.append(
                VesselDetection(
                    source=VesselDetectionSource.NISAR,
                    col=int(geometry.shp.centroid.x),
                    row=int(geometry.shp.centroid.y),
                    projection=geometry.projection,
                    score=feature.properties["score"],
                    scene_id=scene_datas[scene_idx].scene_id,
                    # We use this metadata to keep track of which scene each detection
                    # came from.
                    metadata={"task_idx": scene_idx},
                )
            )

    return dedupe_detections(detections)


def dedupe_detections(detections: list[VesselDetection]) -> list[VesselDetection]:
    """Drop repeat detections of one vessel found in two overlapping tiles.

    Tiles overlap so that a vessel on a seam is fully inside at least one of them, which
    means anything in the overlap band is detected twice. All tiles of a scene share its
    projection, so their pixel coordinates are directly comparable.

    Args:
        detections: the detections from every tile.

    Returns:
        the detections with near-duplicates from the same scene removed, keeping the
        highest scoring of each group.
    """
    kept: list[VesselDetection] = []
    kept_by_scene: dict[int, list[VesselDetection]] = {}
    # Highest scoring first, so the survivor of each duplicate pair is the best one.
    for detection in sorted(detections, key=lambda d: d.score, reverse=True):
        scene_idx = detection.metadata["task_idx"]
        neighbors = kept_by_scene.setdefault(scene_idx, [])
        # Euclidean, matching how the merger combines detections across the crops
        # within a window, so the two stages agree on what counts as the same vessel.
        if any(
            math.hypot(detection.col - other.col, detection.row - other.row)
            <= DEDUPE_DISTANCE_PIXELS
            for other in neighbors
        ):
            continue
        neighbors.append(detection)
        kept.append(detection)

    if len(kept) < len(detections):
        logger.info(
            f"Dropped {len(detections) - len(kept)} detection(s) duplicated across "
            "tile seams"
        )
    return kept


def predict_pipeline(
    tasks: list[PredictionTask],
    score_threshold: float,
    scratch_path: str | None = None,
) -> list[list[VesselDetection]]:
    """Run the NISAR vessel prediction pipeline.

    Given a NISAR granule, the pipeline produces the vessel detections. Specifically, it
    outputs a list of the vessel detection locations along with crops of each detection.

    This is a plain detector rather than a change detector, so a granule is all it
    needs, and it predicts no vessel attributes.

    Args:
        tasks: prediction tasks to execute.
        score_threshold: override the detector's score threshold for this run, replacing
            the value baked into the model config.
        scratch_path: directory to use to store temporary dataset.

    Returns:
        list of vessel detections for each task.
    """
    if len(tasks) == 0:
        return []

    if scratch_path is None:
        tmp_dir = tempfile.TemporaryDirectory()
        scratch_path = tmp_dir.name

    ds_path = UPath(scratch_path)
    ds_path.mkdir(parents=True, exist_ok=True)

    with time_operation(TimerOperations.SetupDataset):
        scene_datas = setup_dataset(ds_path, tasks)

    with time_operation(TimerOperations.GetVesselDetections):
        detections = get_vessel_detections(
            ds_path, scene_datas, score_threshold=score_threshold
        )

    with time_operation(TimerOperations.BuildPredictionsAndCrops):
        detections_by_task = _build_predictions_and_crops(
            detections, scene_datas, tasks
        )

    for task, task_detections in zip(tasks, detections_by_task):
        if task.json_path is not None:
            json_upath = UPath(task.json_path)
            json_upath.parent.mkdir(parents=True, exist_ok=True)
            with json_upath.open("w") as f:
                json.dump([d.to_dict() for d in task_detections], f)

        if task.geojson_path is not None:
            geojson_upath = UPath(task.geojson_path)
            geojson_upath.parent.mkdir(parents=True, exist_ok=True)
            with geojson_upath.open("w") as f:
                json.dump(
                    {
                        "type": "FeatureCollection",
                        "properties": {},
                        "features": [d.to_feature() for d in task_detections],
                    },
                    f,
                )

    return detections_by_task


def _build_predictions_and_crops(
    detections: list[VesselDetection],
    scene_datas: list[SceneData],
    tasks: list[PredictionTask],
) -> list[list[VesselDetection]]:
    """Filter the detections and save a crop image per band for each one.

    Args:
        detections: the detections from the detector.
        scene_datas: the SceneData per task, indexed by a detection's task_idx.
        tasks: the prediction tasks, indexed by a detection's task_idx.

    Returns:
        the kept detections, grouped by task.
    """
    detections_by_task: list[list[VesselDetection]] = [[] for _ in tasks]

    near_infra_filter = NearInfraFilter(
        infra_path=MARINE_INFRA_PATH,
        infra_distance_threshold=INFRA_DISTANCE_THRESHOLD_KM,
    )
    for detection in detections:
        # Apply near infra filter (True -> filter out, False -> keep).
        lon, lat = detection.get_lon_lat()
        if near_infra_filter.should_filter(lon, lat):
            continue

        task_idx = detection.metadata["task_idx"]
        task = tasks[task_idx]

        if task.crop_path is not None:
            detection.crop_fnames = _write_crops(
                detection, scene_datas[task_idx], UPath(task.crop_path)
            )

        detections_by_task[task_idx].append(detection)

    return detections_by_task


def _write_crops(
    detection: VesselDetection, scene_data: SceneData, crop_upath: UPath
) -> dict[str, UPath]:
    """Save one PNG crop per band around a detection.

    Crops come from the scene's GeoTIFF rather than the tile window the detection was
    found in, so a detection near a tile seam still gets a full crop instead of one half
    filled with nodata. The read is a windowed one, so it costs the crop, not the scene.

    Args:
        detection: the detection to crop around.
        scene_data: the scene the detection was found in.
        crop_upath: the directory to write the crops to.

    Returns:
        map from crop key (see CROP_KEYS) to the file the crop was written to.
    """
    crop_upath.mkdir(parents=True, exist_ok=True)
    half_size = CROP_WINDOW_SIZE // 2
    bounds: PixelBounds = (
        detection.col - half_size,
        detection.row - half_size,
        detection.col + half_size,
        detection.row + half_size,
    )

    # The GeoTIFF holds both bands, so one read returns the whole stack. Bounds that run
    # off the edge of the scene are filled with the raster's nodata value.
    geotiff_path = UPath(scene_data.geotiff_path)
    image = (
        GeotiffRasterFormat()
        .decode_raster(
            geotiff_path.parent,
            scene_data.projection,
            bounds,
            fname=geotiff_path.name,
        )
        .get_chw_array()
    )

    crop_fnames: dict[str, UPath] = {}
    for band_idx, band_name in enumerate(BAND_NAMES):
        crop_fname = crop_upath / f"{detection.col}_{detection.row}_{band_name}.png"
        with crop_fname.open("wb") as f:
            Image.fromarray(_to_uint8(image[band_idx, :, :])).save(f, format="PNG")
        crop_fnames[CROP_KEYS[band_name]] = crop_fname

    return crop_fnames


def _to_uint8(band_image: np.ndarray) -> np.ndarray:
    """Stretch linear backscatter over the display range.

    Args:
        band_image: one band of a crop, in linear power.

    Returns:
        the same crop as uint8.
    """
    # NaN is GCOV's nodata, and survives the log as NaN, so clear it first.
    linear = np.nan_to_num(band_image, nan=0.0, posinf=0.0, neginf=0.0)
    decibels = 10 * np.log10(np.clip(linear, DECIBEL_EPSILON, None))
    min_db, max_db = CROP_DECIBEL_RANGE
    scaled = (decibels - min_db) / (max_db - min_db) * 255
    return np.clip(scaled, 0, 255).astype(np.uint8)
