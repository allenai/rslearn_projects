"""Extract Forest Loss Alerts for Forest Loss Driver Prediction Pipeline."""

import io
import json
import math
import multiprocessing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import fiona
import numpy as np
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.ops
import tqdm
from rasterio.crs import CRS
from rslearn.const import SHAPEFILE_AUX_EXTENSIONS, WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry
from rslearn.utils.fsspec import get_upath_local
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import SingleImageRasterFormat
from upath import UPath

from rslp.forest_loss_driver.const import GROUP
from rslp.forest_loss_driver.inference.config import ExtractAlertsArgs
from rslp.log_utils import get_logger

logger = get_logger(__name__)

# Time corresponding to 0 in alertDate GeoTIFF files.
BASE_DATETIME = datetime(2019, 1, 1, tzinfo=timezone.utc)

# How big the rslearn windows should be.
WINDOW_SIZE = 128

# Create windows at WebMercator zoom 13 (512x512 tiles).
WEB_MERCATOR_CRS = CRS.from_epsg(3857)
WEB_MERCATOR_M = 2 * math.pi * 6378137
PIXEL_SIZE = WEB_MERCATOR_M / (2**13) / 512
WEB_MERCATOR_PROJECTION = Projection(WEB_MERCATOR_CRS, PIXEL_SIZE, -PIXEL_SIZE)

ANNOTATION_WEBSITE_MERCATOR_OFFSET = 512 * (2**12)
INFERENCE_DATASET_CONFIG = str(
    Path(__file__).resolve().parents[3] / "data" / "forest_loss_driver" / "config.json"
)

# Filename used to indicate that alert extraction is done for a given dataset.
COMPLETED_FNAME = "extract_alerts_completed"


class ForestLossEvent:
    """Details about a forest loss event."""

    def __init__(
        self,
        polygon_geom: STGeometry,
        center_geom: STGeometry,
        center_pixel: tuple[int, int],
        ts: datetime,
    ) -> None:
        """Create a new ForestLossEvent.

        Args:
            polygon_geom: the polygon specifying the connected component, in projection
                coordinates.
            center_geom: the center of the polygon (clipped to the polygon) in
                projection coordinates.
            center_pixel: the center in pixel coordinates.
            ts: the timestamp of the forest loss event based on img_point.
        """
        self.polygon_geom = polygon_geom
        self.center_geom = center_geom
        self.center_pixel = center_pixel
        self.ts = ts


def output_forest_event_metadata(
    event: ForestLossEvent,
    fname: str,
    ds_path: UPath,
    mercator_point: tuple[int, int],
    projection: Projection,
) -> None:
    """Output the info.json metadata for a forest loss event."""
    polygon_wgs84_shp: shapely.Polygon = event.polygon_geom.to_projection(
        projection
    ).shp
    with (ds_path / "info.json").open("w") as f:
        json.dump(
            {
                "fname": fname,
                "img_point": event.center_pixel,
                "mercator_point": mercator_point,
                "date": event.ts.isoformat(),
                "wkt": polygon_wgs84_shp.wkt,
            },
            f,
        )


def output_mask_raster(
    event: ForestLossEvent,
    ds_path: UPath,
    bounds: tuple[int, int, int, int],
    window: Window,
    projection: Projection,
    window_size: int = WINDOW_SIZE,
) -> None:
    """Output the mask raster for a forest loss event."""
    # Get pixel coordinates of the mask.
    polygon_webm_shp = event.polygon_geom.to_projection(projection).shp

    def to_out_pixel(points: np.ndarray) -> np.ndarray:
        points[:, 0] -= bounds[0]
        points[:, 1] -= bounds[1]
        return points

    polygon_out_shp = shapely.transform(polygon_webm_shp, to_out_pixel)
    mask_im = rasterio.features.rasterize(
        [(polygon_out_shp, 255)],
        out_shape=(window_size, window_size),
        # If out is not provided, dtype is required according to the docs.
        dtype=np.int32,
    )
    layer_dir = ds_path / "layers" / "mask"
    SingleImageRasterFormat().encode_raster(
        layer_dir / "mask",
        window.projection,
        window.bounds,
        mask_im[None, :, :],
    )
    # Should this happen every time we encode the raster
    with (layer_dir / "completed").open("w"):
        pass


def output_window_metadata(event: ForestLossEvent, ds_path: UPath) -> tuple:
    """Output the window metadata."""
    # Transform the center to Web-Mercator for populating the window.
    center_webm_shp = event.center_geom.to_projection(WEB_MERCATOR_PROJECTION).shp

    # WebMercator point to include in the name.
    # The annotation website will use this so we have adjusted this to have the same offset as multisat/other code.
    mercator_point = (
        int(center_webm_shp.x) + ANNOTATION_WEBSITE_MERCATOR_OFFSET,
        int(center_webm_shp.y) + ANNOTATION_WEBSITE_MERCATOR_OFFSET,
    )

    # While the bounds is for rslearn.
    bounds = (
        int(center_webm_shp.x) - WINDOW_SIZE // 2,
        int(center_webm_shp.y) - WINDOW_SIZE // 2,
        int(center_webm_shp.x) + WINDOW_SIZE // 2,
        int(center_webm_shp.y) + WINDOW_SIZE // 2,
    )
    time_range = (
        event.ts,
        event.ts + timedelta(days=1),
    )

    # Create the new rslearn windows.
    window_name = f"feat_x_{mercator_point[0]}_{mercator_point[1]}_{event.center_pixel[0]}_{event.center_pixel[1]}"
    window_path = ds_path / "windows" / GROUP / window_name
    window = Window(
        path=window_path,
        group=GROUP,
        name=window_name,
        projection=WEB_MERCATOR_PROJECTION,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()
    return window, window_path, mercator_point, bounds


def write_event(event: ForestLossEvent, fname: str, ds_path: UPath) -> None:
    """Write a window for this forest loss event.

    This function creates several output files for each forest loss event:
    1. Saves the window metadata to ds_path/windows/default/feat_x_.../ directory.
    2. Saves the info.json metadata to the same directory.
    3. Saves the mask image to ds_path/windows/default/feat_x_.../layers/mask/
    4. Creates an empty 'completed' file to indicate successful processing.

    Args:
        event: the event details.
        fname: the GeoTIFF filename that this alert came from.
        ds_path: the path of dataset to write to.
    """
    window, window_path, mercator_point, bounds = output_window_metadata(event, ds_path)

    output_forest_event_metadata(
        event, fname, window_path, mercator_point, WGS84_PROJECTION
    )

    output_mask_raster(event, window_path, bounds, window, WEB_MERCATOR_PROJECTION)


def load_country_polygon(country_data_path: UPath) -> shapely.Polygon:
    """Load the country polygon.

    Please make sure the necessary AUX Shapefiles are in the same directory as the
    main Shapefile.
    """
    logger.info(f"loading country polygon from {country_data_path}")
    prefix = ".".join(country_data_path.name.split(".")[:-1])
    aux_files: list[UPath] = []
    for ext in SHAPEFILE_AUX_EXTENSIONS:
        aux_files.append(country_data_path.parent / (prefix + ext))
    country_wgs84_shp: shapely.Polygon | None = None
    with get_upath_local(country_data_path, extra_paths=aux_files) as local_fname:
        with fiona.open(local_fname) as src:
            for feat in src:
                if feat["properties"]["ISO_A2"] != "PE":
                    continue
                cur_shp = shapely.geometry.shape(feat["geometry"])
                if country_wgs84_shp:
                    country_wgs84_shp = country_wgs84_shp.union(cur_shp)
                else:
                    country_wgs84_shp = cur_shp

    return country_wgs84_shp


def read_forest_alerts_confidence_raster(
    fname: str,
    conf_prefix: str,
) -> tuple[np.ndarray, rasterio.DatasetReader]:
    """Read the forest alerts confidence raster."""
    conf_path = UPath(conf_prefix) / fname
    buf = io.BytesIO()
    logger.debug(f"conf_path: {conf_path}")
    with conf_path.open("rb") as f:
        buf.write(f.read())
    buf.seek(0)
    conf_raster = rasterio.open(buf)
    conf_data = conf_raster.read(1)
    return conf_data, conf_raster


def read_forest_alerts_date_raster(
    fname: str,
    date_prefix: str,
) -> tuple[np.ndarray, rasterio.DatasetReader]:
    """Read the forest alerts date raster."""
    date_path = UPath(date_prefix) / fname
    logger.debug(f"date_path: {date_path}")
    buf = io.BytesIO()
    with date_path.open("rb") as f:
        buf.write(f.read())
    buf.seek(0)
    date_raster = rasterio.open(buf)
    date_data = date_raster.read(1)
    return date_data, date_raster


def process_shapes_into_events(
    shapes: list[shapely.Geometry],
    conf_raster: rasterio.DatasetReader,
    date_data: np.ndarray,
    country_wgs84_shp: shapely.Polygon,
    min_area: float,
) -> list[ForestLossEvent]:
    """Process the masked out shapes into a forest loss event."""
    events: list[ForestLossEvent] = []
    logger.info(f"processing {len(shapes)} shapes")
    background_skip_count = 0
    area_skip_count = 0
    country_skip_count = 0
    logger.info(f"min area: {min_area}")
    # Set pbar update interval to 5 to avoid token too long error on gcp buffer "bufio.Scanner token too long"
    pbar_update_interval = 5
    for shp, value in tqdm.tqdm(
        shapes, desc="process shapes", mininterval=pbar_update_interval
    ):
        # Skip shapes corresponding to the background.
        if value != 1:
            background_skip_count += 1
            continue

        shp = shapely.geometry.shape(shp)
        if shp.area < min_area:
            area_skip_count += 1
            continue
        # Get center point (clipped to shape) and note the corresponding date.
        center_shp, _ = shapely.ops.nearest_points(shp, shp.centroid)
        center_pixel = (int(center_shp.x), int(center_shp.y))
        cur_days = int(
            date_data[center_pixel[1], center_pixel[0]]
        )  # WE should document if date data is inverted coords
        cur_date = BASE_DATETIME + timedelta(days=cur_days)
        center_proj_coords = conf_raster.xy(center_pixel[1], center_pixel[0])
        center_proj_shp = shapely.Point(center_proj_coords[0], center_proj_coords[1])
        center_proj_geom = STGeometry(
            Projection(conf_raster.crs, 1, 1), center_proj_shp, None
        )

        # Check if it's in the Country Polygon.
        center_wgs84_shp = center_proj_geom.to_projection(WGS84_PROJECTION).shp
        if not country_wgs84_shp.contains(center_wgs84_shp):
            country_skip_count += 1
            # logger.debug("skipping shape not in Peru")
            continue

        def raster_pixel_to_proj(points: np.ndarray) -> np.ndarray:
            for i in range(points.shape[0]):
                points[i, 0:2] = conf_raster.xy(points[i, 1], points[i, 0])
            return points

        polygon_proj_shp = shapely.transform(shp, raster_pixel_to_proj)
        polygon_proj_geom = STGeometry(
            Projection(conf_raster.crs, 1, 1), polygon_proj_shp, None
        )
        forest_loss_event = ForestLossEvent(
            polygon_proj_geom, center_proj_geom, center_pixel, cur_date
        )
        events.append(forest_loss_event)
    logger.debug(f"Skipped {background_skip_count} shapes as background")
    logger.debug(f"Skipped {area_skip_count} shapes due to area")
    logger.debug(f"Skipped {country_skip_count} shapes not in country polygon")
    return events


def create_forest_loss_mask(
    conf_data: np.ndarray,
    date_data: np.ndarray,
    min_confidence: int,
    days: int,
    current_utc_time: datetime = datetime.now(timezone.utc),
) -> np.ndarray:
    """Create a mask based on the given time range and confidence threshold.

    Args:
        conf_data: the confidence data.
        date_data: the date data.
        min_confidence: the minimum confidence threshold.
        days: the number of days to look back.
        current_utc_time: the current time.

    Returns:
        a mask with the same shape as conf_data and date_data with 1s where the
        confidence and date conditions are met and 0s otherwise.
    """
    now_days = (current_utc_time - BASE_DATETIME).days
    min_days = now_days - days
    date_mask = date_data >= min_days
    conf_mask = conf_data >= min_confidence
    mask = date_mask & conf_mask
    return mask.astype(
        np.uint8
    )  # THis seems to be causing issues for writing unless we change it when we write it


def save_inference_dataset_config(
    ds_path: UPath, index_cache_dir: str, tile_store_dir: str
) -> None:
    """Save the inference dataset config.

    Args:
        ds_path: the path to store the rslearn dataset for these alerts.
        index_cache_dir: path to use for data source index.
        tile_store_dir: path to use for tile store.
    """
    with open(INFERENCE_DATASET_CONFIG) as config_file:
        # The dataset config has placeholders for INDEX_CACHE_DIR and TILE_STORE_DIR.
        config_str = config_file.read()
        config_str = config_str.replace("${INDEX_CACHE_DIR}", index_cache_dir)
        config_str = config_str.replace("${TILE_STORE_DIR}", tile_store_dir)
        config_json = json.loads(config_str)
    with (ds_path / "config.json").open("w") as f:
        json.dump(config_json, f)


def extract_alerts_pipeline(
    ds_root: UPath,
    extract_alerts_args: ExtractAlertsArgs,
) -> None:
    """Extract alerts from a single GeoTIFF file.

    Args:
        ds_root: the root path to the dataset.
        extract_alerts_args: the extract_alerts_args
    """
    # Skip extraction if it was marked completed.
    completed_fname = ds_root / COMPLETED_FNAME
    if completed_fname.exists():
        return

    # Create the dataset configuration file.
    save_inference_dataset_config(
        ds_root,
        index_cache_dir=extract_alerts_args.index_cache_dir,
        tile_store_dir=extract_alerts_args.tile_store_dir,
    )

    # Process the GLAD alert tiles one tile at a time.
    # Each tile has two files we need to read, the confidence raster (which we use to
    # threshold pixels by confidence threshold) and date raster (which we use to only
    # select pixels with recent forest loss based on specified number of days).
    total_events = 0
    logger.info(f"Extract_alerts for {str(extract_alerts_args)}")
    country_wgs84_shp = load_country_polygon(extract_alerts_args.country_data_path)
    for fname in extract_alerts_args.gcs_tiff_filenames:
        logger.info(f"Read confidences for {fname}")
        conf_data, conf_raster = read_forest_alerts_confidence_raster(
            fname, extract_alerts_args.conf_prefix
        )
        logger.info(f"Read dates for {fname}")
        date_data, date_raster = read_forest_alerts_date_raster(
            fname, extract_alerts_args.date_prefix
        )
        logger.info(f"Create mask for {fname}")
        forest_loss_mask = create_forest_loss_mask(
            conf_data,
            date_data,
            extract_alerts_args.min_confidence,
            extract_alerts_args.days,
            extract_alerts_args.prediction_utc_time,
        )
        # check if mask is all 0s
        if np.all(forest_loss_mask == 0):
            logger.warning(f"No forest loss events found for {fname}")
            continue
        logger.info(f"Create shapes from mask for {fname}")
        # Rasterio uses True for features and False for background
        ignore_mask = forest_loss_mask == 1
        shapes = list(rasterio.features.shapes(forest_loss_mask, mask=ignore_mask))
        logger.info(f"Process shapes into events for {fname}")
        events = process_shapes_into_events(
            shapes,
            conf_raster,
            date_data,
            country_wgs84_shp,
            extract_alerts_args.min_area,
        )
        # Close raster files
        conf_raster.close()
        date_raster.close()
        if extract_alerts_args.max_number_of_events is not None:
            logger.info(
                f"Limiting to {extract_alerts_args.max_number_of_events} \
                        events"
            )
            events = events[: extract_alerts_args.max_number_of_events]
        logger.info(f"Writing {len(events)} windows")
        total_events += len(events)
        jobs = [
            dict(
                event=event,
                fname=fname,
                ds_path=ds_root,
            )
            for event in events
        ]
        p = multiprocessing.Pool(extract_alerts_args.workers)
        outputs = star_imap_unordered(p, write_event, jobs)
        for _ in tqdm.tqdm(outputs, desc="Writing windows", total=len(jobs)):
            pass
        p.close()
    logger.info(f"Total events: {total_events}")
    if total_events == 0:
        # Raise an error since this likely means there is a misconfiguration.
        raise ValueError(
            "No forest loss events found in the given GeoTIFF files. \
            Please check the GeoTIFF files and the configuration."
        )

    # Mark it completed so we don't run this again in case user reruns the pipeline.
    completed_fname.touch()
