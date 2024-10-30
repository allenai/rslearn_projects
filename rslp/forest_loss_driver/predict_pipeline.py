"""Forest loss driver prediction pipeline."""

import io
import json
import math
import multiprocessing
import os
from datetime import datetime, timedelta, timezone

import fiona
import numpy as np
import rasterio
import rasterio.features
import shapely
import shapely.geometry
import shapely.ops
import tqdm
from PIL import Image
from rasterio.crs import CRS
from rslearn.const import SHAPEFILE_AUX_EXTENSIONS, WGS84_PROJECTION
from rslearn.dataset import Window
from rslearn.utils import Projection, STGeometry
from rslearn.utils.fsspec import get_upath_local
from rslearn.utils.mp import star_imap_unordered
from rslearn.utils.raster_format import SingleImageRasterFormat
from upath import UPath

from rslp.log_utils import get_logger

logger = get_logger(__name__)

# Time corresponding to 0 in alertDate GeoTIFF files.
BASE_DATETIME = datetime(2019, 1, 1, tzinfo=timezone.utc)


## Constants.py
# Where to obtain the forest loss alert data.
GCS_CONF_PREFIX = "gs://earthenginepartners-hansen/S2alert/alert/"
GCS_DATE_PREFIX = "gs://earthenginepartners-hansen/S2alert/alertDate/"
GCS_FILENAMES = [
    "070W_10S_060W_00N.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "070W_20S_060W_10S.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "080W_10S_070W_00N.tif",  # What are these files I presume the s2 tiffs of the quarter?
    "080W_20S_070W_10S.tif",  # What are these files I presume the s2 tiffs of the quarter?
]

# How big the rslearn windows should be.
WINDOW_SIZE = 128

# Create windows at WebMercator zoom 13 (512x512 tiles).
WEB_MERCATOR_CRS = CRS.from_epsg(3857)
WEB_MERCATOR_M = 2 * math.pi * 6378137
PIXEL_SIZE = WEB_MERCATOR_M / (2**13) / 512
WEB_MERCATOR_PROJECTION = Projection(WEB_MERCATOR_CRS, PIXEL_SIZE, -PIXEL_SIZE)

ANNOTATION_WEBSITE_MERCATOR_OFFSET = 512 * (2**12)
# INFERENCE_DATASET_CONFIG = os.environ.get(
#     "INFERENCE_DATASET_CONFIG",
#     # str(
#     #     Path(__file__).resolve().parents[3] / "data" / "forest_loss_driver" / "config.json"
#     # ),
# )


class PredictPipelineConfig:
    """Prediction pipeline config for forest loss driver classification."""

    # Maybe put this in init or must be in a constants file?
    rslp_bucket = os.environ.get("RSLP_BUCKET", "rslearn-eai")

    peru_shape_data_path = f"gcs://{rslp_bucket}/artifacts/natural_earth_countries/20240830/ne_10m_admin_0_countries.shp"

    def __init__(
        self,
        ds_root: str,
        workers: int = 1,
        days: int = 365,
        min_confidence: int = 2,
        min_area: float = 16,
        country_data_path: UPath | None = None,
    ) -> None:
        """Create a new PredictPipelineConfig.

        Args:
            ds_root: optional dataset root to write the dataset. This defaults to GCS.
            workers: number of workers.
            days: only consider forest loss events in this many past days.
            min_confidence: threshold on the GLAD alert confidence.
            min_area: minimum area in pixels of forest loss polygons. Pixels are
                roughly 10x10 m.
            country_data_path: the path to access country boundary data, so we can
                select the subset of forest loss events that are within Peru.
        """
        if country_data_path is None:
            logger.info(
                f"using default peru shape data path: {self.peru_shape_data_path}"
            )
            country_data_path = self.peru_shape_data_path
        self.path = UPath(ds_root)
        self.workers = workers
        self.days = days
        self.min_confidence = min_confidence
        self.min_area = min_area
        self.country_data_path = UPath(country_data_path)

    def __str__(self) -> str:
        """Return a string representation of the config."""
        return (
            f"PredictPipelineConfig(path={self.path}, workers={self.workers}, "
            f"days={self.days}, min_confidence={self.min_confidence}, "
            f"min_area={self.min_area}, country_data_path={self.country_data_path})"
        )


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


## Event Writer
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
    window_path = ds_path / "windows" / "default" / window_name
    window = Window(
        path=window_path,
        group="default",
        name=window_name,
        projection=WEB_MERCATOR_PROJECTION,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()
    return window, window_path, mercator_point, bounds


#  TODO:This should really be a class that handles this
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


## Alert Extractor
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
    logger.info(f"min area: {min_area}")
    background_skip_count = 0
    area_skip_count = 0
    country_skip_count = 0

    for shp, value in tqdm.tqdm(shapes, desc="process shapes"):
        # Skip shapes corresponding to the background.
        if value != 1:
            background_skip_count += 1
            # logger.debug(f"skipping shape with value {value} as it is in background")
            continue

        shp = shapely.geometry.shape(shp)
        if shp.area < min_area:
            area_skip_count += 1
            # logger.debug(f"skipping shape with area {shp.area}")
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
    mask = (date_data >= min_days) & (conf_data >= min_confidence)
    return mask.astype(
        np.uint8
    )  # THis seems to be causing issues for writing unless we change it when we write it


def save_inference_dataset_config(ds_path: UPath) -> None:
    """Save the inference dataset config."""
    # TODO SLoppy needs a better way to handle this
    INFERENCE_DATASET_CONFIG = os.environ["INFERENCE_DATASET_CONFIG"]
    with open(INFERENCE_DATASET_CONFIG) as config_file:
        config_json = json.load(config_file)
    with (ds_path / "config.json").open("w") as f:
        json.dump(config_json, f)


def extract_alerts(
    config: PredictPipelineConfig,
    fname: str,
    date_prefix: str = GCS_DATE_PREFIX,
    conf_prefix: str = GCS_CONF_PREFIX,
    current_utc_time: datetime = datetime.now(timezone.utc),
) -> None:
    """Extract alerts from a single GeoTIFF file.

    Args:
        config: the pipeline config.
        fname: the filename
        current_utc_time: the time to look back from for forest loss events.
        date_prefix: the prefix for the date raster.
        conf_prefix: the prefix for the confidence raster.
    """
    logger.info(f"extract_alerts for {str(config)}")
    # Load Peru country polygon which will be used to sub-select the forest loss
    # events.
    country_wgs84_shp = load_country_polygon(config.country_data_path)

    logger.info(f"read confidences for {fname}")
    conf_data, conf_raster = read_forest_alerts_confidence_raster(fname, conf_prefix)

    logger.info(f"read dates for {fname}")
    date_data, date_raster = read_forest_alerts_date_raster(fname, date_prefix)
    logger.info(f"create mask for {fname}")
    mask = create_forest_loss_mask(
        conf_data,
        date_data,
        config.min_confidence,
        config.days,
        current_utc_time,
    )
    shapes = list(rasterio.features.shapes(mask))
    events = process_shapes_into_events(
        shapes, conf_raster, date_data, country_wgs84_shp, config.min_area
    )
    logger.info(f"writing {len(events)} windows")
    jobs = [
        dict(
            event=event,
            fname=fname,
            ds_path=config.path,
        )
        for event in events
    ]
    p = multiprocessing.Pool(config.workers)
    outputs = star_imap_unordered(p, write_event, jobs)
    for _ in tqdm.tqdm(outputs, desc="Writing windows", total=len(jobs)):
        pass
    p.close()
    save_inference_dataset_config(config.path)


## Image Selector
def select_best_images(window_path: UPath) -> None:
    """Select the best images for the specified window.

    Best just means least cloudy pixels based on a brightness threshold.

    It writes best pre and post images to the best_pre_X/best_post_X layers and also
    produces a best_times.json indicating the timestamp of the images selected for
    those layers.

    Args:
        window_path: the window root.
    """
    num_outs = 3
    min_choices = 5

    items_fname = window_path / "items.json"
    if not items_fname.exists():
        return

    # Get the timestamp of each expected layer.
    layer_times = {}
    with items_fname.open() as f:
        item_data = json.load(f)
        for layer_data in item_data:
            layer_name = layer_data["layer_name"]
            if "planet" in layer_name:
                continue
            for group_idx, group in enumerate(layer_data["serialized_item_groups"]):
                if group_idx == 0:
                    cur_layer_name = layer_name
                else:
                    cur_layer_name = f"{layer_name}.{group_idx}"
                layer_times[cur_layer_name] = group[0]["geometry"]["time_range"][0]

    # Find best pre and post images.
    image_lists: dict = {"pre": [], "post": []}
    options = window_path.glob("layers/*/R_G_B/image.png")
    for fname in options:
        # "pre" or "post"
        layer_name = fname.parent.parent.name
        k = layer_name.split(".")[0].split("_")[0]
        if "planet" in k or "best" in k:
            continue
        with fname.open("rb") as f:
            im = np.array(Image.open(f))[32:96, 32:96, :]
        image_lists[k].append((im, fname))

    # Copy the images to new "best" layer.
    # Keep track of the timestamps and write them to a separate file.
    best_times = {}
    for k, image_list in image_lists.items():
        if len(image_list) < min_choices:
            return
        image_list.sort(
            key=lambda t: np.count_nonzero(
                (t[0].max(axis=2) == 0) | (t[0].min(axis=2) > 140)
            )
        )
        for idx, (im, fname) in enumerate(image_list[0:num_outs]):
            dst_layer = f"best_{k}_{idx}"
            layer_dir = window_path / "layers" / dst_layer
            (layer_dir / "R_G_B").mkdir(parents=True, exist_ok=True)
            fname.fs.cp(fname.path, (layer_dir / "R_G_B" / "image.png").path)
            (layer_dir / "completed").touch()

            src_layer = fname.parent.parent.name
            layer_time = layer_times[src_layer]
            best_times[dst_layer] = layer_time

    with (window_path / "best_times.json").open("w") as f:
        json.dump(best_times, f)


def select_best_images_pipeline(ds_path: str | UPath, workers: int = 64) -> None:
    """Run the best image pipeline.

    This picks the best three pre/post images and puts them in the corresponding layers
    so the model can read them.

    It is based on amazon_conservation/make_dataset/select_images.py.

    Args:
        ds_path: the dataset root path
        workers: number of workers to use.
    """
    ds_path = UPath(ds_path) if not isinstance(ds_path, UPath) else ds_path
    window_paths = list(ds_path.glob("windows/*/*"))
    p = multiprocessing.Pool(workers)
    outputs = p.imap_unordered(select_best_images, window_paths)
    for _ in tqdm.tqdm(outputs, total=len(window_paths)):
        pass
    p.close()


# This is the main function that should be called to run the prediction pipeline. the alerts stuff likely should be in a different module
def predict_pipeline(pred_config: PredictPipelineConfig) -> None:
    """Run the prediction pipeline.

    Currently this is just for populating the initial rslearn dataset based on GLAD
    forest loss events in Peru over the last year.

    So need to prepare/ingest/materialize the dataset afterward, and run the
    select_best_images_pipeline. Then apply the model.

    Args:
        pred_config: the pipeline configuration
    """
    for fname in GCS_FILENAMES:
        extract_alerts(pred_config, fname)
