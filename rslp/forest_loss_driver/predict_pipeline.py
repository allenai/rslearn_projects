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

# Time corresponding to 0 in alertDate GeoTIFF files.
BASE_DATETIME = datetime(2019, 1, 1, tzinfo=timezone.utc)

# Where to obtain the forest loss alert data.
GCS_CONF_PREFIX = "gs://earthenginepartners-hansen/S2alert/alert/"
GCS_DATE_PREFIX = "gs://earthenginepartners-hansen/S2alert/alertDate/"
GCS_FILENAMES = [
    "070W_10S_060W_00N.tif",
    "070W_20S_060W_10S.tif",
    "080W_10S_070W_00N.tif",
    "080W_20S_070W_10S.tif",
]

# How big the rslearn windows should be.
WINDOW_SIZE = 128

# Create windows at WebMercator zoom 13 (512x512 tiles).
WEB_MERCATOR_CRS = CRS.from_epsg(3857)
WEB_MERCATOR_M = 2 * math.pi * 6378137
PIXEL_SIZE = WEB_MERCATOR_M / (2**13) / 512
WEB_MERCATOR_PROJECTION = Projection(WEB_MERCATOR_CRS, PIXEL_SIZE, -PIXEL_SIZE)


class PredictPipelineConfig:
    """Prediction pipeline config for forest loss driver classification."""

    def __init__(
        self,
        ds_root: str | None = None,
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
        rslp_bucket = os.environ["RSLP_BUCKET"]
        # TODO: SAve these paths in constants for readability and clarity
        if ds_root is None:
            ds_root = f"gcs://{rslp_bucket}/datasets/forest_loss_driver/prediction/dataset_20240828/"
        if country_data_path is None:
            country_data_path = f"gcs://{rslp_bucket}/artifacts/natural_earth_countries/20240830/ne_10m_admin_0_countries.shp"

        self.path = UPath(ds_root)
        self.workers = workers
        self.days = days
        self.min_confidence = min_confidence
        self.min_area = min_area
        self.country_data_path = UPath(country_data_path)


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


def write_event(event: ForestLossEvent, fname: str, ds_path: UPath) -> None:
    """Write a window for this forest loss event.

    Args:
        event: the event details.
        fname: the GeoTIFF filename that this alert came from.
        ds_path: the path of dataset to write to.
    """
    # Transform the center to Web-Mercator for populating the window.
    center_webm_shp = event.center_geom.to_projection(WEB_MERCATOR_PROJECTION).shp

    # WebMercator point to include in the name.
    # The annotation website will use this so we have adjusted this to have the same offset as multisat/other code.
    mercator_point = (
        int(center_webm_shp.x) + 512 * (2**12),
        int(center_webm_shp.y) + 512 * (2**12),
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

    # Output some metadata.
    polygon_wgs84_shp: shapely.Polygon = event.polygon_geom.to_projection(
        WGS84_PROJECTION
    ).shp
    with (window_path / "info.json").open("w") as f:
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

    # Get pixel coordinates of the mask.
    polygon_webm_shp = event.polygon_geom.to_projection(WEB_MERCATOR_PROJECTION).shp

    def to_out_pixel(points: np.ndarray) -> np.ndarray:
        points[:, 0] -= bounds[0]
        points[:, 1] -= bounds[1]
        return points

    polygon_out_shp = shapely.transform(polygon_webm_shp, to_out_pixel)
    mask_im = rasterio.features.rasterize(
        [(polygon_out_shp, 255)],
        out_shape=(WINDOW_SIZE, WINDOW_SIZE),
    )
    layer_dir = window_path / "layers" / "mask"
    SingleImageRasterFormat().encode_raster(
        layer_dir / "mask",
        window.projection,
        window.bounds,
        mask_im[None, :, :],
    )
    with (layer_dir / "completed").open("w"):
        pass


def extract_alerts(config: PredictPipelineConfig, fname: str) -> None:
    """Extract alerts from a single GeoTIFF file.

    Args:
        config: the pipeline config.
        fname: the filename
    """
    # Load Peru country polygon which will be used to sub-select the forest loss
    # events.
    prefix = ".".join(config.country_data_path.name.split(".")[:-1])
    aux_files: list[UPath] = []
    for ext in SHAPEFILE_AUX_EXTENSIONS:
        aux_files.append(config.country_data_path.parent / (prefix + ext))
    peru_wgs84_shp: shapely.Polygon | None = None
    with get_upath_local(
        config.country_data_path, extra_paths=aux_files
    ) as local_fname:
        with fiona.open(local_fname) as src:
            for feat in src:
                if feat["properties"]["ISO_A2"] != "PE":
                    continue
                cur_shp = shapely.geometry.shape(feat["geometry"])
                if peru_wgs84_shp:
                    peru_wgs84_shp = peru_wgs84_shp.union(cur_shp)
                else:
                    peru_wgs84_shp = cur_shp

    print(fname, "read confidences")
    conf_path = UPath(GCS_CONF_PREFIX) / fname
    buf = io.BytesIO()
    with conf_path.open("rb") as f:
        buf.write(f.read())
    buf.seek(0)
    conf_raster = rasterio.open(buf)
    conf_data = conf_raster.read(1)

    print(fname, "read dates")
    date_path = UPath(GCS_DATE_PREFIX) / fname
    buf = io.BytesIO()
    with date_path.open("rb") as f:
        buf.write(f.read())
    buf.seek(0)
    date_raster = rasterio.open(buf)
    date_data = date_raster.read(1)

    # Create mask based on the given time range and confidence threshold.
    now_days = (datetime.now(timezone.utc) - BASE_DATETIME).days
    min_days = now_days - config.days
    mask = (date_data >= min_days) & (conf_data >= config.min_confidence)
    mask = mask.astype(np.uint8)

    print(fname, "extract shapes")
    shapes = list(rasterio.features.shapes(mask))

    # Start by processing the shapes into ForestLossEvent.
    # Then prepare windows in a second phase.
    # We separate these steps because the first phase can't be parallelized easily
    # since it needs access to the date_data to lookup date of each polygon.
    events: list[ForestLossEvent] = []
    for shp, value in tqdm.tqdm(shapes, desc="process shapes"):
        # Skip shapes corresponding to the background.
        if value != 1:
            continue

        shp = shapely.geometry.shape(shp)
        if shp.area < config.min_area:
            continue

        # Get center point (clipped to shape) and note the corresponding date.
        center_shp, _ = shapely.ops.nearest_points(shp, shp.centroid)
        center_pixel = (int(center_shp.x), int(center_shp.y))
        cur_days = int(date_data[center_pixel[1], center_pixel[0]])
        cur_date = BASE_DATETIME + timedelta(days=cur_days)
        center_proj_coords = conf_raster.xy(center_pixel[1], center_pixel[0])
        center_proj_shp = shapely.Point(center_proj_coords[0], center_proj_coords[1])
        center_proj_geom = STGeometry(
            Projection(conf_raster.crs, 1, 1), center_proj_shp, None
        )

        # Check if it's in Peru.
        center_wgs84_shp = center_proj_geom.to_projection(WGS84_PROJECTION).shp
        if peru_wgs84_shp is None or not peru_wgs84_shp.contains(center_wgs84_shp):
            continue

        def raster_pixel_to_proj(points: np.ndarray) -> np.ndarray:
            for i in range(points.shape[0]):
                points[i, 0:2] = conf_raster.xy(points[i, 1], points[i, 0])
            return points

        polygon_proj_shp = shapely.transform(shp, raster_pixel_to_proj)
        polygon_proj_geom = STGeometry(
            Projection(conf_raster.crs, 1, 1), polygon_proj_shp, None
        )

        events.append(
            ForestLossEvent(polygon_proj_geom, center_proj_geom, center_pixel, cur_date)
        )

    conf_raster.close()
    date_raster.close()

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
