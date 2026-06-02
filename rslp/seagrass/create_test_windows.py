"""Create rslearn test windows for the Jamaica seagrass reference polygons.

The test set is a GeoJSON of labeled seagrass polygons in WGS84/CRS84 with two
classes ("Seagrass Sparse" -> 1, "Seagrass Dense" -> 2). Each feature becomes
one rslearn window in its centroid's UTM zone. The window bounds are the
polygon's UTM bounding box snapped to the 10 m label grid (with optional
padding). The ``label_raster`` is the polygon rasterized at 10 m: pixels
inside the polygon get the seagrass class id and pixels outside get
``--negative_class`` (default 255 = ignore), so by default only annotated
pixels contribute to evaluation. Pass ``--negative_class 0`` to instead treat
outside pixels as background.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_GEOJSON_PATH = Path(
    "/weka/dfive-default/piperw/jamaica_seagrass_reference_polygons.geojson"
)
DEFAULT_DATASET_PATH = Path("/weka/dfive-default/piperw/rslearn_projects/data/seagrass")

WINDOW_RESOLUTION = 10
LABEL_LAYER = "label_raster"
LABEL_BAND = "label"
LABEL_NAMES = {
    0: "background",
    1: "sparse_seagrass",
    2: "dense_seagrass",
}
CLASS_TO_LABEL = {
    "Seagrass Sparse": 1,
    "Seagrass Dense": 2,
}
IGNORE_LABEL = 255


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """UTM EPSG code (north or south) for a lon/lat point."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if lat >= 0 else 32700) + zone


def reproject_to_utm(geometry: dict[str, Any]) -> tuple[Any, int]:
    """Reproject a GeoJSON geometry from WGS84 to its centroid UTM zone."""
    from pyproj import Transformer
    from shapely.geometry import shape
    from shapely.ops import transform

    geom = shape(geometry)
    centroid = geom.centroid
    epsg = utm_epsg_for_lonlat(centroid.x, centroid.y)
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    utm_geom = transform(transformer.transform, geom)
    return utm_geom, epsg


def compute_window_bounds(
    utm_bbox: tuple[float, float, float, float],
    padding_pixels: int,
) -> tuple[int, int, int, int]:
    """Snap a UTM bbox to the 10 m pixel grid and apply padding.

    rslearn convention: ``pixel_x = utm_x / x_res`` and
    ``pixel_y = utm_y / y_res`` where ``y_res = -10`` (north-up). Returns
    ``(pixel_x_min, pixel_y_min, pixel_x_max, pixel_y_max)`` with
    ``pixel_y_min`` corresponding to the top of the window (max utm_y).
    """
    min_x, min_y, max_x, max_y = utm_bbox
    px_x_min = math.floor(min_x / WINDOW_RESOLUTION) - padding_pixels
    px_x_max = math.ceil(max_x / WINDOW_RESOLUTION) + padding_pixels
    # y_resolution is negative: larger utm_y -> smaller (more negative) pixel_y.
    px_y_min = math.floor(max_y / -WINDOW_RESOLUTION) - padding_pixels
    px_y_max = math.ceil(min_y / -WINDOW_RESOLUTION) + padding_pixels
    if px_x_max <= px_x_min:
        px_x_max = px_x_min + 1
    if px_y_max <= px_y_min:
        px_y_max = px_y_min + 1
    return px_x_min, px_y_min, px_x_max, px_y_max


def rasterize_polygon(
    utm_geom: Any,
    bounds: tuple[int, int, int, int],
    label_value: int,
    fill_value: int,
    all_touched: bool,
) -> Any:
    """Rasterize a polygon into a (1, H, W) uint8 array matching window bounds."""
    import numpy as np
    from affine import Affine
    from rasterio.features import rasterize

    px_x_min, px_y_min, px_x_max, px_y_max = bounds
    width = px_x_max - px_x_min
    height = px_y_max - px_y_min
    utm_x_origin = px_x_min * WINDOW_RESOLUTION
    utm_y_origin = px_y_min * -WINDOW_RESOLUTION  # top: largest utm_y
    transform = Affine(
        WINDOW_RESOLUTION,
        0,
        utm_x_origin,
        0,
        -WINDOW_RESOLUTION,
        utm_y_origin,
    )
    raster = rasterize(
        [(utm_geom, label_value)],
        out_shape=(height, width),
        transform=transform,
        fill=fill_value,
        dtype="uint8",
        all_touched=all_touched,
    )
    return raster[np.newaxis, :, :]


def write_info_json(
    ds_path: Any, group: str, name: str, payload: dict[str, Any]
) -> None:
    """Write per-window metadata alongside rslearn metadata.json."""
    from rslearn.dataset import Window

    window_root = Window.get_window_root(ds_path, group, name)
    with (window_root / "info.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def create_polygon_window(
    feature: dict[str, Any],
    feature_index: int,
    ds_path: Any,
    group: str,
    year: int,
    padding_pixels: int,
    negative_class: int,
    all_touched: bool,
) -> str:
    """Create one polygon-bounded test window with a rasterized label mask."""
    from rasterio.crs import CRS
    from rslearn.config.dataset import StorageConfig
    from rslearn.dataset import Window
    from rslearn.utils import Projection
    from rslearn.utils.raster_array import RasterArray
    from rslearn.utils.raster_format import GeotiffRasterFormat

    properties = feature.get("properties") or {}
    class_name = properties.get("Class_name")
    if class_name not in CLASS_TO_LABEL:
        raise ValueError(
            f"Unexpected Class_name {class_name!r} on feature {feature_index}"
        )
    label_value = CLASS_TO_LABEL[class_name]

    utm_geom, epsg = reproject_to_utm(feature["geometry"])
    utm_bbox = utm_geom.bounds
    bounds = compute_window_bounds(utm_bbox, padding_pixels)

    projection = Projection(CRS.from_epsg(epsg), WINDOW_RESOLUTION, -WINDOW_RESOLUTION)
    window_name = f"polygon_{feature_index:06d}"
    time_range = (
        datetime(year, 1, 1, tzinfo=timezone.utc),
        datetime(year, 12, 31, tzinfo=timezone.utc),
    )
    centroid = utm_geom.centroid

    window = Window(
        storage=StorageConfig()
        .instantiate_window_storage_factory()
        .get_storage(ds_path),
        group=group,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
        options={
            "split": "test",
            "feature_index": feature_index,
            "label": label_value,
            "label_name": LABEL_NAMES[label_value],
            "class_name": class_name,
            "utm_epsg": epsg,
        },
    )
    window.save()

    raster = rasterize_polygon(
        utm_geom, bounds, label_value, negative_class, all_touched
    )
    raster_dir = window.get_raster_dir(LABEL_LAYER, [LABEL_BAND])
    GeotiffRasterFormat().encode_raster(
        raster_dir,
        window.projection,
        window.bounds,
        RasterArray(chw_array=raster),
    )
    window.mark_layer_completed(LABEL_LAYER)

    info = {
        "feature_index": feature_index,
        "class_name": class_name,
        "label": label_value,
        "label_name": LABEL_NAMES[label_value],
        "utm_epsg": epsg,
        "utm_bbox": list(utm_bbox),
        "pixel_bounds": list(bounds),
        "padding_pixels": padding_pixels,
        "negative_class": negative_class,
        "all_touched": all_touched,
        "centroid_utm": [centroid.x, centroid.y],
        "properties": properties,
    }
    write_info_json(ds_path, group, window_name, info)
    return "test"


def iter_features(geojson_path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield (index, feature) pairs from the GeoJSON FeatureCollection."""
    with geojson_path.open() as f:
        gj = json.load(f)
    if gj.get("type") != "FeatureCollection":
        raise ValueError(f"Expected FeatureCollection, got {gj.get('type')!r}")
    yield from enumerate(gj["features"])


def run_window_jobs(
    jobs: list[dict[str, Any]],
    workers: int,
) -> list[str]:
    """Run polygon window jobs either serially or via rslearn multiprocessing."""
    import tqdm

    if workers == 1:
        return [create_polygon_window(**job) for job in tqdm.tqdm(jobs)]

    from rslearn.utils.mp import star_imap_unordered

    pool = multiprocessing.Pool(workers)
    outputs = star_imap_unordered(pool, create_polygon_window, jobs)
    splits = [split for split in tqdm.tqdm(outputs, total=len(jobs))]
    pool.close()
    return splits


def create_windows(args: argparse.Namespace) -> dict[str, int]:
    """Create polygon test windows according to CLI args."""
    from upath import UPath

    ds_path = UPath(args.ds_path)
    ds_path.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = [
        dict(
            feature=feature,
            feature_index=idx,
            ds_path=ds_path,
            group=args.group,
            year=args.year,
            padding_pixels=args.padding_pixels,
            negative_class=args.negative_class,
            all_touched=args.all_touched,
        )
        for idx, feature in iter_features(Path(args.geojson_path))
    ]
    splits = run_window_jobs(jobs, args.workers)
    summary = {"test": sum(1 for s in splits if s == "test")}
    with (ds_path / "test_polygon_split_summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geojson_path",
        default=str(DEFAULT_GEOJSON_PATH),
        help="Path to the reference polygons GeoJSON.",
    )
    parser.add_argument(
        "--ds_path",
        default=str(DEFAULT_DATASET_PATH),
        help="Output rslearn dataset path.",
    )
    parser.add_argument(
        "--group",
        default="jamaica_2025_test_polygons",
        help="rslearn window group name.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year used for window time ranges.",
    )
    parser.add_argument(
        "--padding_pixels",
        type=int,
        default=4,
        help="Extra pixels of padding around each polygon's UTM bbox.",
    )
    parser.add_argument(
        "--negative_class",
        type=int,
        default=IGNORE_LABEL,
        help=(
            "Label value for pixels outside the polygon. "
            "Default 255 = ignore (only annotated pixels contribute to metrics). "
            "Pass 0 to treat outside pixels as background."
        ),
    )
    parser.add_argument(
        "--all_touched",
        action="store_true",
        help=(
            "If set, label every pixel touched by the polygon edge. Default is "
            "False (only pixels whose center is inside the polygon get the label)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of worker processes.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver")
    summary = create_windows(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
