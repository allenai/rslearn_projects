"""Shared helpers for vessel prediction pipeline integration tests."""

import math

import numpy as np
import yaml
from rslearn.const import WGS84_PROJECTION
from rslearn.utils.geometry import PixelBounds, Projection, STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from shapely.geometry import box
from upath import UPath

# ~1.3 km box near the UK coast so the window stays within a single 512x512 patch at 10m.
WGS84_ITEM_BOUNDS: tuple[float, float, float, float] = (-0.01, 50.5, 0.01, 50.512)


def get_projection_and_bounds(
    x_resolution: float, y_resolution: float
) -> tuple[Projection, PixelBounds]:
    """Derive a UTM projection and pixel bounds from WGS84_ITEM_BOUNDS."""
    wgs84_geom = STGeometry(WGS84_PROJECTION, box(*WGS84_ITEM_BOUNDS), None)
    projection = get_utm_ups_projection(
        wgs84_geom.shp.centroid.x, wgs84_geom.shp.centroid.y, x_resolution, y_resolution
    )
    utm_geom = wgs84_geom.to_projection(projection)
    minx, miny, maxx, maxy = utm_geom.shp.bounds
    bounds: PixelBounds = (
        math.floor(minx),
        math.floor(miny),
        math.ceil(maxx),
        math.ceil(maxy),
    )
    return projection, bounds


def write_synthetic_geotiff(
    directory: UPath,
    projection: Projection,
    bounds: PixelBounds,
    nbands: int = 1,
    dtype: str = "uint16",
    fname: str = "geotiff.tif",
) -> None:
    """Write a small synthetic GeoTIFF with random data via GeotiffRasterFormat."""
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    data = np.random.randint(100, 5000, size=(nbands, height, width), dtype=np.uint16)
    if dtype == "uint8":
        data = (data % 256).astype(np.uint8)
    GeotiffRasterFormat().encode_raster(
        directory,
        projection,
        bounds,
        RasterArray(chw_array=data),
        fname=fname,
    )


# --- Model config patching ---


def apply_common_config_patches(cfg: dict) -> None:
    """Apply patches common to all test model configs."""
    cfg["data"]["init_args"]["num_workers"] = 0
    cfg["data"]["init_args"]["batch_size"] = 1
    cfg["load_checkpoint_required"] = "no"
    cfg["model"]["init_args"].pop("restore_config", None)


def create_tiny_detect_config(original_path: str, output_path: str) -> None:
    """Patch a detection model config (S2 or Landsat) to use swin_t."""
    with open(original_path) as f:
        cfg = yaml.safe_load(f)

    model_args = cfg["model"]["init_args"]
    encoder_list = model_args["model"]["init_args"]["encoder"]
    encoder_list[0]["init_args"]["arch"] = "swin_t"
    encoder_list[0]["init_args"]["pretrained"] = False
    encoder_list[1]["init_args"]["in_channels"] = [96, 192, 384, 768]

    apply_common_config_patches(cfg)

    with open(output_path, "w") as f:
        yaml.dump(cfg, f)
