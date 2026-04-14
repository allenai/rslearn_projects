"""Utils functions for the repository."""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import Affine, from_origin

from data_preproc_script.constants import BANDS_10, BANDS_20, BANDS_60, EE_CRS
from data_preproc_script.preprocess.temporal_grid_agg import build_temporal_grid


def create_logger(name: str, log_file: str) -> logging.Logger:
    """Create a Logger.

    Args:
        name (str): Name of the logger.
        log_file (str): Name of the file for logging.

    Returns:
        logging.Logger: Logger object.
    """
    # Get the current time and format it as a string
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create a logger
    logger = logging.getLogger(name)

    # Set the level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Create a file handler that writes to a file named after the current time
    log_path = Path(f"{log_file}_{time_string}.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set the formatter for the handler
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


def center_crop(img: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """Return a centered crop with the requested output size."""
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = out_size
    crop_top = int((image_height - crop_height + 1) * 0.5)
    crop_left = int((image_width - crop_width + 1) * 0.5)
    return img[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]


def adjust_coords(
    coords: list[list[float]],
    old_size: tuple[int, int],
    new_size: tuple[int, int],
) -> list[list[float]]:
    """Adjust bounding coordinates after applying a centered crop."""
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [
            coords[0][0] + ((xoff + new_size[1]) * xres),
            coords[0][1] - ((yoff + new_size[0]) * yres),
        ],
    ]


def save_geotiff(
    img: np.ndarray, coords: list[list[float]], filename: str, compression: str = "LZW"
) -> None:
    """Function to write numpy arry as GeoTiff.

    Args:
        img (np.ndarray): Input image numpy array
        coords (List[List[float]]): Coordinates of the image
        filename (str): Filename for the geotiff file.
        compression (str, optional): Compression of the data. Defaults to "LZW".
    """
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(
        coords[0][0] - xres / 2, coords[0][1] + yres / 2
    ) * Affine.scale(xres, -yres)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": channels,
        "crs": "+proj=latlong",
        "transform": transform,
        "dtype": img.dtype,
        "compress": compression,
    }
    with rasterio.open(filename, "w", **profile) as f:
        f.write(img.transpose(2, 0, 1))


def S2_process(img: np.ndarray) -> np.ndarray:
    """Scale raw Sentinel-2 reflectance values to an 8-bit preview image."""
    from skimage.exposure import rescale_intensity

    img = img / 10000
    img = rescale_intensity(img, in_range=(0, 1), out_range=np.uint8)
    return img


def wind_comp(image: Any) -> Any:
    """Calculate the wind speed from the u and v components."""
    u = image.select("u_component_of_wind_10m")
    v = image.select("v_component_of_wind_10m")
    wind = u.pow(2).add(v.pow(2)).sqrt()
    wind = wind.rename("wind_speed_10m")
    return wind


def relative_humidity(image: Any) -> Any:
    """Calculate the relative humidity from the temperature and dewpoint."""
    import ee  # noqa: F811

    temp = image.select("temperature_2m").subtract(273.15)
    dew = image.select("dewpoint_temperature_2m").subtract(273.15)
    act_vp = ee.Image(6.11).multiply(
        ee.Image(10).pow(ee.Image(7.5).multiply(dew).divide(ee.Image(237.3).add(dew)))
    )
    sat_vp = ee.Image(6.11).multiply(
        ee.Image(10).pow(ee.Image(7.5).multiply(temp).divide(ee.Image(237.3).add(temp)))
    )
    rh = act_vp.divide(sat_vp).multiply(100)
    rh = rh.rename("relative_humidity_2m")
    return rh


def vapor_pressure(image: Any) -> Any:
    """Calculate the vapor pressure from the temperature and dewpoint."""
    import ee  # noqa: F811

    temp = image.select("temperature_2m").subtract(273.15)
    dew = image.select("dewpoint_temperature_2m").subtract(273.15)
    act_vp = ee.Image(6.11).multiply(
        ee.Image(10).pow(ee.Image(7.5).multiply(dew).divide(ee.Image(237.3).add(dew)))
    )
    sat_vp = ee.Image(6.11).multiply(
        ee.Image(10).pow(ee.Image(7.5).multiply(temp).divide(ee.Image(237.3).add(temp)))
    )
    vp = sat_vp.subtract(act_vp)
    vp = vp.rename("vapor_pressure_2m")
    return vp


def lst_qa(image: Any) -> Any:
    """Mask for LST MODIS data based QC_DAY band."""
    qc = image.select("QC_Day")

    mandatory_qa = qc.bitwiseAnd(3).lte(1)
    data_qa = qc.rightShift(2).bitwiseAnd(3).eq(0)

    # emis_qa = qc.rightShift(4).bitwiseAnd(3).lte(2)
    # lst_qa = qc.rightShift(6).bitwiseAnd(3).lte(2)

    mask = mandatory_qa.And(data_qa)
    return image.updateMask(mask)


def fwi_agg(
    file_path: os.PathLike, year: int, temp_offset: int, res: float, store: bool = False
) -> xr.DataArray | None:
    """Aggregate daily FWI data into temporal-grid windows and save as NetCDF.

    Optionally also exports per-date GeoTIFFs (requires ``rioxarray``).
    """
    temp_grid_dates = build_temporal_grid(year, year, temp_offset)
    tot_ds_dc_fwi = xr.open_dataset(file_path)
    tot_ds_dc_fwi["longitude"] = xr.where(
        tot_ds_dc_fwi["longitude"] > 180,
        tot_ds_dc_fwi["longitude"] - 360,
        tot_ds_dc_fwi["longitude"],
    )

    tot_list_ds = []

    for date in temp_grid_dates:
        end_date = date + timedelta(days=temp_offset - 1)
        ds_dc_fwi = tot_ds_dc_fwi.sel(valid_time=slice(date, end_date))
        ds_dc_fwi_mean = ds_dc_fwi.mean(dim="valid_time")
        ds_dc_fwi_min = ds_dc_fwi.min(dim="valid_time")
        ds_dc_fwi_max = ds_dc_fwi.max(dim="valid_time")
        ds_dc_fwi = xr.merge(
            [
                ds_dc_fwi_mean.rename(
                    {var: f"{var}_mean" for var in ds_dc_fwi_mean.data_vars}
                ),
                ds_dc_fwi_min.rename(
                    {var: f"{var}_min" for var in ds_dc_fwi_min.data_vars}
                ),
                ds_dc_fwi_max.rename(
                    {var: f"{var}_max" for var in ds_dc_fwi_max.data_vars}
                ),
            ],
            compat="override",
        )
        ds_dc_fwi = ds_dc_fwi.assign_coords(valid_time=date)
        tot_list_ds.append(ds_dc_fwi)

    print("Concatenating aggregated values...")
    tot_ds_dc_fwi = xr.concat(tot_list_ds, dim="valid_time")
    tot_ds_dc_fwi = tot_ds_dc_fwi.sortby("latitude", ascending=False)
    tot_ds_dc_fwi = tot_ds_dc_fwi.sortby("longitude", ascending=True)

    if store:
        file_path = Path(file_path)

        print("Saving aggregated NetCDF...")
        tot_ds_dc_fwi.to_netcdf(file_path.parent / f"fwi_dc_agg_{year}.nc")

        # GeoTIFF export is optional — only attempted if rioxarray is available.
        try:
            import rioxarray  # noqa: F401

            tot_ds_dc_fwi.rio.write_crs(EE_CRS, inplace=True)
            west = tot_ds_dc_fwi.longitude.min().values
            north = tot_ds_dc_fwi.latitude.max().values
            transform = from_origin(west - (res / 2), north + (res / 2), res, res)
            tot_ds_dc_fwi.rio.write_transform(transform, inplace=True)

            print("Saving per-date GeoTIFFs...")
            for date in temp_grid_dates:
                date_str = date.strftime("%Y%m%d")
                ds_dc_fwi = tot_ds_dc_fwi.sel(valid_time=date)
                ds_path = file_path.parent / f"fwi_dc_agg_{date_str}.tif"
                ds_dc_fwi.rio.to_raster(ds_path)
        except ImportError:
            print(
                "rioxarray not installed — skipping GeoTIFF export. "
                "Install with: pip install rioxarray"
            )

        return None

    else:
        return tot_ds_dc_fwi


def assign_val(
    geo_ref: gpd.GeoDataFrame,
    geo_val: gpd.GeoDataFrame,
    key_cols: list[str],
    val_col: str,
    temp_cols: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Assign the value of the geo_val to the geo_ref based on the spatial join."""
    geo_val = geo_val.copy()
    geo_ref = geo_ref.copy()

    # Keep right geometry
    geo_val["geometry_right"] = geo_val["geometry"]
    geo_ref["geometry_left"] = geo_ref["geometry"]

    # Spatial Join and Geometry Intersection | Temporal Join
    geo_ref = gpd.sjoin(geo_ref, geo_val, how="inner", predicate="intersects")
    if temp_cols is not None:
        geo_ref = geo_ref[geo_ref[temp_cols[0]] == geo_ref[temp_cols[1]]]
    geo_ref["geometry"] = geo_ref.apply(
        lambda x: x["geometry"].intersection(x["geometry_right"]), axis=1
    )

    # Calculate the area of the intersection and the ratio of the area
    geo_ref["area"] = geo_ref["geometry"].area
    sum_geo_ref = geo_ref.groupby(key_cols).agg({"area": "sum"}).reset_index()
    sum_geo_ref = sum_geo_ref.rename(columns={"area": "sum_area"})
    geo_ref = geo_ref.merge(sum_geo_ref, on=key_cols, how="inner")
    geo_ref["ratio_area"] = geo_ref["area"] / geo_ref["sum_area"]

    # Compute weighted mean
    geo_ref[val_col] = geo_ref[val_col] * geo_ref["ratio_area"]
    geo_ref = (
        geo_ref.groupby(key_cols + ["geometry_left"])
        .agg({val_col: "sum"})
        .reset_index()
    )
    geo_ref.rename(columns={"geometry_left": "geometry"}, inplace=True)

    return geo_ref


def load_and_concatenate_tif(
    folder_path: str, file_names: list[str], to_tensor: bool = False
) -> tuple:
    """Extract and concatenate multiple S2 bands from GeoTIFFs.

    Returns (images_10m, images_20m, images_60m) as numpy arrays or tensors.
    """
    import torch
    import torchvision.transforms as transforms
    from PIL import Image

    images_10 = []
    images_20 = []
    images_60 = []

    for name in file_names:
        file_path = os.path.join(folder_path, name + ".tif")
        image = Image.open(file_path)
        if name in BANDS_10:
            images_10.append(np.asarray(image))
        elif name in BANDS_20:
            images_20.append(np.asarray(image))
        elif name in BANDS_60:
            images_60.append(np.asarray(image))

    # Apply any necessary transformations to the images

    if to_tensor:
        transform = transforms.ToTensor()
        images_10 = [transform(image) for image in images_10]
        images_20 = [transform(image) for image in images_20]
        images_60 = [transform(image) for image in images_60]

        concatenated_images_10 = (
            torch.cat(images_10, dim=0) if len(images_10) > 0 else None
        )
        concatenated_images_20 = (
            torch.cat(images_20, dim=0) if len(images_20) > 0 else None
        )
        concatenated_images_60 = (
            torch.cat(images_60, dim=0) if len(images_60) > 0 else None
        )

    else:
        concatenated_images_10 = (
            np.stack(images_10, axis=0) if len(images_10) > 0 else None
        )
        concatenated_images_20 = (
            np.stack(images_20, axis=0) if len(images_20) > 0 else None
        )
        concatenated_images_60 = (
            np.stack(images_60, axis=0) if len(images_60) > 0 else None
        )

    return concatenated_images_10, concatenated_images_20, concatenated_images_60


def extract_location(file_path: str | os.PathLike[str], target_band: str) -> np.ndarray:
    """Return per-pixel x/y coordinates for the requested raster band."""
    with rasterio.open(Path(file_path) / f"{target_band}.tif") as src:
        height = src.height
        width = src.width
        rows, cols = np.indices((height, width))
        x, y = src.xy(rows, cols)
        pixels_loc = np.stack([x, y], axis=0)

    return pixels_loc
