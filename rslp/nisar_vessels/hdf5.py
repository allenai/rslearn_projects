"""Reading raster bands out of local NISAR HDF5 granules.

GDAL's HDF5 driver lists a granule's band datasets but reports no georeferencing for
them, so the grid is read here with h5py and written back out as a GeoTIFF.
"""

import dataclasses

import affine
import h5py
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.crs import CRS

from rslp.log_utils import get_logger

logger = get_logger(__name__)

# Datasets describing the grid, siblings of the band datasets in each group.
X_COORDINATES_DATASET = "xCoordinates"
Y_COORDINATES_DATASET = "yCoordinates"
PROJECTION_DATASET = "projection"
EPSG_CODE_ATTRIBUTE = "epsg_code"
FILL_VALUE_ATTRIBUTE = "_FillValue"

# ASF only distributes NISAR's L-band products, so LSAR is the only group we read.
INSTRUMENT_GROUP = "LSAR"

# Default L2 product name as it appears in the group path. Every other L2 product lays
# its grids out the same way, so this is a parameter rather than a constant in the paths.
DEFAULT_PRODUCT = "GCOV"

# A granule does not record which frequency group a band landed in, so both are probed.
# These are cheap metadata lookups.
FREQUENCY_GROUPS = ("frequencyA", "frequencyB")

# How many granule rows to convert at a time. A GCOV band runs to a few hundred MB, and
# the sidecar has no reason to hold a whole one in memory just to copy it out.
BLOCK_ROWS = 1024

# GeoTIFF tile size. Tiling matters because rslearn materializes the scene by reading
# windows out of this file rather than the whole thing at once.
GEOTIFF_BLOCK_SIZE = 512


@dataclasses.dataclass(frozen=True)
class GranuleGrid:
    """The geocoded grid that a NISAR band is sampled on.

    ``x_origin`` and ``y_origin`` are the outer corner of the first pixel, and
    ``y_resolution`` is signed, so ``transform`` is valid whichever direction the
    granule's y coordinates run.
    """

    crs: CRS
    x_resolution: float
    y_resolution: float
    x_origin: float
    y_origin: float
    width: int
    height: int

    @property
    def transform(self) -> affine.Affine:
        """The affine transform mapping pixel coordinates to CRS coordinates."""
        return affine.Affine(
            self.x_resolution, 0, self.x_origin, 0, self.y_resolution, self.y_origin
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """The grid's extent in CRS coordinates, as (minx, miny, maxx, maxy)."""
        far_x = self.x_origin + self.width * self.x_resolution
        far_y = self.y_origin + self.height * self.y_resolution
        return (
            min(self.x_origin, far_x),
            min(self.y_origin, far_y),
            max(self.x_origin, far_x),
            max(self.y_origin, far_y),
        )


def find_band_group(
    granule: h5py.File, band: str, product: str = DEFAULT_PRODUCT
) -> h5py.Group:
    """Find the HDF5 group holding a band, probing each frequency group.

    Args:
        granule: the opened granule.
        band: the band (HDF5 dataset) name to look for.
        product: the L2 product name as it appears in the group path.

    Returns:
        the group containing the band dataset.

    Raises:
        ValueError: if no candidate group contains the band.
    """
    probed: list[str] = []
    for frequency in FREQUENCY_GROUPS:
        group_path = f"science/{INSTRUMENT_GROUP}/{product}/grids/{frequency}"
        probed.append(group_path)
        group = granule.get(group_path)
        if isinstance(group, h5py.Group) and band in group:
            logger.debug(f"Found NISAR band {band} in group {group_path}")
            return group
    raise ValueError(f"No group in the granule has band {band}, probed: {probed}")


def read_grid(group: h5py.Group, band: str) -> GranuleGrid:
    """Read the grid a band is sampled on from its group's coordinate datasets.

    Args:
        group: the group holding the band, from :func:`find_band_group`.
        band: the band (HDF5 dataset) name.

    Returns:
        the band's GranuleGrid.

    Raises:
        ValueError: if the grid is too small to derive a resolution from, or the band is
            not a 2-D raster.
    """
    x_dataset = group[X_COORDINATES_DATASET]
    y_dataset = group[Y_COORDINATES_DATASET]
    if x_dataset.shape[0] < 2 or y_dataset.shape[0] < 2:
        raise ValueError(
            f"Band {band} needs at least two coordinates per axis to derive a "
            f"resolution, got {x_dataset.shape[0]}x{y_dataset.shape[0]}"
        )

    # Evenly spaced, so two coordinates suffice; the full arrays are hundreds of KB.
    x_coordinates = x_dataset[:2]
    y_coordinates = y_dataset[:2]

    shape = group[band].shape
    if len(shape) != 2:
        raise ValueError(f"Expected band {band} to be a 2-D raster, got shape {shape}")
    height, width = shape

    x_resolution = float(x_coordinates[1] - x_coordinates[0])
    y_resolution = float(y_coordinates[1] - y_coordinates[0])
    epsg_code = int(group[PROJECTION_DATASET].attrs[EPSG_CODE_ATTRIBUTE])

    return GranuleGrid(
        crs=CRS.from_epsg(epsg_code),
        x_resolution=x_resolution,
        y_resolution=y_resolution,
        # The coordinate datasets give pixel centers; a transform needs the corner.
        x_origin=float(x_coordinates[0]) - x_resolution / 2,
        y_origin=float(y_coordinates[0]) - y_resolution / 2,
        width=width,
        height=height,
    )


def read_fill_value(group: h5py.Group, band: str) -> float | None:
    """Read a band's fill value from its attributes, without reading pixels.

    Args:
        group: the group holding the band, from :func:`find_band_group`.
        band: the band (HDF5 dataset) name.

    Returns:
        the fill value, or None if the band does not declare one.
    """
    fill_value = group[band].attrs.get(FILL_VALUE_ATTRIBUTE)
    if fill_value is None:
        return None
    return float(fill_value)


def granule_to_geotiff(
    h5_path: str,
    bands: list[str],
    geotiff_path: str,
    product: str = DEFAULT_PRODUCT,
) -> GranuleGrid:
    """Convert bands of a local NISAR granule into one georeferenced GeoTIFF.

    Args:
        h5_path: local path of the HDF5 granule.
        bands: the bands to extract, written to the GeoTIFF in the order given.
        geotiff_path: local path to write the GeoTIFF to.
        product: the L2 product name as it appears in the group path.

    Returns:
        the grid the bands are sampled on, which is also the GeoTIFF's grid.

    Raises:
        ValueError: if no bands were requested, or if the bands do not all share a
            single grid.
    """
    if not bands:
        raise ValueError("No bands requested")

    with h5py.File(h5_path, "r") as granule:
        groups = {band: find_band_group(granule, band, product) for band in bands}
        grid = read_grid(groups[bands[0]], bands[0])
        for band in bands[1:]:
            band_grid = read_grid(groups[band], band)
            if band_grid != grid:
                # The bands would have to go in separate band sets (and so separate
                # GeoTIFFs) to be materialized from different grids.
                raise ValueError(
                    f"Band {band} is on a different grid than {bands[0]}: "
                    f"{band_grid} vs {grid}"
                )

        datasets = [groups[band][band] for band in bands]
        fill_value = read_fill_value(groups[bands[0]], bands[0])
        logger.info(
            f"Converting {len(bands)} band(s) of {h5_path} "
            f"({grid.width}x{grid.height} in {grid.crs}) to {geotiff_path}"
        )

        with rasterio.open(
            geotiff_path,
            "w",
            driver="GTiff",
            width=grid.width,
            height=grid.height,
            count=len(bands),
            dtype=datasets[0].dtype,
            crs=grid.crs,
            transform=grid.transform,
            nodata=fill_value,
            tiled=True,
            blockxsize=GEOTIFF_BLOCK_SIZE,
            blockysize=GEOTIFF_BLOCK_SIZE,
        ) as dest:
            for row_start in range(0, grid.height, BLOCK_ROWS):
                row_stop = min(row_start + BLOCK_ROWS, grid.height)
                block: npt.NDArray[np.floating] = np.stack(
                    [dataset[row_start:row_stop, :] for dataset in datasets]
                )
                dest.write(
                    block,
                    window=rasterio.windows.Window(
                        0, row_start, grid.width, row_stop - row_start
                    ),
                )

    return grid
