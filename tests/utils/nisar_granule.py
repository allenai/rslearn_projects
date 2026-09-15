"""Building synthetic NISAR GCOV HDF5 granules for tests.

The grid mirrors how NISAR posts GCOV grids: the coordinate datasets hold pixel centers,
and corners land on multiples of the spacing, which keeps the grid aligned with
rslearn's pixel bounds.
"""

import pathlib
from collections.abc import Sequence

import h5py
import numpy as np
import numpy.typing as npt

from rslp.nisar_vessels import hdf5

# A UTM zone somewhere off the US west coast, unless a test picks its own.
EPSG_CODE = 32610
X_RESOLUTION = 20.0
Y_RESOLUTION = -20.0
X_ORIGIN = 500000.0
Y_ORIGIN = 4200000.0
WIDTH = 40
HEIGHT = 30
FILL_VALUE = -9999.0

GCOV_GROUP = f"science/{hdf5.INSTRUMENT_GROUP}/{hdf5.DEFAULT_PRODUCT}/grids/frequencyA"
GCOV_GROUP_FREQUENCY_B = (
    f"science/{hdf5.INSTRUMENT_GROUP}/{hdf5.DEFAULT_PRODUCT}/grids/frequencyB"
)
DUAL_POL_BANDS = ("HHHH", "HVHV")


def position_encoded_bands(
    bands: Sequence[str], width: int, height: int
) -> dict[str, npt.NDArray[np.float32]]:
    """Build band arrays whose pixels encode their own position.

    Each pixel holds ``row * 1000 + col``, plus ``100000`` per band, so a test can assert
    both which source pixels a read returned and which band they came from.

    Args:
        bands: the band names, in order.
        width: the grid width in pixels.
        height: the grid height in pixels.

    Returns:
        map from band name to its array.
    """
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return {
        band: (rows * 1000 + cols + offset * 100000).astype(np.float32)
        for offset, band in enumerate(bands)
    }


def write_granule(
    path: pathlib.Path,
    group_path: str = GCOV_GROUP,
    bands: Sequence[str] = DUAL_POL_BANDS,
    data: dict[str, npt.NDArray[np.float32]] | None = None,
    epsg_code: int = EPSG_CODE,
    x_origin: float = X_ORIGIN,
    y_origin: float = Y_ORIGIN,
    x_resolution: float = X_RESOLUTION,
    y_resolution: float = Y_RESOLUTION,
    width: int = WIDTH,
    height: int = HEIGHT,
    fill_value: float | None = FILL_VALUE,
) -> None:
    """Write a NISAR-like GCOV granule, appending the group if the file already exists.

    Args:
        path: the file to write.
        group_path: the HDF5 group to write the grid into.
        bands: the band datasets to create, in order.
        data: band arrays to write. Defaults to :func:`position_encoded_bands`.
        epsg_code: the EPSG code the grid is projected in.
        x_origin: the x coordinate of the outer corner of the first pixel.
        y_origin: the y coordinate of the outer corner of the first pixel.
        x_resolution: the x pixel spacing.
        y_resolution: the y pixel spacing, negative for a north-up grid.
        width: the grid width in pixels.
        height: the grid height in pixels.
        fill_value: the band fill value, or None to declare none.
    """
    if data is None:
        data = position_encoded_bands(bands, width, height)

    with h5py.File(path, "a") as granule:
        group = granule.create_group(group_path)
        group.create_dataset(
            hdf5.X_COORDINATES_DATASET,
            # The coordinate datasets give pixel centers, not corners.
            data=x_origin
            + x_resolution / 2
            + x_resolution * np.arange(width, dtype=np.float64),
        )
        group.create_dataset(
            hdf5.Y_COORDINATES_DATASET,
            data=y_origin
            + y_resolution / 2
            + y_resolution * np.arange(height, dtype=np.float64),
        )
        projection = group.create_dataset(hdf5.PROJECTION_DATASET, data=np.int32(1))
        projection.attrs[hdf5.EPSG_CODE_ATTRIBUTE] = np.int32(epsg_code)

        for band in bands:
            dataset = group.create_dataset(band, data=data[band])
            if fill_value is not None:
                dataset.attrs[hdf5.FILL_VALUE_ATTRIBUTE] = np.float32(fill_value)
