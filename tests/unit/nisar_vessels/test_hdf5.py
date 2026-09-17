"""Unit tests for reading NISAR L2 GCOV granules out of local HDF5 files."""

import pathlib

import h5py
import numpy as np
import pytest
import rasterio

from rslp.nisar_vessels import hdf5
from tests.utils.nisar_granule import (
    EPSG_CODE,
    FILL_VALUE,
    GCOV_GROUP,
    GCOV_GROUP_FREQUENCY_B,
    HEIGHT,
    WIDTH,
    X_ORIGIN,
    X_RESOLUTION,
    Y_ORIGIN,
    Y_RESOLUTION,
    position_encoded_bands,
    write_granule,
)

BANDS = ["HHHH", "HVHV"]


def test_geotiff_matches_the_granule_grid(tmp_path: pathlib.Path) -> None:
    """The GeoTIFF carries the granule's CRS, transform, nodata and pixel values."""
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path)
    geotiff_path = tmp_path / "granule.tif"

    grid = hdf5.granule_to_geotiff(str(h5_path), BANDS, str(geotiff_path))

    assert grid.crs.to_epsg() == EPSG_CODE
    assert (grid.width, grid.height) == (WIDTH, HEIGHT)
    # The transform has to come from the pixel corner, not the center the coordinate
    # datasets hold, or everything lands half a pixel off.
    assert grid.x_origin == pytest.approx(X_ORIGIN)
    assert grid.y_origin == pytest.approx(Y_ORIGIN)
    assert grid.bounds == pytest.approx(
        (
            X_ORIGIN,
            Y_ORIGIN + HEIGHT * Y_RESOLUTION,
            X_ORIGIN + WIDTH * X_RESOLUTION,
            Y_ORIGIN,
        )
    )

    with rasterio.open(geotiff_path) as raster:
        assert raster.crs.to_epsg() == EPSG_CODE
        assert raster.count == len(BANDS)
        assert (raster.width, raster.height) == (WIDTH, HEIGHT)
        assert raster.nodata == pytest.approx(FILL_VALUE)
        assert raster.transform == grid.transform
        expected = position_encoded_bands(BANDS, WIDTH, HEIGHT)
        for band_idx, band in enumerate(BANDS):
            np.testing.assert_array_equal(raster.read(band_idx + 1), expected[band])


def test_bands_are_written_in_the_requested_order(tmp_path: pathlib.Path) -> None:
    """Band order follows the argument, not the order they appear in the granule."""
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path)
    geotiff_path = tmp_path / "granule.tif"

    hdf5.granule_to_geotiff(str(h5_path), ["HVHV", "HHHH"], str(geotiff_path))

    expected = position_encoded_bands(BANDS, WIDTH, HEIGHT)
    with rasterio.open(geotiff_path) as raster:
        np.testing.assert_array_equal(raster.read(1), expected["HVHV"])
        np.testing.assert_array_equal(raster.read(2), expected["HHHH"])


def test_band_is_found_in_frequency_b(tmp_path: pathlib.Path) -> None:
    """A granule that puts its bands in frequencyB is read just the same."""
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path, group_path=GCOV_GROUP_FREQUENCY_B)
    geotiff_path = tmp_path / "granule.tif"

    grid = hdf5.granule_to_geotiff(str(h5_path), BANDS, str(geotiff_path))

    assert (grid.width, grid.height) == (WIDTH, HEIGHT)


def test_rows_beyond_one_block_are_written(tmp_path: pathlib.Path) -> None:
    """A granule taller than one conversion block comes through complete.

    Bands are copied a block of rows at a time to bound memory, so the block seam is
    worth covering: a mistake there silently truncates or repeats rows.
    """
    height = hdf5.BLOCK_ROWS + 17
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path, height=height)
    geotiff_path = tmp_path / "granule.tif"

    hdf5.granule_to_geotiff(str(h5_path), BANDS, str(geotiff_path))

    expected = position_encoded_bands(BANDS, WIDTH, height)
    with rasterio.open(geotiff_path) as raster:
        assert raster.height == height
        for band_idx, band in enumerate(BANDS):
            np.testing.assert_array_equal(raster.read(band_idx + 1), expected[band])


def test_another_l2_product_can_be_read(tmp_path: pathlib.Path) -> None:
    """The product name is a parameter, so GCOV is not baked into the group path."""
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path, group_path="science/LSAR/GSLC/grids/frequencyA")
    geotiff_path = tmp_path / "granule.tif"

    grid = hdf5.granule_to_geotiff(
        str(h5_path), BANDS, str(geotiff_path), product="GSLC"
    )

    assert (grid.width, grid.height) == (WIDTH, HEIGHT)


def test_missing_band_raises(tmp_path: pathlib.Path) -> None:
    """A single-pol granule is rejected rather than silently producing one band.

    Roughly all NISAR science acquisitions are dual-pol, but a granule that is not
    carries no HVHV at all, and the detector needs both bands.
    """
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path, bands=["HHHH"])

    with pytest.raises(ValueError, match="No group in the granule has band HVHV"):
        hdf5.granule_to_geotiff(str(h5_path), BANDS, str(tmp_path / "granule.tif"))


def test_bands_on_different_grids_raise(tmp_path: pathlib.Path) -> None:
    """Bands posted on different grids cannot share one GeoTIFF."""
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path, bands=["HHHH"])
    write_granule(
        h5_path,
        group_path=GCOV_GROUP_FREQUENCY_B,
        bands=["HVHV"],
        x_resolution=X_RESOLUTION * 2,
    )

    with pytest.raises(ValueError, match="is on a different grid than"):
        hdf5.granule_to_geotiff(str(h5_path), BANDS, str(tmp_path / "granule.tif"))


def test_no_bands_requested_raises(tmp_path: pathlib.Path) -> None:
    """Asking for nothing is a caller bug, not an empty GeoTIFF."""
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path)

    with pytest.raises(ValueError, match="No bands requested"):
        hdf5.granule_to_geotiff(str(h5_path), [], str(tmp_path / "granule.tif"))


def test_granule_without_fill_value(tmp_path: pathlib.Path) -> None:
    """A band that declares no fill value produces a GeoTIFF with no nodata."""
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path, fill_value=None)
    geotiff_path = tmp_path / "granule.tif"

    hdf5.granule_to_geotiff(str(h5_path), BANDS, str(geotiff_path))

    with rasterio.open(geotiff_path) as raster:
        assert raster.nodata is None


def test_grid_too_small_to_derive_resolution_raises(tmp_path: pathlib.Path) -> None:
    """A one-pixel-wide grid gives no spacing to build a transform from."""
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path, width=1, height=1)

    with pytest.raises(ValueError, match="at least two x coordinates"):
        hdf5.granule_to_geotiff(str(h5_path), BANDS, str(tmp_path / "granule.tif"))


@pytest.mark.parametrize("axis", ["x", "y"])
def test_irregular_grid_spacing_raises(axis: str, tmp_path: pathlib.Path) -> None:
    """A grid whose coordinates are not evenly spaced is rejected.

    One affine transform can only describe a regular grid. Deriving a resolution from
    the first two coordinates of an irregular one would succeed and put every pixel, and
    so every detection, in the wrong place.
    """
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path)

    # Nudge one coordinate well past the tolerance, leaving the rest evenly spaced.
    dataset_name = "xCoordinates" if axis == "x" else "yCoordinates"
    with h5py.File(h5_path, "a") as granule:
        coordinates = granule[f"{GCOV_GROUP}/{dataset_name}"]
        coordinates[len(coordinates) // 2] += X_RESOLUTION / 2

    with pytest.raises(ValueError, match=f"irregular {axis} grid"):
        hdf5.granule_to_geotiff(str(h5_path), BANDS, str(tmp_path / "granule.tif"))


def test_float_noise_in_coordinates_is_tolerated(tmp_path: pathlib.Path) -> None:
    """Spacing that wobbles at float precision is still a regular grid."""
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path)

    with h5py.File(h5_path, "a") as granule:
        coordinates = granule[f"{GCOV_GROUP}/xCoordinates"]
        coordinates[len(coordinates) // 2] += X_RESOLUTION * 1e-9

    grid = hdf5.granule_to_geotiff(str(h5_path), BANDS, str(tmp_path / "granule.tif"))

    assert grid.x_resolution == pytest.approx(X_RESOLUTION)


def test_south_up_grid_keeps_its_orientation(tmp_path: pathlib.Path) -> None:
    """A grid whose y coordinates increase is transformed without flipping the data.

    y_resolution is signed precisely so this case needs no special handling.
    """
    h5_path = tmp_path / "granule.h5"
    write_granule(h5_path, y_resolution=-Y_RESOLUTION)
    geotiff_path = tmp_path / "granule.tif"

    grid = hdf5.granule_to_geotiff(str(h5_path), BANDS, str(geotiff_path))

    assert grid.y_resolution == pytest.approx(-Y_RESOLUTION)
    assert grid.bounds == pytest.approx(
        (
            X_ORIGIN,
            Y_ORIGIN,
            X_ORIGIN + WIDTH * X_RESOLUTION,
            Y_ORIGIN + HEIGHT * -Y_RESOLUTION,
        )
    )
    expected = position_encoded_bands(BANDS, WIDTH, HEIGHT)
    with rasterio.open(geotiff_path) as raster:
        np.testing.assert_array_equal(raster.read(1), expected["HHHH"])
