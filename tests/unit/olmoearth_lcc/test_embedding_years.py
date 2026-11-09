"""Tests for the annual embedding year selection used by the LCC stack transforms."""

from datetime import datetime, timezone

import torch
from rslearn.train.model_context import RasterImage

from rslp.olmoearth_lcc.lcc_model.transforms import (
    EMBEDDING_DIMS,
    EMBEDDING_NODATA,
    EMBEDDING_YEARS,
    select_embedding_years,
)

H = W = 4


def _raster() -> RasterImage:
    """Window raster whose values encode the year index (year 2017 -> 0, ...)."""
    n = len(EMBEDDING_YEARS)
    image = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1, 1)
    image = image.expand(n, EMBEDDING_DIMS, 1, H, W).reshape(n * EMBEDDING_DIMS, 1, H, W)
    return RasterImage(image=image.clone())


def _years(img: RasterImage) -> list[int]:
    assert img.timestamps is not None
    return [t0.year for t0, _ in img.timestamps]


def test_selects_years_ending_before_boundary() -> None:
    boundary = datetime(2024, 3, 10, tzinfo=timezone.utc)
    out = select_embedding_years(_raster(), boundary, 4)
    assert out.image.shape == (EMBEDDING_DIMS, 4, H, W)
    assert _years(out) == [2020, 2021, 2022, 2023]
    for k, year in enumerate(_years(out)):
        assert (out.image[:, k] == EMBEDDING_YEARS.index(year)).all()
    # Each slot covers one calendar year, all ending on or before the boundary.
    assert out.timestamps is not None
    assert all(t1 <= boundary for _, t1 in out.timestamps)


def test_boundary_on_jan_1_includes_previous_year() -> None:
    out = select_embedding_years(_raster(), datetime(2024, 1, 1, tzinfo=timezone.utc), 2)
    assert _years(out) == [2022, 2023]


def test_drop_latest_simulates_lag() -> None:
    boundary = datetime(2024, 3, 10, tzinfo=timezone.utc)
    out = select_embedding_years(_raster(), boundary, 4, drop_latest=True)
    assert _years(out) == [2019, 2020, 2021, 2022]


def test_years_outside_store_are_nodata() -> None:
    out = select_embedding_years(_raster(), datetime(2018, 6, 1, tzinfo=timezone.utc), 4)
    assert _years(out) == [2014, 2015, 2016, 2017]
    assert (out.image[:, :3] == EMBEDDING_NODATA).all()
    assert (out.image[:, 3] == 0).all()
    out = select_embedding_years(_raster(), datetime(2027, 6, 1, tzinfo=timezone.utc), 2)
    assert _years(out) == [2025, 2026]
    assert (out.image[:, 0] == len(EMBEDDING_YEARS) - 1).all()
    assert (out.image[:, 1] == EMBEDDING_NODATA).all()


def test_missing_raster_is_all_nodata() -> None:
    out = select_embedding_years(
        None, datetime(2024, 3, 10, tzinfo=timezone.utc), 3, image_hw=(H, W)
    )
    assert out.image.shape == (EMBEDDING_DIMS, 3, H, W)
    assert (out.image == EMBEDDING_NODATA).all()
