"""Tests for the baked-in coverage mask."""

import numpy as np

from rslp.large_scale_embeddings.coverage import COVERAGE_RES_DEG, is_covered
from rslp.large_scale_embeddings.tiling import LAND_STEP_SIZE


def _at(lat: float, lon: float) -> bool:
    return bool(is_covered(np.array([lat]), np.array([lon]))[0])


def test_mask_covers_small_islands_and_excludes_open_ocean() -> None:
    """Narrow islands are covered, open ocean is not."""
    assert _at(27.00, -97.35), "Padre Island TX is missing"
    assert _at(24.71, -81.09), "Florida Keys (Marathon) is missing"
    assert _at(25.00, -90.00), "deep central Gulf of Mexico is missing"
    assert not _at(-48.9, -123.4), "Point Nemo should not be covered"
    assert not _at(30.0, -40.0), "mid-Atlantic gyre should not be covered"


def test_sampling_is_finer_than_the_mask_it_queries() -> None:
    """Sample spacing must be below the mask's cell size, or covered cells are missed."""
    sample_spacing_m = LAND_STEP_SIZE * 10
    cell_m = COVERAGE_RES_DEG * 111_320
    assert sample_spacing_m < cell_m
