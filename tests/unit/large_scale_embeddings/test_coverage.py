"""Tests for the baked-in coverage mask and the enumeration cache."""

from pathlib import Path

import numpy as np

from rslp.large_scale_embeddings.coverage import COVERAGE_RES_DEG, is_covered
from rslp.large_scale_embeddings.tiling import LAND_STEP_SIZE
from rslp.large_scale_embeddings.write_jobs import (
    _enumeration_cache_key,
    _read_enumeration_cache,
    _write_enumeration_cache,
)


def _at(lat: float, lon: float) -> bool:
    return bool(is_covered(np.array([lat]), np.array([lon]))[0])


def test_mask_covers_land_the_old_rule_missed_and_excludes_open_ocean() -> None:
    """The two failures that motivated replacing `global_land_mask`.

    It called Padre Island and the Florida Keys ocean, so they were never processed.
    The open-ocean cases are the other half: a mask that keeps everything is no mask.
    """
    assert _at(27.00, -97.35), "Padre Island TX is missing"
    assert _at(24.71, -81.09), "Florida Keys (Marathon) is missing"
    assert _at(25.00, -90.00), "deep central Gulf of Mexico is missing"
    assert not _at(-48.9, -123.4), "Point Nemo should not be covered"
    assert not _at(30.0, -40.0), "mid-Atlantic gyre should not be covered"


def test_sampling_is_finer_than_the_mask_it_queries() -> None:
    """A crop is kept when a sampled point is covered.

    So a sample spacing coarser than the mask's own cells lets a feature the mask does
    know about fall between samples. That is how 3 km barrier islands went missing
    under the old 2.56 km lattice, and it would happen again if this regressed.
    """
    sample_spacing_m = LAND_STEP_SIZE * 10
    cell_m = COVERAGE_RES_DEG * 111_320
    assert sample_spacing_m < cell_m, (
        f"sampling every {sample_spacing_m:.0f} m against {cell_m:.0f} m cells lets "
        "features hide between sample points"
    )


def test_enumeration_cache_key_covers_everything_that_changes_the_result() -> None:
    """A cache built for one mask or area must never be read back for another.

    The enumeration is only deterministic given the mask and the area arguments, so
    both go into the key. Otherwise swapping the mask silently reuses the old blocks
    and the new coverage is never computed.
    """
    key = _enumeration_cache_key(4096, None, None, None)
    assert key == _enumeration_cache_key(4096, None, None, None)
    assert key != _enumeration_cache_key(8192, None, None, None)
    assert key != _enumeration_cache_key(4096, 32610, None, None)
    assert key != _enumeration_cache_key(4096, None, (-1.0, 2.0, 3.0, 4.0), None)


def test_enumeration_cache_round_trips(tmp_path: Path) -> None:
    """Cached blocks must come back as the blocks that went in."""
    from rslearn.utils.geometry import Projection
    from rslearn.utils.get_utm_ups_crs import CRS

    projection = Projection(CRS.from_epsg(32610), 10, -10)
    tasks = [(projection, (0, 0, 4096, 4096)), (projection, (4096, 0, 8192, 4096))]
    _write_enumeration_cache(str(tmp_path), "k", tasks)

    got = _read_enumeration_cache(str(tmp_path), "k")
    assert got is not None
    assert [bounds for _, bounds in got] == [bounds for _, bounds in tasks]
    assert [str(p.crs) for p, _ in got] == [str(p.crs) for p, _ in tasks]


def test_unusable_enumeration_cache_falls_back_to_enumerating(tmp_path: Path) -> None:
    """A cold or corrupt cache costs time, not a failed supervision cycle."""
    assert _read_enumeration_cache(str(tmp_path), "absent") is None

    (tmp_path / "enumeration_bad.json").write_text("{not json")
    assert _read_enumeration_cache(str(tmp_path), "bad") is None
