"""Tests for the baked-in coverage mask.

These pin the two failures that motivated replacing `global_land_mask`: features it
omitted entirely, and features it knew about but that fell between sample points.
"""

import numpy as np

from rslp.large_scale_embeddings.coverage import (
    COVERAGE_RES_DEG,
    is_covered,
)
from rslp.large_scale_embeddings.tiling import LAND_STEP_SIZE


def _at(lat: float, lon: float) -> bool:
    return bool(is_covered(np.array([lat]), np.array([lon]))[0])


def test_features_the_old_mask_called_ocean_are_covered() -> None:
    """global_land_mask reported these as ocean, so they were never processed.

    Padre Island is a ~3 km barrier island and Marathon sits in the Florida Keys; both
    are real land that the previous mask denied.
    """
    assert _at(27.00, -97.35), "Padre Island TX is missing from the coverage mask"
    assert _at(24.71, -81.09), "Florida Keys (Marathon) is missing"


def test_open_ocean_is_not_covered() -> None:
    """The mask has to exclude something, or it is not a mask.

    These are the emptiest water on Earth, thousands of km from any acquisition.
    """
    assert not _at(-48.9, -123.4), "Point Nemo should not be covered"
    assert not _at(30.0, -40.0), "mid-Atlantic gyre should not be covered"
    assert not _at(-35.0, -120.0), "South Pacific gyre should not be covered"


def test_well_observed_shelf_water_is_covered() -> None:
    """The deep central Gulf has all three sensors every month of every year.

    AlphaEarth leaves it as nodata; the measurement says we can compute it.
    """
    assert _at(25.0, -90.0), "deep central Gulf of Mexico should be covered"


def test_sampling_is_finer_than_the_mask_it_queries() -> None:
    """The bug that finer data alone would not have fixed.

    A crop is kept when a sampled point is covered, so a sample spacing coarser than
    the mask's own cells lets a known feature fall between samples. At 10 m/pixel the
    lattice must stay under the mask's cell size.
    """
    sample_spacing_m = LAND_STEP_SIZE * 10
    cell_m = COVERAGE_RES_DEG * 111_320
    assert sample_spacing_m < cell_m, (
        f"sampling every {sample_spacing_m:.0f} m against {cell_m:.0f} m cells lets "
        "features hide between sample points"
    )


def test_longitudes_wrap_at_the_antimeridian() -> None:
    """Zone 1 and zone 60 both reach the dateline; neither may index out of range."""
    for lon in (-180.0, 180.0, 179.999, -179.999):
        is_covered(np.array([0.0]), np.array([lon]))


def test_lookup_shape_matches_input() -> None:
    """list_kept_crops indexes the result as a 2-D lattice, not a flat list."""
    lats = np.array([[10.0, 10.0], [20.0, 20.0]])
    lons = np.array([[-70.0, -60.0], [-70.0, -60.0]])
    assert is_covered(lats, lons).shape == (2, 2)


def test_enumeration_cache_key_changes_with_the_mask(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A cache built against one mask must never be read back against another.

    The enumeration is only deterministic given the mask, so the mask's identity is
    part of the key. Without this, swapping the mask would silently reuse the old
    block list and the new coverage would never be computed.
    """
    import importlib

    wj = importlib.import_module("rslp.large_scale_embeddings.write_jobs")
    key = wj._enumeration_cache_key(4096, None, None, None)
    assert len(key) == 16

    # same inputs -> same key
    assert key == wj._enumeration_cache_key(4096, None, None, None)
    # any area or sizing change -> different key
    assert key != wj._enumeration_cache_key(8192, None, None, None)
    assert key != wj._enumeration_cache_key(4096, 32610, None, None)
    assert key != wj._enumeration_cache_key(4096, None, (-1.0, 2.0, 3.0, 4.0), None)


def test_enumeration_cache_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A cached enumeration must come back as the same blocks it went in as."""
    import importlib

    from rslearn.utils.geometry import Projection
    from rslearn.utils.get_utm_ups_crs import CRS

    wj = importlib.import_module("rslp.large_scale_embeddings.write_jobs")
    proj = Projection(CRS.from_epsg(32610), 10, -10)
    tasks = [(proj, (0, 0, 4096, 4096)), (proj, (4096, 0, 8192, 4096))]
    wj._write_enumeration_cache(str(tmp_path), "k", tasks)
    got = wj._read_enumeration_cache(str(tmp_path), "k")
    assert got is not None
    assert [b for _, b in got] == [b for _, b in tasks]
    assert [str(p.crs) for p, _ in got] == [str(p.crs) for p, _ in tasks]


def test_missing_enumeration_cache_is_not_an_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A cold cache must enumerate, not crash a supervision cycle."""
    import importlib

    wj = importlib.import_module("rslp.large_scale_embeddings.write_jobs")
    assert wj._read_enumeration_cache(str(tmp_path), "nope") is None


def test_corrupt_enumeration_cache_is_not_an_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A truncated write must degrade to re-enumerating, not fail the run."""
    import importlib

    wj = importlib.import_module("rslp.large_scale_embeddings.write_jobs")
    (tmp_path / "enumeration_bad.json").write_text("{not json")
    assert wj._read_enumeration_cache(str(tmp_path), "bad") is None


def test_mask_resolves_when_rslp_is_installed_elsewhere(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """CI installs rslp to site-packages, where there is no data/ beside the module.

    Deriving the path from __file__ alone resolved to site-packages and every test
    touching the mask failed with RasterioIOError. The working directory is checked
    first because that is what both CI and the image actually provide.
    """
    from rslp.large_scale_embeddings import coverage

    monkeypatch.setattr(
        coverage, "__file__", "/opt/conda/lib/python3.11/site-packages/rslp/x/y.py"
    )
    assert coverage.resolve_mask_path().exists()


def test_missing_mask_raises_rather_than_enumerating_nothing(
    monkeypatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    """A silent miss would enumerate zero blocks and look like a finished run."""
    import pytest

    from rslp.large_scale_embeddings import coverage

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(coverage, "__file__", str(tmp_path / "a" / "b" / "c.py"))
    with pytest.raises(FileNotFoundError):
        coverage.resolve_mask_path()
