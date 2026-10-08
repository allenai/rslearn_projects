"""Unit tests for rslp.large_scale_embeddings.status_page."""

import io
import json
import pathlib

import pytest
from PIL import Image
from rasterio.crs import CRS
from rslearn.utils.geometry import Projection
from upath import UPath

from rslp.large_scale_embeddings import status_page
from rslp.large_scale_embeddings.predict_pipeline import get_marker_fname

PROJECTION = Projection(CRS.from_epsg(32637), 10, -10)
# Two 8192 px blocks near Nairobi, on the store's absolute pixel grid.
BLOCKS = [
    (PROJECTION, (24576, 0, 32768, 8192)),
    (PROJECTION, (32768, 0, 40960, 8192)),
]


def _write_marker(
    completed: pathlib.Path, block: tuple, gpu_seconds: float | None, crops: int
) -> None:
    fname = get_marker_fname(str(completed), *block)
    fname.parent.mkdir(parents=True, exist_ok=True)
    marker: dict = {"written": [[0, 0]] * crops}
    if gpu_seconds is not None:
        marker["gpu_seconds"] = gpu_seconds
    fname.write_text(json.dumps(marker))


def test_year_colors_span_the_spectrum() -> None:
    """Nine years take one stop each, oldest red and newest violet."""
    colors = status_page.year_colors(list(range(2017, 2026)))
    assert colors[2017] == status_page.SPECTRUM[0]
    assert colors[2025] == status_page.SPECTRUM[-1]
    assert len(set(colors.values())) == 9


def test_read_markers_refuses_a_shrunken_listing(tmp_path: pathlib.Path) -> None:
    """A marker seen before and missing now means a truncated read."""
    completed = tmp_path / "completed_2025"
    cache = UPath(tmp_path / "cache.json")
    _write_marker(completed, BLOCKS[0], 60.0, 16)
    _write_marker(completed, BLOCKS[1], 30.0, 4)
    assert len(status_page.read_markers(str(completed), cache)) == 2

    get_marker_fname(str(completed), *BLOCKS[1]).unlink()
    with pytest.raises(status_page.TruncatedListingError):
        status_page.read_markers(str(completed), cache)


def test_publish_status(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The page reports per-year progress and sums GPU time from the markers."""
    monkeypatch.setattr(status_page, "enumerate_blocks", lambda **kwargs: BLOCKS)
    _write_marker(tmp_path / "completed_2025", BLOCKS[0], 3600.0, 16)
    _write_marker(tmp_path / "completed_2025", BLOCKS[1], 1800.0, 16)
    _write_marker(tmp_path / "completed_2024", BLOCKS[0], None, 16)

    out = tmp_path / "status"
    status_page.publish_status(
        status_path=str(out),
        years=[2023, 2024, 2025],
        completed_path_template=str(tmp_path / "completed_{year}"),
        title="Test coverage",
        cache_dir=str(tmp_path / "cache"),
        workers={"working": 3, "allocated": [], "spare": 0, "outside": 0, "waiting": 1},
    )

    html = (out / "index.html").read_text()
    data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert data["title"] == "Test coverage"
    assert data["workers"]["working"] == 3
    assert [(layer["year"], layer["done"]) for layer in data["layers"]] == [
        (2023, 0),
        (2024, 1),
        (2025, 2),
    ]
    assert all(layer["total"] == 2 for layer in data["layers"])
    # The 2024 marker predates gpu_seconds, so it is left out of both stats.
    assert data["gpu_hours"] == round(1.5)
    assert data["km2_per_gpu_hour"] == round(32 * status_page.KM2_PER_CROP / 1.5)

    for fname in ("2023.png", "2024.png", "2025.png", "coverage.png"):
        image = Image.open(io.BytesIO((out / fname).read_bytes()))
        assert image.size == (status_page.MAP_PIXELS, status_page.MAP_PIXELS)
    assert Image.open(out / "2023.png").getbbox() is None
    assert Image.open(out / "2025.png").getbbox() is not None
