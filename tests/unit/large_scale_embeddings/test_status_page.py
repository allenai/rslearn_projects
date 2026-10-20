"""Unit tests for rslp.large_scale_embeddings.status_page."""

import io
import json
import pathlib

import pytest
import shapely
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
    completed: pathlib.Path,
    block: tuple,
    gpu_seconds: float | None,
    crops: int,
    worker: str | None = None,
) -> None:
    fname = get_marker_fname(str(completed), *block)
    fname.parent.mkdir(parents=True, exist_ok=True)
    marker: dict = {"written": [[0, 0]] * crops}
    if gpu_seconds is not None:
        marker["gpu_seconds"] = gpu_seconds
    if worker is not None:
        marker["worker"] = worker
    fname.write_text(json.dumps(marker))


def test_year_colors_span_the_spectrum() -> None:
    """Nine years take one stop each, oldest red and newest violet."""
    colors = status_page.year_colors(list(range(2017, 2026)))
    assert colors[2017] == status_page.SPECTRUM[0]
    assert colors[2025] == status_page.SPECTRUM[-1]
    assert len(set(colors.values())) == 9


def test_read_markers_forgets_a_few_deleted_markers(tmp_path: pathlib.Path) -> None:
    """Markers deleted to be recomputed are dropped, not taken for a truncation.

    Deleting markers to recompute their blocks stopped the page rebuilding for good,
    first for five out of 23,000 and again for 1,415, since the cache still named them.
    """
    completed = tmp_path / "completed_2025"
    cache = UPath(tmp_path / "cache.json")
    blocks = [(BLOCKS[0][0], (2048 * i, 0, 2048 * (i + 1), 2048)) for i in range(12)]
    for block in blocks:
        _write_marker(completed, block, 60.0, 16)
    assert len(status_page.read_markers(str(completed), cache)) == 12

    get_marker_fname(str(completed), *blocks[3]).unlink()
    assert len(status_page.read_markers(str(completed), cache)) == 11
    # Rewritten by the recompute, it is read again.
    _write_marker(completed, blocks[3], 90.0, 16)
    assert status_page.read_markers(str(completed), cache)[
        get_marker_fname(str(completed), *blocks[3]).name
    ] == (90.0, 16, False)

    # Thousands can be deleted at once to recompute them; that is not a truncation.
    for block in blocks[:7]:
        get_marker_fname(str(completed), *block).unlink()
    assert len(status_page.read_markers(str(completed), cache)) == 5


def test_read_markers_refuses_a_listing_that_omits_markers_still_there(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listing cut short leaves out markers that still exist, and must not publish."""
    completed = tmp_path / "completed_2025"
    cache = UPath(tmp_path / "cache.json")
    blocks = [(BLOCKS[0][0], (2048 * i, 0, 2048 * (i + 1), 2048)) for i in range(12)]
    for block in blocks:
        _write_marker(completed, block, 60.0, 16)
    names = status_page.read_markers(str(completed), cache)

    kept = sorted(names)[:4]
    monkeypatch.setattr(status_page, "_list_names", lambda completed: set(kept))
    with pytest.raises(status_page.TruncatedListingError):
        status_page.read_markers(str(completed), cache)
    # The cache is untouched, so the next good listing still finds every marker.
    monkeypatch.undo()
    assert len(status_page.read_markers(str(completed), cache)) == 12


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
        vector_max_blocks=1,
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

    for fname in ("2025.png", "coverage.png"):
        image = Image.open(io.BytesIO((out / fname).read_bytes()))
        assert image.size == (status_page.MAP_WIDTH, status_page.MAP_HEIGHT)
    assert json.loads((out / "countries.json").read_text())["features"]
    assert data["map"]["crs"] == "EPSG:8857"
    assert Image.open(out / "2025.png").getbbox() is not None

    # Years with few blocks come as outlines, with no world-sized image behind them.
    by_year = {layer["year"]: layer for layer in data["layers"]}
    assert not (out / "2023.png").exists() and not (out / "2024.png").exists()
    assert by_year[2023]["outlines"] == {"type": "MultiPolygon", "coordinates": []}
    (polygon,) = by_year[2024]["outlines"]["coordinates"]
    outline = shapely.Polygon(polygon[0])
    expected = status_page._block_lonlat(*BLOCKS[0])
    # Equal up to rounding to OUTLINE_DECIMALS places, about 10 m.
    assert outline.symmetric_difference(expected).area < 1e-3 * expected.area
    assert "file" in by_year[2025] and "outlines" not in by_year[2025]


def test_every_year_is_a_share_of_the_whole_area(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Percentages share one denominator: everything any year covers.

    A year limited to a small area must read as a small share of the world, not as a
    large share of its own list.
    """
    monkeypatch.setattr(
        status_page,
        "enumerate_blocks",
        lambda **kwargs: BLOCKS
        if kwargs["blocks_fname"] == "world.json"
        else BLOCKS[:1],
    )
    _write_marker(tmp_path / "completed_2025", BLOCKS[1], 600.0, 16)
    _write_marker(tmp_path / "completed_2019", BLOCKS[0], 600.0, 16)
    # Outside 2019's own list, so it does not count for 2019.
    _write_marker(tmp_path / "completed_2019", BLOCKS[1], 600.0, 16)

    out = tmp_path / "status"
    status_page.publish_status(
        status_path=str(out),
        years=[2019, 2025],
        completed_path_template=str(tmp_path / "completed_{year}"),
        title="Test coverage",
        cache_dir=str(tmp_path / "cache"),
        blocks_fname="small.json",
        year_blocks_fname={2025: "world.json"},
    )

    html = (out / "index.html").read_text()
    data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    totals = {
        layer["year"]: (layer["done"], layer["total"]) for layer in data["layers"]
    }
    assert totals == {2019: (1, 2), 2025: (1, 2)}


def test_km2_per_gpu_hour_is_split_by_pool(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GCE blocks count as outside Beaker; Beaker and unnamed markers as Beaker."""
    monkeypatch.setattr(status_page, "enumerate_blocks", lambda **kwargs: BLOCKS)
    _write_marker(tmp_path / "completed_2025", BLOCKS[0], 3600.0, 16, "worker_x_1")
    _write_marker(tmp_path / "completed_2025", BLOCKS[1], 7200.0, 16, "gce_embed-1")
    _write_marker(tmp_path / "completed_2024", BLOCKS[0], 3600.0, 16)

    out = tmp_path / "status"
    status_page.publish_status(
        status_path=str(out),
        years=[2024, 2025],
        completed_path_template=str(tmp_path / "completed_{year}"),
        title="Test coverage",
        cache_dir=str(tmp_path / "cache"),
    )

    html = (out / "index.html").read_text()
    data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    km2 = status_page.KM2_PER_CROP
    assert data["km2_per_gpu_hour_by_pool"] == {
        "beaker": round(32 * km2 / 2.0),
        "outside": round(16 * km2 / 2.0),
    }
    assert data["km2_per_gpu_hour"] == round(48 * km2 / 4.0)


def test_a_cache_without_the_pool_field_is_reread(tmp_path: pathlib.Path) -> None:
    """A cache from before the split is discarded, so no marker is misfiled."""
    completed = tmp_path / "completed"
    cache = tmp_path / "cache.json"
    _write_marker(completed, BLOCKS[0], 60.0, 16, "gce_embed-1")
    name = get_marker_fname(str(completed), *BLOCKS[0]).name
    cache.write_text(json.dumps({name: [60.0, 16]}))
    assert status_page.read_markers(str(completed), cache)[name] == (60.0, 16, True)


def test_upload_is_never_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A remote write carries no-store, so no CDN or browser keeps a stale page."""
    calls: list[dict] = []

    class _Fs:
        def pipe_file(self, path: str, data: bytes, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(
        status_page, "url_to_fs", lambda url: (_Fs(), url.split("://", 1)[1])
    )
    status_page._upload("gs://bucket/status/index.html", b"<html>", "text/html")
    assert calls == [
        {
            "content_type": "text/html",
            "fixed_key_metadata": {"cache_control": "no-store"},
        }
    ]
