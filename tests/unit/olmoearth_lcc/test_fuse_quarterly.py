"""Tests for fusing yearly LCC runs into a quarterly event series."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from upath import UPath

from rslp.olmoearth_lcc.lcc_model import fuse_quarterly as fq

# Runs ending 2022-10-20, 2023-10-20 and 2024-10-20 (quarters 2022Q4..2024Q4).
END_QUARTERS = np.array(
    [fq.quarter_index(date(y, 10, 20)) for y in (2022, 2023, 2024)], np.int16
)
FIRST_QUARTER = fq.quarter_index(date(2022, 1, 1))
NUM_QUARTERS = 12

NEW_BUILDING = fq.EVENT_CATEGORY_NAMES.index("new_building")
TREE = fq.LC_CLASS_NAMES.index("tree")
URBAN = fq.LC_CLASS_NAMES.index("urban/built-up")


def month(year: int, m: int) -> int:
    """Summary month value of a calendar month."""
    return (year - 2015) * 12 + m


def quarter_pos(year: int, m: int) -> int:
    """Position of a month's quarter in the output bands."""
    return fq.quarter_index(date(year, m, 1)) - FIRST_QUARTER


def no_change() -> list[int]:
    """Summary bands of a run that sees no change over tree cover."""
    return [10, 1, 1, TREE, TREE, 250, 250, 0, 0]


def building(score: int, ts_pre: int, ts_post: int) -> list[int]:
    """Summary bands of a run reporting a new building (tree -> urban)."""
    return [score, 1, NEW_BUILDING, TREE, URBAN, 250, 250, ts_pre, ts_post]


def fuse(*runs: list[int] | None) -> np.ndarray:
    """Fuse one pixel; None = the run has no prediction there."""
    stack = np.zeros((len(runs), 9, 1, 1), np.uint8)
    for k, bands in enumerate(runs):
        if bands is not None:
            stack[k, :, 0, 0] = bands
    out = fq.fuse_block(
        stack, END_QUARTERS, FIRST_QUARTER, NUM_QUARTERS, fq.FusionParams()
    )
    assert out.shape == (3 * NUM_QUARTERS + 6, 1, 1)
    return out[:, 0, 0]


def parts(out: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = NUM_QUARTERS
    return out[:n], out[n : 2 * n], out[2 * n : 3 * n], out[3 * n :]


def test_single_run_event_when_no_other_run_covers_it() -> None:
    """A strong report seen by one run is accepted if no covering run disagrees."""
    t = quarter_pos(2023, 2)
    out = fuse(no_change(), building(200, month(2022, 11), month(2023, 2)), None)
    event, lc, conf, summary = parts(out)
    assert event[t] == NEW_BUILDING
    assert np.all(np.delete(event, t) == fq.NONE_CATEGORY_IDX)
    assert np.all(lc[:t] == TREE) and np.all(lc[t:] == URBAN)
    assert conf[t] == 200
    first, last, n_events, last_code, lc_start, lc_end = summary
    assert (first, last, n_events, last_code) == (t + 1, t + 1, 1, NEW_BUILDING)
    assert (lc_start, lc_end) == (TREE, URBAN)


def test_single_run_event_rejected_by_covering_run() -> None:
    """A later run whose window covers the quarter but sees nothing rejects it."""
    t = quarter_pos(2023, 2)
    out = fuse(no_change(), building(200, month(2022, 11), month(2023, 2)), no_change())
    event, lc, conf, summary = parts(out)
    assert np.all(event == fq.NONE_CATEGORY_IDX)
    assert np.all(lc == TREE)
    # Rejected candidates lower the quarter's confidence to 255 * (1 - support).
    assert conf[t] == 128
    assert summary[2] == 0


def test_overlapping_reports_merge_into_one_event() -> None:
    """Two runs' reports more than a quarter apart merge when their intervals overlap.

    The later run's last unchanged image (Oct 2022) predates the earlier run's first
    changed image (Feb 2023), so this is one change, dated at Feb 2023.
    """
    out = fuse(
        no_change(),
        building(200, month(2022, 11), month(2023, 2)),
        building(180, month(2022, 10), month(2023, 8)),
    )
    event, lc, conf, summary = parts(out)
    t = quarter_pos(2023, 2)
    assert event[t] == NEW_BUILDING
    assert summary[2] == 1
    assert conf[t] == 190
    assert np.all(lc[t:] == URBAN)


def test_genuine_repeat_stays_two_events() -> None:
    """A later report whose last unchanged image follows the first change is new."""
    out = fuse(
        no_change(),
        building(200, month(2022, 11), month(2023, 2)),
        building(200, month(2023, 6), month(2024, 2)),
    )
    event, _, _, summary = parts(out)
    t0, t1 = quarter_pos(2023, 2), quarter_pos(2024, 2)
    assert event[t0] == NEW_BUILDING and event[t1] == NEW_BUILDING
    assert (summary[0], summary[1], summary[2]) == (t0 + 1, t1 + 1, 2)


def test_no_prediction_is_nodata() -> None:
    out = fuse(None, None, None)
    assert np.all(out == 0)


def test_quarter_helpers() -> None:
    q = fq.quarter_index(date(2026, 10, 20))
    assert fq.quarter_name(q) == "2026Q4"
    assert fq.quarter_start(q) == date(2026, 10, 1)


def test_layer_path() -> None:
    base = UPath("gs://bucket/lcc")
    assert fq._layer_path(UPath("gs://bucket/lcc/end_2024"), base) == "end_2024/"
    assert (
        fq._layer_path(UPath("gs://other/x/end_2024"), base)
        == "https://storage.googleapis.com/other/x/end_2024/"
    )
    with pytest.raises(ValueError):
        fq._layer_path(UPath("/local/end_2024"), base)


def write_summary_cog(path: Path, bands: list[int], size: int = 64) -> None:
    """A small 9-band summary COG at zoom 14 with a prediction in the top half."""
    res = fq.mercator_resolution(14)
    data = np.zeros((9, size, size), np.uint8)
    data[:, : size // 2, :] = np.array(bands, np.uint8)[:, None, None]
    with rasterio.open(
        path,
        "w",
        driver="COG",
        width=size,
        height=size,
        count=9,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(-1000 * res, 1000 * res, res, res),
    ) as dst:
        dst.write(data)


def test_fuse_runs_end_to_end(tmp_path: Path) -> None:
    """Fused COGs, index.json, overview and manifest from two runs' COG dirs."""
    runs = []
    for year, bands in (
        (2023, building(200, month(2023, 3), month(2023, 5))),
        (2024, building(200, month(2023, 2), month(2023, 8))),
    ):
        run_dir = tmp_path / f"end_{year}1020"
        run_dir.mkdir()
        names = ["EPSG:32630_0_0.tif"] + (
            ["EPSG:32630_2048_0.tif"] if year == 2024 else []
        )
        for name in names:
            write_summary_cog(run_dir / name, bands)
        entries = []
        for name in names:
            with rasterio.open(run_dir / name) as src:
                entries.append(
                    {"cog": name, "bounds3857": list(src.bounds), "max_zoom": 14}
                )
        (run_dir / "index.json").write_text(json.dumps(entries))
        runs.append(fq.Run(UPath(run_dir), date(year, 10, 20)))

    out_dir = UPath(tmp_path / "quarterly")
    fq.fuse_runs(runs, out_dir, UPath(tmp_path / "manifest.json"), workers=1)

    # Default quarters: 2023Q1 (3 before the first end date's quarter) to 2024Q4.
    num_quarters = 8
    with rasterio.open(tmp_path / "quarterly" / "EPSG:32630_0_0.tif") as src:
        assert src.count == 3 * num_quarters + 6
        assert src.descriptions[0] == "event_2023Q1"
        assert src.descriptions[-1] == "lc_end"
        assert src.profile["interleave"] == "band"
        data = src.read()
    # One merged event in 2023Q2 (the earliest changed image, May 2023).
    assert data[1, 0, 0] == NEW_BUILDING
    assert data[3 * num_quarters + 2, 0, 0] == 1
    # The bottom half has no prediction in either run.
    assert np.all(data[:, -1] == 0)

    index = json.loads((tmp_path / "quarterly" / "index.json").read_text())
    assert [e["cog"] for e in index] == [
        "EPSG:32630_0_0.tif",
        "EPSG:32630_2048_0.tif",
        "overview.tif",
    ]
    assert index[0]["max_zoom"] == 14
    assert index[-1]["overview"] is True and index[-1]["max_zoom"] == 7

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "title" not in manifest and "description" not in manifest
    assert [layer["path"] for layer in manifest["layers"]] == [
        "quarterly/",
        "end_20231020/",
        "end_20241020/",
    ]
    fused = manifest["layers"][0]
    assert fused["kind"] == "quarterly"
    assert fused["quarters"][0] == "2023Q1" and len(fused["quarters"]) == num_quarters
    assert fused["bands"] == {"event": 0, "lc": 8, "conf": 16, "summary": 24}
    assert "2024 Q4 covers Oct 1 to Oct 20 only." in fused["description"]
    assert manifest["layers"][1]["end_date"] == "2023-10-20"
    west, south, east, north = manifest["bounds"]
    assert west < east and south < north
