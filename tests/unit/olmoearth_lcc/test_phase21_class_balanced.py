import json
import pathlib
from datetime import datetime, timezone

import numpy as np
import pytest
from upath import UPath

from rslp.olmoearth_lcc.annotation_scripts.phase21_class_balanced import (
    create_class_balanced_annotations as phase21,
)
from rslp.olmoearth_lcc.lcc_model import geozarr
from rslp.olmoearth_lcc.lcc_model.output_bands import (
    BINARY_CHANGE_BAND,
    DST_BAND_OFFSET,
    LC_CLASS_NAMES,
    NONE_CATEGORY_IDX,
    OUTPUT_BANDS,
    POST_CHANGE_BAND_OFFSET,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_BAND_OFFSET,
    PRE_CHANGE_CATEGORY_NAMES,
    SRC_BAND_OFFSET,
    TS_POST_DAYS_BAND,
    TS_PRE_DAYS_BAND,
)
from rslp.olmoearth_lcc.lcc_model.predict_pipeline import get_meta_fname
from rslp.olmoearth_lcc.lcc_model.tiling import (
    PATCH_SIZE,
    RESOLUTION,
    TILE_SIZE,
    enumerate_zone_tiles,
    get_zone_projection,
)

TIMESTAMP = datetime(2025, 6, 1, tzinfo=timezone.utc)
NEW_BUILDING = POST_CHANGE_CATEGORY_NAMES.index("new_building")
FOREST_LOSS = PRE_CHANGE_CATEGORY_NAMES.index("forest_loss")
BARE = LC_CLASS_NAMES.index("bare")
TREE = LC_CLASS_NAMES.index("tree")
URBAN = LC_CLASS_NAMES.index("urban/built-up")


def _window_output() -> np.ndarray:
    """No change except a new building patch and a forest-loss-only patch."""
    arr = np.zeros((len(OUTPUT_BANDS), PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
    arr[BINARY_CHANGE_BAND - 1] = 230
    arr[BINARY_CHANGE_BAND] = 25
    arr[SRC_BAND_OFFSET + TREE] = 200
    arr[DST_BAND_OFFSET + TREE] = 200
    arr[PRE_CHANGE_BAND_OFFSET + NONE_CATEGORY_IDX] = 250
    arr[POST_CHANGE_BAND_OFFSET + NONE_CATEGORY_IDX] = 250
    arr[TS_PRE_DAYS_BAND] = 3000
    arr[TS_POST_DAYS_BAND] = 3100

    building = arr[:, 500:600, 500:600]
    building[BINARY_CHANGE_BAND] = 230
    building[DST_BAND_OFFSET + TREE] = 0
    building[DST_BAND_OFFSET + URBAN] = 220
    building[POST_CHANGE_BAND_OFFSET + NONE_CATEGORY_IDX] = 0
    building[POST_CHANGE_BAND_OFFSET + NEW_BUILDING] = 240

    forest_loss = arr[:, 1000:1100, 1000:1100]
    forest_loss[BINARY_CHANGE_BAND] = 220
    forest_loss[DST_BAND_OFFSET + TREE] = 0
    forest_loss[DST_BAND_OFFSET + BARE] = 210
    forest_loss[PRE_CHANGE_BAND_OFFSET + NONE_CATEGORY_IDX] = 0
    forest_loss[PRE_CHANGE_BAND_OFFSET + FOREST_LOSS] = 200
    return arr


def _write_tile(tmp_path: pathlib.Path, detail: str) -> tuple[str, UPath]:
    """Write one window of a zone 10 tile to a store and its _meta.json."""
    store_path = str(tmp_path / f"lcc_{detail}.zarr")
    geozarr.init_store(store_path, detail=detail, zone_numbers=[10])
    projection = get_zone_projection(10, RESOLUTION)
    col, row = next(enumerate_zone_tiles(10, RESOLUTION, TILE_SIZE))
    tile_bounds = (
        col * TILE_SIZE,
        row * TILE_SIZE,
        col * TILE_SIZE + 2 * PATCH_SIZE,
        row * TILE_SIZE + PATCH_SIZE,
    )
    window_bounds = (
        tile_bounds[0],
        tile_bounds[1],
        tile_bounds[0] + PATCH_SIZE,
        tile_bounds[3],
    )
    arr = _window_output()
    change, dates = geozarr.geozarr_window_arrays(arr)
    geozarr.write_window(
        store_path,
        projection,
        window_bounds,
        change,
        dates,
        prob_arrays=(
            geozarr.prob_window_arrays(arr) if detail == geozarr.DETAIL_FULL else None
        ),
    )

    out_path = tmp_path / f"out_{detail}"
    out_path.mkdir()
    meta_fname = get_meta_fname(str(out_path), projection, tile_bounds)
    window_metas = [
        dict(name="a", bounds=list(window_bounds), layers={}),
        # A window without a prediction.
        dict(
            name="b",
            bounds=[window_bounds[2], tile_bounds[1], tile_bounds[2], tile_bounds[3]],
            layers={},
        ),
    ]
    with meta_fname.open("w") as f:
        json.dump(
            dict(
                projection=projection.serialize(),
                bounds=list(tile_bounds),
                time_range=[TIMESTAMP.isoformat(), TIMESTAMP.isoformat()],
                windows=window_metas,
            ),
            f,
        )
    return store_path, meta_fname


@pytest.mark.parametrize("detail", geozarr.DETAILS)
def test_phase21_reads_store(tmp_path: pathlib.Path, detail: str) -> None:
    store_path, meta_fname = _write_tile(tmp_path, detail)
    candidates = phase21.process_tile(
        str(meta_fname), store_path, min_pixels=10, seed=0
    )
    assert {cand["class_key"] for cand in candidates} == {
        "post:new_building",
        "pre:forest_loss",
    }
    for cand in candidates:
        entry = phase21.candidate_to_entry(cand, "phase21", 128)
        # Only the binary change score is recorded, the same for both store modes.
        assert not {"pre_score", "post_score"} & set(entry["pred"])
        if cand["class_key"] == "post:new_building":
            assert entry["pred"]["binary_change_score"] == 230
            point = entry["positive_points"][0]
            assert point["post_change_category"] == "new_building"
            assert point["post_category"] == "urban/built-up"
            # The predicted change start day (3000 days after 2015-01-01).
            assert point["pre_change"] == "2023-03-20"
        else:
            assert entry["pred"]["binary_change_score"] == 220
