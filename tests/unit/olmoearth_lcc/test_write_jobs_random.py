import os
import pathlib
import random
from datetime import datetime, timezone

import numpy as np
import pytest
import shapely

from rslp.olmoearth_lcc.lcc_model import geozarr
from rslp.olmoearth_lcc.lcc_model.job_dir import tile_name
from rslp.olmoearth_lcc.lcc_model.predict_pipeline import get_meta_fname
from rslp.olmoearth_lcc.lcc_model.tiling import PATCH_SIZE
from rslp.olmoearth_lcc.lcc_model.write_jobs import write_tasks
from rslp.olmoearth_lcc.lcc_model.write_jobs_random_2048 import (
    CHINA_POLYGON,
    RandomTileCollector,
    sample_region_point,
    snap_random_tile,
)

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_southern_tile_fits_store(tmp_path: pathlib.Path) -> None:
    # A point in Nairobi (zone 37, south of the equator) snaps in EPSG:32637.
    tile = snap_random_tile(36.82, -1.29)
    assert tile is not None
    projection, bounds = tile
    assert projection.crs.to_epsg() == 32637
    assert all(value % PATCH_SIZE == 0 for value in bounds)
    assert bounds[1] > 0

    store_path = str(tmp_path / "lcc.zarr")
    geozarr.init_store(store_path, zone_numbers=[37])
    change = np.ones((len(geozarr.CHANGE_BANDS), PATCH_SIZE, PATCH_SIZE), np.uint8)
    dates = np.ones((2, PATCH_SIZE, PATCH_SIZE), np.uint16)
    geozarr.write_window(store_path, projection, bounds, change, dates)
    arrays = geozarr.read_window(store_path, projection, bounds)
    assert arrays[geozarr.CHANGE_ARRAY].all()


def test_random_tile_collector_dedup() -> None:
    tile = snap_random_tile(-122.3, 47.6)
    other = snap_random_tile(-122.0, 47.6)
    assert tile is not None and other is not None

    collector = RandomTileCollector(START, END, random.Random(0))
    assert collector.add(*tile)
    assert not collector.add(*tile)
    assert collector.add(*other)
    assert len(collector.tasks) == 2
    for task in collector.tasks:
        t0, t1 = task.time_range
        assert t0 == t1 and START <= t0 <= END


def test_sample_region_point() -> None:
    rng = random.Random(0)
    for _ in range(20):
        lon, lat = sample_region_point(rng, "china")
        assert CHINA_POLYGON.contains(shapely.Point(lon, lat))
        lon, lat = sample_region_point(rng, "global")
        assert -180 <= lon <= 180 and -60 <= lat <= 70
    with pytest.raises(KeyError):
        sample_region_point(rng, "mars")


def test_write_tasks_skips_queued_and_done(tmp_path: pathlib.Path) -> None:
    out_path = tmp_path / "out"
    out_path.mkdir()
    job_dir = str(tmp_path / "jobs")
    store_path = str(tmp_path / "lcc.zarr")
    tiles = [snap_random_tile(lon, 47.6) for lon in [-122.3, -122.0, -121.5]]
    assert all(tile is not None for tile in tiles)
    collector = RandomTileCollector(START, END, random.Random(0))
    for tile in tiles[:2]:
        assert tile is not None
        collector.add(*tile)
    write_tasks(job_dir, collector.tasks, str(out_path), store_path)
    assert geozarr.read_store_detail(store_path) is not None
    assert len(os.listdir(os.path.join(job_dir, "tasks"))) == 2

    # A rerun still collects the queued and done tiles, but the job directory skips
    # them: only the new tile gets a task.
    done = tiles[1]
    assert done is not None
    os.remove(os.path.join(job_dir, "tasks", tile_name(*done) + ".json"))
    get_meta_fname(str(out_path), *done).touch()
    collector = RandomTileCollector(START, END, random.Random(1))
    for tile in tiles:
        assert tile is not None
        assert collector.add(*tile)
    write_tasks(job_dir, collector.tasks, str(out_path), store_path)
    assert sorted(os.listdir(os.path.join(job_dir, "tasks"))) == sorted(
        tile_name(*tile) + ".json" for tile in [tiles[0], tiles[2]] if tile is not None
    )
