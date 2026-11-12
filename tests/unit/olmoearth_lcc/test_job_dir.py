import json
import multiprocessing
import os
import pathlib
import time
from datetime import datetime, timezone

import pytest

from rslp.olmoearth_lcc.lcc_model import job_dir
from rslp.olmoearth_lcc.lcc_model.job_dir import PredictTaskArgs
from rslp.olmoearth_lcc.lcc_model.predict_pipeline import get_meta_fname
from rslp.olmoearth_lcc.lcc_model.tiling import (
    PATCH_SIZE,
    RESOLUTION,
    get_zone_projection,
)

TIMESTAMP = datetime(2025, 6, 1, tzinfo=timezone.utc)
PROJECTION = get_zone_projection(10, RESOLUTION)


def _tasks(num_tasks: int) -> list[PredictTaskArgs]:
    return [
        PredictTaskArgs(
            PROJECTION,
            (i * PATCH_SIZE, 0, (i + 1) * PATCH_SIZE, PATCH_SIZE),
            (TIMESTAMP, TIMESTAMP),
        )
        for i in range(num_tasks)
    ]


def _make_job_dir(
    tmp_path: pathlib.Path, num_tasks: int
) -> tuple[str, job_dir.JobConfig]:
    out_path = tmp_path / "out"
    out_path.mkdir(parents=True)
    config = job_dir.JobConfig(out_path=str(out_path), zarr_store_path="lcc.zarr")
    path = str(tmp_path / "jobs")
    assert job_dir.write_job_dir(path, config, _tasks(num_tasks)) == num_tasks
    return path, config


def _make_stale(claim: job_dir.Claim) -> None:
    old = time.time() - 2 * job_dir.DEFAULT_STALE_SECONDS
    os.utime(claim.path, (old, old))


def test_write_job_dir(tmp_path: pathlib.Path) -> None:
    path, config = _make_job_dir(tmp_path, 3)
    assert job_dir.read_job_config(path) == config
    tiles = sorted(os.listdir(os.path.join(path, "tasks")))
    assert tiles == sorted(
        job_dir.tile_name(PROJECTION, task.bounds) + ".json" for task in _tasks(3)
    )
    # The task files hold the serialized tasks.
    with open(os.path.join(path, "tasks", tiles[0])) as f:
        task = PredictTaskArgs.deserialize(json.load(f))
    assert task.time_range == (TIMESTAMP, TIMESTAMP)

    # Re-running only adds new tiles, and skips tiles that are done.
    tasks = _tasks(5)
    get_meta_fname(config.out_path, PROJECTION, tasks[4].bounds).touch()
    assert job_dir.write_job_dir(path, config, tasks) == 1
    assert len(os.listdir(os.path.join(path, "tasks"))) == 4

    # The config must match.
    other = job_dir.JobConfig(out_path=config.out_path, zarr_store_path="other.zarr")
    with pytest.raises(ValueError):
        job_dir.write_job_dir(path, other, tasks)
    with pytest.raises(ValueError):
        job_dir.write_job_dir("gs://bucket/jobs", config, tasks)


def _claim_all(path: str, out_path: str, worker_id: str) -> list[str]:
    tiles: list[str] = []
    while True:
        claim = job_dir.claim_next(path, out_path, worker_id)
        if claim is None:
            return tiles
        tiles.append(claim.tile)


def test_claim_race(tmp_path: pathlib.Path) -> None:
    path, config = _make_job_dir(tmp_path, 40)
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(8) as pool:
        results = pool.starmap(
            _claim_all, [(path, config.out_path, f"worker{i}") for i in range(8)]
        )
    claimed = [tile for tiles in results for tile in tiles]
    # Every tile is claimed exactly once.
    assert len(claimed) == 40
    assert len(set(claimed)) == 40


def test_stale_claims(tmp_path: pathlib.Path) -> None:
    path, config = _make_job_dir(tmp_path, 1)
    claim = job_dir.claim_next(path, config.out_path, "a")
    assert claim is not None and claim.path.endswith(".claim.1")
    # A fresh claim can't be taken over.
    assert job_dir.claim_next(path, config.out_path, "b") is None

    # Refreshing keeps it fresh.
    _make_stale(claim)
    refresher = job_dir.ClaimRefresher()
    refresher.add(claim)
    refresher.refresh()
    assert job_dir.claim_next(path, config.out_path, "b") is None

    # A stale claim is taken over with a new generation.
    _make_stale(claim)
    counts = job_dir.job_status(path)
    assert counts["stale"] == 1 and counts["remaining"] == 1
    taken = job_dir.claim_next(path, config.out_path, "b")
    assert taken is not None and taken.tile == claim.tile
    assert taken.path.endswith(".claim.2")
    assert job_dir.job_status(path)["claimed"] == 1


def test_finish_and_release(tmp_path: pathlib.Path) -> None:
    path, config = _make_job_dir(tmp_path, 2)
    claim = job_dir.claim_next(path, config.out_path, "a")
    assert claim is not None
    job_dir.finish(path, claim)
    assert len(os.listdir(os.path.join(path, "tasks"))) == 1
    assert not os.listdir(os.path.join(path, "claims"))

    # Released claims can be claimed again until max_failures attempts failed.
    for attempt in range(job_dir.DEFAULT_MAX_FAILURES):
        claim = job_dir.claim_next(path, config.out_path, "a")
        assert claim is not None
        job_dir.release(path, claim, failed=True, message=f"attempt {attempt}")
        assert not os.path.exists(claim.path)
    assert len(os.listdir(os.path.join(path, "failures"))) == 3
    assert job_dir.claim_next(path, config.out_path, "a") is None
    assert job_dir.job_status(path)["failed"] == 1

    # Releasing without failure (e.g. preemption) writes no marker.
    path2, config2 = _make_job_dir(tmp_path / "second", 1)
    claim = job_dir.claim_next(path2, config2.out_path, "a")
    assert claim is not None
    job_dir.release(path2, claim, failed=False)
    assert not os.listdir(os.path.join(path2, "failures"))
    assert job_dir.job_status(path2)["unclaimed"] == 1


def test_claim_skips_done_tiles(tmp_path: pathlib.Path) -> None:
    path, config = _make_job_dir(tmp_path, 1)
    task = _tasks(1)[0]
    get_meta_fname(config.out_path, PROJECTION, task.bounds).touch()
    # The done tile is cleaned up instead of claimed.
    assert job_dir.claim_next(path, config.out_path, "a") is None
    assert not os.listdir(os.path.join(path, "tasks"))
    assert not os.listdir(os.path.join(path, "claims"))
    assert job_dir.job_status(path)["done"] == 1
