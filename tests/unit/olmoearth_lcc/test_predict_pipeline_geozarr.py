import json
import os
import pathlib
import shutil
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr
import zarr
from jsonargparse import ArgumentParser
from pyproj import Transformer
from rslearn.dataset import Dataset
from rslearn.train.prediction_writer import PendingCropOutput, RasterMerger
from rslearn.utils.geometry import PixelBounds, Projection
from upath import UPath

from rslp.olmoearth_lcc.lcc_model import geozarr, geozarr_pyramid, predict_pipeline
from rslp.olmoearth_lcc.lcc_model import job_dir as job_dir_module
from rslp.olmoearth_lcc.lcc_model import worker as worker_module
from rslp.olmoearth_lcc.lcc_model.output_bands import (
    BINARY_CHANGE_BAND,
    DST_BAND_OFFSET,
    LC_CLASS_NAMES,
    NONE_CATEGORY_IDX,
    OUTPUT_BANDS,
    OUTPUT_LAYER,
    POST_CHANGE_BAND_OFFSET,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_BAND_OFFSET,
    SRC_BAND_OFFSET,
    TS_POST_DAYS_BAND,
    TS_PRE_DAYS_BAND,
)
from rslp.olmoearth_lcc.lcc_model.prediction_writer import LCCTileWriter
from rslp.olmoearth_lcc.lcc_model.tiling import (
    PATCH_SIZE,
    RESOLUTION,
    TILE_SIZE,
    get_zone_projection,
)

TIMESTAMP = datetime(2025, 6, 1, tzinfo=timezone.utc)
NEW_BUILDING = POST_CHANGE_CATEGORY_NAMES.index("new_building")
TREE = LC_CLASS_NAMES.index("tree")
URBAN = LC_CLASS_NAMES.index("urban/built-up")


def _fake_output() -> np.ndarray:
    """A window output with no change everywhere except a 100x100 new building."""
    arr = np.zeros((len(OUTPUT_BANDS), PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
    arr[BINARY_CHANGE_BAND - 1] = 230
    arr[BINARY_CHANGE_BAND] = 25
    arr[SRC_BAND_OFFSET + TREE] = 200
    arr[DST_BAND_OFFSET + TREE] = 200
    arr[PRE_CHANGE_BAND_OFFSET + NONE_CATEGORY_IDX] = 250
    arr[POST_CHANGE_BAND_OFFSET + NONE_CATEGORY_IDX] = 250
    arr[TS_PRE_DAYS_BAND] = 3000
    arr[TS_POST_DAYS_BAND] = 3100

    patch = (slice(None), slice(500, 600), slice(500, 600))
    change = arr[patch]
    change[BINARY_CHANGE_BAND - 1] = 25
    change[BINARY_CHANGE_BAND] = 230
    change[DST_BAND_OFFSET + TREE] = 0
    change[DST_BAND_OFFSET + URBAN] = 220
    change[POST_CHANGE_BAND_OFFSET + NONE_CATEGORY_IDX] = 0
    change[POST_CHANGE_BAND_OFFSET + NEW_BUILDING] = 240
    return arr


def _fake_materialize(ds_path: UPath, **kwargs: Any) -> None:
    for window in Dataset(ds_path).load_windows():
        window.mark_layer_completed("sentinel2_quarterly")


def _fake_trainer(ds_path: UPath) -> Any:
    """A trainer with the data module path that LCCTileWriter.setup uses."""
    return SimpleNamespace(datamodule=SimpleNamespace(path=ds_path))


def _instantiate_writer(writer_config: dict[str, Any]) -> LCCTileWriter:
    """Instantiate a writer callback config like LightningCLI does."""
    parser = ArgumentParser()
    parser.add_argument("--writer", type=LCCTileWriter)
    cfg = parser.parse_object({"writer": writer_config})
    return parser.instantiate_classes(cfg).writer


# The writers created by _fake_model_predict, so tests can inspect them.
_writers: list[LCCTileWriter] = []


def _fake_model_predict(
    model_config: str,
    ds_path: UPath,
    callbacks: list[dict[str, Any]],
    **kwargs: Any,
) -> SimpleNamespace:
    """Drive the configured writer as rslearn model predict would.

    Only the first window gets a prediction, delivered as two crops (the top and
    bottom halves, overlapping by the merger's overlap_pixels), and its first crop
    is delivered again afterwards like the all-crops dataset does when padding
    dataloader workers.
    """
    (writer_config,) = callbacks
    assert writer_config["class_path"].endswith("LCCTileWriter")
    writer = _instantiate_writer(writer_config)
    _writers.append(writer)
    writer.setup(
        trainer=_fake_trainer(ds_path),
        pl_module=None,  # type: ignore[arg-type]
    )
    window = sorted(Dataset(ds_path).load_windows(), key=lambda w: w.bounds)[0]
    arr = _fake_output()
    x0, y0, x1, y1 = window.bounds
    assert isinstance(writer.merger, RasterMerger)
    top_end = PATCH_SIZE // 2 + (writer.merger.overlap_pixels or 0) // 2
    bottom_start = PATCH_SIZE // 2 - (writer.merger.overlap_pixels or 0) // 2
    crops = [
        ((x0, y0, x1, y0 + top_end), arr[:, :top_end]),
        ((x0, y0 + bottom_start, x1, y1), arr[:, bottom_start:]),
    ]
    for crop_idx, (crop_bounds, crop) in enumerate(crops + crops[:1]):
        writer.process_output(
            window, crop_idx % len(crops), len(crops), crop_bounds, crop
        )
    writer.on_predict_end(trainer=None, pl_module=None)  # type: ignore[arg-type]
    return SimpleNamespace(trainer=SimpleNamespace(callbacks=[writer]))


def _seattle_tile(width: int, height: int, align: int) -> PixelBounds:
    """Get bounds of the given size near Seattle in zone 10, aligned to align."""
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True).transform(
        -122.3, 47.6
    )
    col = int(x // RESOLUTION) // align * align
    row = int(-y // RESOLUTION) // align * align
    return (col, row, col + width, row + height)


def _read_region(
    store_path: str, bounds: PixelBounds, group: str = "utm10", **kwargs: Any
) -> xr.Dataset:
    ds = xr.open_zarr(store_path, group=group, consolidated=False, **kwargs)
    return ds.sel(
        x=slice(bounds[0] * RESOLUTION, bounds[2] * RESOLUTION),
        y=slice(-bounds[1] * RESOLUTION, -bounds[3] * RESOLUTION),
    )


def test_predict_pipeline_geozarr(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = str(tmp_path / "lcc.zarr")
    out_path = tmp_path / "out"
    out_path.mkdir()
    geozarr.init_store(store_path, zone_numbers=[10])

    monkeypatch.setattr(predict_pipeline, "materialize_dataset", _fake_materialize)
    monkeypatch.setattr(predict_pipeline, "run_model_predict", _fake_model_predict)

    # A 4096x2048 tile (two windows).
    projection = get_zone_projection(10, RESOLUTION)
    bounds = _seattle_tile(2 * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)

    kwargs: dict[str, Any] = dict(
        projection_json=json.dumps(projection.serialize()),
        bounds=bounds,
        time_range=(TIMESTAMP, TIMESTAMP),
        out_path=str(out_path),
        zarr_store_path=store_path,
        writer_workers=2,
        fast_inference=False,
    )
    predict_pipeline.predict_pipeline(scratch_path=str(tmp_path / "scratch"), **kwargs)

    meta_fname = predict_pipeline.get_meta_fname(str(out_path), projection, bounds)
    assert meta_fname.exists()
    with meta_fname.open() as f:
        timings = json.load(f)["timings"]
    assert set(timings) == {
        "create_windows",
        "materialize",
        "predict",
        "write_wait",
        "total",
    }
    # The window was written once, and its repeated crop was ignored.
    writer = _writers[-1]
    assert len(writer.results) == 1
    assert not writer.pending_outputs
    # The _meta.json is the only output file.
    assert [p.name for p in out_path.iterdir()] == [meta_fname.name]

    change = _read_region(store_path, bounds)["change"]
    assert change.shape == (7, PATCH_SIZE, 2 * PATCH_SIZE)
    first, second = change[:, :, :PATCH_SIZE].values, change[:, :, PATCH_SIZE:].values
    # The window without a prediction is nodata.
    assert not second.any()
    # The change patch keeps its details; the rest has them zeroed.
    np.testing.assert_array_equal(
        first[:, 550, 550],
        [230, TREE, URBAN, NONE_CATEGORY_IDX, NEW_BUILDING, 250, 240],
    )
    np.testing.assert_array_equal(
        first[:, 0, 0], [25, 0, 0, NONE_CATEGORY_IDX, NONE_CATEGORY_IDX, 0, 0]
    )
    dates = _read_region(store_path, bounds, decode_times=False)["change_dates"]
    np.testing.assert_array_equal(
        dates[:, 550, 550].values, [3000 + 16436, 3100 + 16436]
    )
    assert not np.nan_to_num(dates[:, 0, 0].values).any()
    # The units attribute lets xarray decode the dates (3000 days after 2015-01-01).
    decoded = _read_region(store_path, bounds)["change_dates"]
    assert decoded[0, 550, 550].values == np.datetime64("2023-03-20")

    # Re-running skips the tile since its marker exists.
    monkeypatch.setattr(
        predict_pipeline,
        "materialize_dataset",
        lambda *args, **kwargs: pytest.fail("should not re-run"),
    )
    predict_pipeline.predict_pipeline(scratch_path=str(tmp_path / "scratch2"), **kwargs)


def test_predict_pipeline_full_store_with_pyramid(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = str(tmp_path / "lcc.zarr")
    pyramid_path = str(tmp_path / "lcc_pyramid.zarr")
    out_path = tmp_path / "out"
    out_path.mkdir()
    geozarr.init_store(store_path, detail=geozarr.DETAIL_FULL, zone_numbers=[10])
    geozarr_pyramid.init_pyramid_store(pyramid_path, store_path, zone_numbers=[10])

    monkeypatch.setattr(predict_pipeline, "materialize_dataset", _fake_materialize)
    monkeypatch.setattr(predict_pipeline, "run_model_predict", _fake_model_predict)
    # Only create two windows of the tile so the test is fast.
    keep = {"first", "second"}

    def fake_has_land(projection: Projection, bounds: PixelBounds) -> bool:
        if not keep:
            return False
        keep.pop()
        return True

    monkeypatch.setattr(predict_pipeline, "window_has_land", fake_has_land)

    projection = get_zone_projection(10, RESOLUTION)
    bounds = _seattle_tile(TILE_SIZE, TILE_SIZE, TILE_SIZE)
    predict_pipeline.predict_pipeline(
        projection_json=json.dumps(projection.serialize()),
        bounds=bounds,
        time_range=(TIMESTAMP, TIMESTAMP),
        out_path=str(out_path),
        scratch_path=str(tmp_path / "scratch"),
        zarr_store_path=store_path,
        zarr_pyramid_path=pyramid_path,
        fast_inference=False,
    )
    with predict_pipeline.get_meta_fname(str(out_path), projection, bounds).open() as f:
        meta = json.load(f)
    assert "write_pyramid" in meta["timings"]
    assert len(meta["windows"]) == 2

    # Full stores hold the per-head probabilities.
    window_bounds = (
        bounds[0],
        bounds[1],
        bounds[0] + PATCH_SIZE,
        bounds[1] + PATCH_SIZE,
    )
    region = _read_region(store_path, window_bounds)
    post_probs = region["post_change_probs"]
    assert post_probs.shape == (
        len(POST_CHANGE_CATEGORY_NAMES) - 1,
        PATCH_SIZE,
        PATCH_SIZE,
    )
    assert post_probs.sel(post_change_probs_band="new_building")[550, 550] == 240
    assert post_probs.sel(post_change_probs_band="none")[0, 0] == 250
    assert region["src_probs"].sel(src_probs_band="tree")[0, 0] == 200

    # The change pixel survives down to the coarsest level, and the rest of the
    # first window stays valid (category "none") rather than nodata.
    for level in geozarr_pyramid.PYRAMID_LEVELS:
        scale = 2**level
        size = PATCH_SIZE // scale
        level_bounds = (
            window_bounds[0] // scale,
            window_bounds[1] // scale,
            window_bounds[0] // scale + size,
            window_bounds[1] // scale + size,
        )
        group = zarr.open_group(pyramid_path, path=f"utm10/{level}", mode="r")
        row, col = geozarr.get_group_offset(group, level_bounds)
        level_ds = xr.open_zarr(
            pyramid_path, group=f"utm10/{level}", consolidated=False
        )
        assert level_ds["x"].values[col] == (level_bounds[0] + 0.5) * RESOLUTION * scale
        change_array = group["change"]
        assert isinstance(change_array, zarr.Array)
        block = np.asarray(change_array[:, row : row + size, col : col + size])
        assert (block[4] == NEW_BUILDING).sum() >= 1
        assert (block[3] == NONE_CATEGORY_IDX).all()
        np.testing.assert_array_equal(
            block[:, 550 // scale, 550 // scale],
            [230, TREE, URBAN, NONE_CATEGORY_IDX, NEW_BUILDING, 250, 240],
        )


def test_worker_sequential(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = str(tmp_path / "lcc.zarr")
    out_path = tmp_path / "out"
    out_path.mkdir()
    job_dir = tmp_path / "jobs"
    scratch_path = tmp_path / "scratch"
    geozarr.init_store(store_path, zone_numbers=[10])
    monkeypatch.setattr(predict_pipeline, "materialize_dataset", _fake_materialize)
    monkeypatch.setattr(predict_pipeline, "run_model_predict", _fake_model_predict)

    projection = get_zone_projection(10, RESOLUTION)
    first = _seattle_tile(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    second = (first[0] + PATCH_SIZE, first[1], first[2] + PATCH_SIZE, first[3])
    failing = (first[0] + 2 * PATCH_SIZE, first[1], first[2] + 2 * PATCH_SIZE, first[3])
    job_dir_module.write_job_dir(
        str(job_dir),
        job_dir_module.JobConfig(
            out_path=str(out_path), zarr_store_path=store_path, fast_inference=False
        ),
        [
            job_dir_module.PredictTaskArgs(projection, bounds, (TIMESTAMP, TIMESTAMP))
            for bounds in [first, second, failing]
        ],
    )

    # Prediction of one tile fails; the others still finish.
    real_predict_tile = worker_module.predict_tile

    def flaky_predict_tile(
        prepared: predict_pipeline.PreparedTile, **kwargs: Any
    ) -> None:
        if tuple(prepared.bounds) == failing:
            raise RuntimeError("model failed")
        real_predict_tile(prepared, **kwargs)

    monkeypatch.setattr(worker_module, "predict_tile", flaky_predict_tile)

    worker_module.worker(
        str(job_dir), scratch_path=str(scratch_path), background_prepare=False
    )
    for bounds in [first, second]:
        meta_fname = predict_pipeline.get_meta_fname(str(out_path), projection, bounds)
        with meta_fname.open() as f:
            assert "materialize_wait" in json.load(f)["timings"]
    assert not predict_pipeline.get_meta_fname(
        str(out_path), projection, failing
    ).exists()
    # Finished tiles are removed from the job directory; the failed one stays, with
    # a failure marker and no claim.
    failing_tile = job_dir_module.tile_name(projection, failing)
    assert os.listdir(job_dir / "tasks") == [failing_tile + ".json"]
    assert not os.listdir(job_dir / "claims")
    (marker,) = os.listdir(job_dir / "failures")
    assert marker.startswith(failing_tile + ".")
    assert "model failed" in (job_dir / "failures" / marker).read_text()
    # Each tile's scratch directory is removed once it is done.
    assert not list(scratch_path.iterdir())


def test_worker_terminates_preparation_of_failed_tile(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "out"
    out_path.mkdir()
    job_dir = tmp_path / "jobs"
    projection = get_zone_projection(10, RESOLUTION)
    # A pyramid store requires tiles aligned to TILE_SIZE, so check_tile fails after
    # the tile's preparation was started.
    unaligned = _seattle_tile(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)
    job_dir_module.write_job_dir(
        str(job_dir),
        job_dir_module.JobConfig(
            out_path=str(out_path),
            zarr_store_path=str(tmp_path / "lcc.zarr"),
            zarr_pyramid_path=str(tmp_path / "lcc_pyramid.zarr"),
            fast_inference=False,
        ),
        [job_dir_module.PredictTaskArgs(projection, unaligned, (TIMESTAMP, TIMESTAMP))],
    )

    terminated: list[bool] = []

    class FakeBackgroundPrepare:
        def __init__(self, kwargs: dict[str, Any]) -> None:
            pass

        def result(self) -> Any:
            pytest.fail("the failed tile should not be predicted")

        def terminate(self) -> None:
            terminated.append(True)

    monkeypatch.setattr(worker_module, "BackgroundPrepare", FakeBackgroundPrepare)
    worker_module.worker(str(job_dir), scratch_path=str(tmp_path / "scratch"))
    assert terminated
    assert len(os.listdir(job_dir / "failures")) == 1


def test_background_prepare_error(tmp_path: pathlib.Path) -> None:
    # prepare_tile fails since the scratch dataset directory already exists.
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    projection = get_zone_projection(10, RESOLUTION)
    background = predict_pipeline.BackgroundPrepare(
        dict(
            projection=projection,
            bounds=(0, 0, PATCH_SIZE, PATCH_SIZE),
            time_range=(TIMESTAMP, TIMESTAMP),
            scratch_path=str(scratch),
        )
    )
    with pytest.raises(RuntimeError, match="FileExistsError"):
        background.result()


def test_writer_stops_threads_on_exception(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "lcc.zarr")
    geozarr.init_store(store_path, zone_numbers=[10])
    ds_path = tmp_path / "ds"
    ds_path.mkdir()
    shutil.copyfile(predict_pipeline.DATASET_CONFIG_FNAME, ds_path / "config.json")
    writer = LCCTileWriter(
        output_layer=OUTPUT_LAYER, zarr_store_path=store_path, workers=1
    )
    writer.setup(
        trainer=_fake_trainer(UPath(ds_path)),
        pl_module=None,  # type: ignore[arg-type]
    )
    executor = writer._executor
    assert executor is not None

    # One running job blocks the single thread, so the others stay queued.
    unblock = threading.Event()
    ran: list[int] = []
    executor.submit(unblock.wait)
    queued = [executor.submit(ran.append, i) for i in range(3)]
    threading.Timer(0.2, unblock.set).start()

    writer.on_exception(trainer=None, pl_module=None, exception=RuntimeError())  # type: ignore[arg-type]
    assert writer._executor is None
    assert all(future.cancelled() for future in queued)
    assert ran == []
    assert not any(t.name.startswith("lcc_writer") for t in threading.enumerate())


def test_writer_trims_crop_overlap(tmp_path: pathlib.Path) -> None:
    store_path = str(tmp_path / "lcc.zarr")
    geozarr.init_store(store_path, zone_numbers=[10])
    ds_path = tmp_path / "ds"
    ds_path.mkdir()
    shutil.copyfile(predict_pipeline.DATASET_CONFIG_FNAME, ds_path / "config.json")
    # The writer gets the merger of the model config's RslearnWriter.
    writer = _instantiate_writer(
        predict_pipeline.get_writer_config(store_path, pyramid=False, writer_workers=1)
    )
    assert isinstance(writer.merger, RasterMerger)
    assert writer.merger.overlap_pixels == 16
    writer.setup(
        trainer=_fake_trainer(UPath(ds_path)),
        pl_module=None,  # type: ignore[arg-type]
    )
    writer.on_predict_end(trainer=None, pl_module=None)  # type: ignore[arg-type]

    # Two crops sharing 16 rows: each keeps the 8 shared rows nearer its interior.
    half = PATCH_SIZE // 2
    window = SimpleNamespace(bounds=(0, 0, PATCH_SIZE, PATCH_SIZE))
    top = np.full((1, half + 8, PATCH_SIZE), 1, dtype=np.uint16)
    bottom = np.full((1, half + 8, PATCH_SIZE), 2, dtype=np.uint16)
    assert writer.merger is not None and writer.layer_config is not None
    merged = writer.merger.merge(
        window,  # type: ignore[arg-type]
        [
            PendingCropOutput((0, 0, PATCH_SIZE, half + 8), top),
            PendingCropOutput((0, half - 8, PATCH_SIZE, PATCH_SIZE), bottom),
        ],
        writer.layer_config,
    )
    assert (merged[0, :half] == 1).all()
    assert (merged[0, half:] == 2).all()


def test_check_tile_rejects_unaligned_pyramid_tile() -> None:
    projection = get_zone_projection(10, RESOLUTION)
    with pytest.raises(ValueError):
        predict_pipeline.check_tile(
            projection, (0, 0, PATCH_SIZE, PATCH_SIZE), zarr_pyramid_path="pyr.zarr"
        )
    predict_pipeline.check_tile(projection, (0, 0, PATCH_SIZE, PATCH_SIZE), None)
