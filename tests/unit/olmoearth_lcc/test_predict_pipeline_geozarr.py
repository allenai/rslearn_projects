import json
import pathlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pytest
import xarray as xr
from pyproj import Transformer
from rslearn.dataset import Dataset
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from upath import UPath

from rslp.olmoearth_lcc.lcc_model import geozarr, predict_pipeline
from rslp.olmoearth_lcc.lcc_model.postprocess import (
    BINARY_CHANGE_BAND,
    DST_BAND_OFFSET,
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
from rslp.olmoearth_lcc.lcc_model.tiling import (
    PATCH_SIZE,
    RESOLUTION,
    get_zone_projection,
)

TIMESTAMP = datetime(2025, 6, 1, tzinfo=timezone.utc)
NEW_BUILDING = POST_CHANGE_CATEGORY_NAMES.index("new_building")


def _fake_output() -> np.ndarray:
    """A window output with no change everywhere except a 100x100 new building."""
    arr = np.zeros((len(OUTPUT_BANDS), PATCH_SIZE, PATCH_SIZE), dtype=np.uint16)
    arr[BINARY_CHANGE_BAND - 1] = 230
    arr[BINARY_CHANGE_BAND] = 25
    arr[SRC_BAND_OFFSET + 9] = 200
    arr[DST_BAND_OFFSET + 9] = 200
    arr[PRE_CHANGE_BAND_OFFSET + NONE_CATEGORY_IDX] = 250
    arr[POST_CHANGE_BAND_OFFSET + NONE_CATEGORY_IDX] = 250
    arr[TS_PRE_DAYS_BAND] = 3000
    arr[TS_POST_DAYS_BAND] = 3100

    patch = (slice(None), slice(500, 600), slice(500, 600))
    change = arr[patch]
    change[BINARY_CHANGE_BAND - 1] = 25
    change[BINARY_CHANGE_BAND] = 230
    change[DST_BAND_OFFSET + 9] = 0
    change[DST_BAND_OFFSET + 10] = 220
    change[POST_CHANGE_BAND_OFFSET + NONE_CATEGORY_IDX] = 0
    change[POST_CHANGE_BAND_OFFSET + NEW_BUILDING] = 240
    return arr


def test_predict_pipeline_geozarr(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = str(tmp_path / "lcc.zarr")
    out_path = tmp_path / "out"
    out_path.mkdir()
    geozarr.init_store(store_path, TIMESTAMP, zone_numbers=[10])

    def fake_materialize(ds_path: UPath, **kwargs: Any) -> None:
        for window in Dataset(ds_path).load_windows():
            window.mark_layer_completed("sentinel2_quarterly")

    def fake_model_predict(model_config: str, ds_path: UPath, **kwargs: Any) -> None:
        # Only the first window gets a prediction.
        window = sorted(Dataset(ds_path).load_windows(), key=lambda w: w.bounds)[0]
        GeotiffRasterFormat().encode_raster(
            window.get_raster_dir(OUTPUT_LAYER, OUTPUT_BANDS),
            window.projection,
            window.bounds,
            RasterArray(chw_array=_fake_output()),
        )

    monkeypatch.setattr(predict_pipeline, "materialize_dataset", fake_materialize)
    monkeypatch.setattr(predict_pipeline, "run_model_predict", fake_model_predict)

    # A 4096x2048 tile (two windows) near Seattle in zone 10.
    projection = get_zone_projection(10, RESOLUTION)
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True).transform(
        -122.3, 47.6
    )
    col = int(x // RESOLUTION) // PATCH_SIZE * PATCH_SIZE
    row = int(-y // RESOLUTION) // PATCH_SIZE * PATCH_SIZE
    bounds = (col, row, col + 2 * PATCH_SIZE, row + PATCH_SIZE)

    kwargs: dict[str, Any] = dict(
        projection_json=json.dumps(projection.serialize()),
        bounds=bounds,
        time_range=(TIMESTAMP, TIMESTAMP),
        out_path=str(out_path),
        workers=0,
        zarr_store_path=store_path,
        geozarr_workers=0,
        fast_inference=False,
    )
    predict_pipeline.predict_pipeline(scratch_path=str(tmp_path / "scratch"), **kwargs)

    meta_fname = predict_pipeline.get_output_fname(
        str(out_path), projection, bounds, meta=True
    )
    assert meta_fname.exists()
    with meta_fname.open() as f:
        timings = json.load(f)["timings"]
    assert set(timings) == {
        "create_windows",
        "materialize",
        "predict",
        "write_geozarr",
        "write_geojson",
        "total",
    }
    with predict_pipeline.get_output_fname(
        str(out_path), projection, bounds
    ).open() as f:
        features = json.load(f)["features"]
    assert len(features) == 1
    assert features[0]["properties"]["post_change_category"] == "new_building"

    ds = xr.open_zarr(store_path, group="utm10", consolidated=False)
    change = ds["change"].sel(
        x=slice(bounds[0] * RESOLUTION, bounds[2] * RESOLUTION),
        y=slice(-bounds[1] * RESOLUTION, -bounds[3] * RESOLUTION),
    )
    assert change.shape == (6, PATCH_SIZE, 2 * PATCH_SIZE)
    first, second = change[:, :, :PATCH_SIZE].values, change[:, :, PATCH_SIZE:].values
    # The window without a prediction is nodata.
    assert not second.any()
    # The change patch keeps its details; the rest has them zeroed.
    np.testing.assert_array_equal(
        first[:, 550, 550], [230, 9, 10, NONE_CATEGORY_IDX, NEW_BUILDING, 240]
    )
    np.testing.assert_array_equal(
        first[:, 0, 0], [25, 0, 0, NONE_CATEGORY_IDX, NONE_CATEGORY_IDX, 0]
    )
    raw_ds = xr.open_zarr(
        store_path, group="utm10", consolidated=False, decode_times=False
    )
    dates = raw_ds["change_dates"].sel(
        x=slice(bounds[0] * RESOLUTION, bounds[2] * RESOLUTION),
        y=slice(-bounds[1] * RESOLUTION, -bounds[3] * RESOLUTION),
    )
    np.testing.assert_array_equal(
        dates[:, 550, 550].values, [3000 + 16436, 3100 + 16436]
    )
    assert not np.nan_to_num(dates[:, 0, 0].values).any()
    # The units attribute lets xarray decode the dates (3000 days after 2015-01-01).
    decoded = ds["change_dates"].sel(
        x=slice(bounds[0] * RESOLUTION, bounds[2] * RESOLUTION),
        y=slice(-bounds[1] * RESOLUTION, -bounds[3] * RESOLUTION),
    )
    assert decoded[0, 550, 550].values == np.datetime64("2023-03-20")

    # Re-running skips the tile since its marker exists.
    monkeypatch.setattr(
        predict_pipeline,
        "materialize_dataset",
        lambda *args, **kwargs: pytest.fail("should not re-run"),
    )
    predict_pipeline.predict_pipeline(scratch_path=str(tmp_path / "scratch2"), **kwargs)
