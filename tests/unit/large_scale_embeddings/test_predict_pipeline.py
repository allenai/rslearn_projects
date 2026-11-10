"""Unit tests for rslp.large_scale_embeddings.predict_pipeline."""

import json
import pathlib
import sys
from datetime import UTC, datetime

import numpy as np
import pytest
from rasterio.crs import CRS
from rslearn.dataset import Dataset
from rslearn.dataset.window import WindowLayerData
from rslearn.utils.geometry import Projection
from upath import UPath

from rslp.large_scale_embeddings.model import NODATA_VALUE
from rslp.large_scale_embeddings.predict_pipeline import (
    AllZeroEmbeddingError,
    _collect_provenance,
    _mask_invalid_pixels,
    get_provenance_fname,
)

MARKER_NAME = "EPSG:32628_45056_-720896.json"


def test_get_provenance_fname_is_marker_sibling() -> None:
    """Provenance lands beside the marker directory, under the same basename."""
    marker_fname = UPath(f"gs://bucket/archive/completed_2025/{MARKER_NAME}")
    assert get_provenance_fname(marker_fname) == UPath(
        f"gs://bucket/archive/provenance_2025/{MARKER_NAME}"
    )


def test_get_provenance_fname_unrecognized_directory() -> None:
    """A marker directory not named completed_* still gets a distinct sibling."""
    marker_fname = UPath(f"gs://bucket/archive/markers/{MARKER_NAME}")
    assert get_provenance_fname(marker_fname) == UPath(
        f"gs://bucket/archive/markers_provenance/{MARKER_NAME}"
    )


class FakeWindow:
    """A window that reports fixed layer datas."""

    def __init__(self, name: str, layer_datas: dict[str, WindowLayerData]) -> None:
        """Initialize a new FakeWindow."""
        self.name = name
        self._layer_datas = layer_datas

    def load_layer_datas(self) -> dict[str, WindowLayerData]:
        """Return the fixed layer datas."""
        return self._layer_datas


def test_collect_provenance_records_items_and_periods() -> None:
    """Each mosaic's source items and requested period are recorded per layer."""
    period = (datetime(2025, 1, 6, tzinfo=UTC), datetime(2025, 2, 5, tzinfo=UTC))
    window = FakeWindow(
        "22_-176",
        {
            "sentinel2_l2a": WindowLayerData(
                layer_name="sentinel2_l2a",
                serialized_item_groups=[[{"name": "S2A_scene", "cloud_cover": 1.5}]],
                group_time_ranges=[period],
            ),
            # The output layer is this pipeline's own product, not a source.
            "output": WindowLayerData(
                layer_name="output",
                serialized_item_groups=[[{"name": "irrelevant"}]],
            ),
        },
    )

    provenance = _collect_provenance([window])

    assert set(provenance) == {"22_-176"}
    assert set(provenance["22_-176"]) == {"sentinel2_l2a"}
    assert provenance["22_-176"]["sentinel2_l2a"] == [
        {
            "time_range": ["2025-01-06T00:00:00+00:00", "2025-02-05T00:00:00+00:00"],
            "items": [{"name": "S2A_scene", "cloud_cover": 1.5}],
        }
    ]


def test_collect_provenance_without_group_time_ranges() -> None:
    """Layers prepared without per-group periods still record their items."""
    window = FakeWindow(
        "0_0",
        {
            "landsat": WindowLayerData(
                layer_name="landsat",
                serialized_item_groups=[[{"name": "LC09_scene"}]],
            )
        },
    )

    provenance = _collect_provenance([window])

    assert provenance["0_0"]["landsat"] == [
        {"time_range": None, "items": [{"name": "LC09_scene"}]}
    ]


def test_collect_provenance_window_without_items(tmp_path: pathlib.Path) -> None:
    """A window whose prepare found nothing contributes an empty entry, not an error."""
    assert _collect_provenance([FakeWindow("1_1", {})]) == {"1_1": {}}


def test_a_released_bundle_is_loaded_as_model_path(tmp_path: pathlib.Path) -> None:
    """A config.json + weights.pth bundle is passed as model_path."""
    from rslp.large_scale_embeddings.predict_pipeline import _checkpoint_arg

    bundle = tmp_path / "v1_3_release_v2"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    (bundle / "weights.pth").write_bytes(b"")
    assert _checkpoint_arg(str(bundle)) == "model_path"


def test_a_training_checkpoint_is_loaded_as_checkpoint_path(
    tmp_path: pathlib.Path,
) -> None:
    """A training checkpoint folder is passed as checkpoint_path."""
    from rslp.large_scale_embeddings.predict_pipeline import _checkpoint_arg

    ckpt = tmp_path / "step667200"
    (ckpt / "model_and_optim").mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    assert _checkpoint_arg(str(ckpt)) == "checkpoint_path"


def test_only_one_loader_argument_survives(tmp_path: pathlib.Path) -> None:
    """Exactly one loader argument reaches the encoder, with Searchlight always on."""
    import json as _json

    import yaml as _yaml

    from rslp.large_scale_embeddings.predict_pipeline import _get_model_extra_args

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    (bundle / "weights.pth").write_bytes(b"")

    config = {
        "model": {
            "init_args": {
                "model": {
                    "init_args": {
                        "encoder": [
                            {
                                "class_path": "rslearn.models.olmoearth_pretrain.model.OlmoEarth",
                                "init_args": {
                                    "projected_location_token_dim": 128,
                                    "checkpoint_path": "placeholder",
                                },
                            }
                        ]
                    }
                }
            }
        },
        "trainer": {
            "callbacks": [
                {"init_args": {"merger": {"init_args": {}}}},
            ]
        },
    }
    config_fname = tmp_path / "model.yaml"
    with config_fname.open("w") as f:
        _yaml.safe_dump(config, f)

    args = _get_model_extra_args(
        model_config_fname=str(config_fname),
        checkpoint_path=str(bundle),
        patch_size=2,
        location_query_patch_size=1,
        window_size=480,
        overlap_size=32,
    )
    encoder = _json.loads(
        args[args.index("--model.init_args.model.init_args.encoder") + 1]
    )
    init_args = encoder[0]["init_args"]
    present = [
        k for k in ("model_id", "model_path", "checkpoint_path") if k in init_args
    ]
    assert present == ["model_path"], f"expected only model_path, got {present}"
    assert init_args["model_path"] == str(bundle)
    assert init_args["projected_location_token_dim"] == 128
    assert init_args["compile_model"] is True
    forward_kwargs = init_args["forward_kwargs"]
    assert forward_kwargs["location_query_patch_size"] == 1
    assert forward_kwargs["searchlight"]["init_args"] == {"compile": True}


def test_predict_writes_a_marker_when_no_crop_has_output(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crop without model output is recorded as skipped, and the block still marked."""
    pp = sys.modules["rslp.large_scale_embeddings.predict_pipeline"]

    calls: list[str] = []
    time_ranges: list[tuple[datetime, datetime]] = []

    def fake_materialize(ds_path: UPath, materialize_pipeline_args: object) -> None:
        calls.append("materialize")
        for window in Dataset(ds_path).load_windows(groups=[pp.PREDICTION_GROUP]):
            time_ranges.append(window.time_range)
            window.mark_layer_completed(pp.SENTINEL2_LAYER)

    monkeypatch.setattr(pp, "materialize_dataset", fake_materialize)
    monkeypatch.setattr(
        pp, "run_model_predict", lambda *a, **k: calls.append("predict")
    )
    monkeypatch.setattr(pp, "get_zone_wedge", lambda crs, res: None)
    monkeypatch.setattr(
        pp, "list_kept_crops", lambda proj, bounds, size, wedge: [(0, 0, 2048, 2048)]
    )
    monkeypatch.setattr(pp, "_crop_crosses_bad_longitude", lambda proj, b: False)

    projection = Projection(CRS.from_epsg(32610), 10, -10)
    pp.predict_pipeline(
        inputs=pp.EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
        projection_json=json.dumps(projection.serialize()),
        bounds=(0, 0, 2048, 2048),
        time_range=(datetime(2024, 1, 1, tzinfo=UTC),) * 2,
        store_path=str(tmp_path / "store.zarr"),
        completed_path=str(tmp_path / "completed"),
        checkpoint_path=str(tmp_path / "ckpt"),
        time_index=0,
        window_size=384,
        scratch_path=str(tmp_path / "scratch"),
    )

    assert calls == ["materialize", "predict"]
    # Widened from the reference timestamp, so the model gets its periods.
    (time_range,) = time_ranges
    assert time_range[1] > time_range[0]
    marker = json.loads(next((tmp_path / "completed").iterdir()).read_text())
    assert marker["skipped_no_data"] == [[0, 0]]
    assert marker["window_size"] == 384
    assert marker["context_pixels"] == 48


def test_context_pixels_pad_the_windows_but_not_what_is_written(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A context margin widens what is materialized and run, not what is written."""
    import multiprocessing.dummy

    pp = sys.modules["rslp.large_scale_embeddings.predict_pipeline"]
    windows: dict[str, tuple[int, ...]] = {}
    reads: list[tuple[int, ...]] = []
    writes: list[tuple[int, ...]] = []

    def fake_materialize(ds_path: UPath, materialize_pipeline_args: object) -> None:
        for window in Dataset(ds_path).load_windows(groups=[pp.PREDICTION_GROUP]):
            windows[window.name] = tuple(window.bounds)
            window.mark_layer_completed(pp.SENTINEL2_LAYER)

    def fake_predict(*args: object, **kwargs: object) -> None:
        for window in Dataset(UPath(tmp_path / "scratch")).load_windows(
            groups=[pp.PREDICTION_GROUP]
        ):
            window.mark_layer_completed(pp.OUTPUT_LAYER)

    def fake_read(d: object, w: object, bounds: tuple, p: int) -> np.ndarray:
        reads.append(tuple(bounds))
        return np.zeros((128, 4, 4), np.int8)

    class _InProcess:
        Pool = staticmethod(multiprocessing.dummy.Pool)

    monkeypatch.setattr(pp, "materialize_dataset", fake_materialize)
    monkeypatch.setattr(pp, "run_model_predict", fake_predict)
    monkeypatch.setattr(pp.multiprocessing, "get_context", lambda name: _InProcess)
    monkeypatch.setattr(pp, "_read_window_embeddings", fake_read)
    monkeypatch.setattr(
        pp, "write_window_region", lambda **kw: writes.append(kw["window_bounds"])
    )
    monkeypatch.setattr(pp, "get_zone_wedge", lambda crs, res: None)
    monkeypatch.setattr(
        pp,
        "list_kept_crops",
        lambda proj, bounds, size, wedge: [(0, 0, 2048, 2048), (2048, 0, 4096, 2048)],
    )
    monkeypatch.setattr(pp, "_crop_crosses_bad_longitude", lambda proj, b: False)
    monkeypatch.setenv(pp.CODE_VERSION_ENV_VAR, "rslp=abc1234 rslearn=def5678")

    projection = Projection(CRS.from_epsg(32610), 10, -10)
    pp.predict_pipeline(
        inputs=pp.EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
        projection_json=json.dumps(projection.serialize()),
        bounds=(0, 0, 4096, 2048),
        time_range=(datetime(2024, 1, 1, tzinfo=UTC),) * 2,
        store_path=str(tmp_path / "store.zarr"),
        completed_path=str(tmp_path / "completed"),
        checkpoint_path=str(tmp_path / "ckpt"),
        time_index=0,
        context_pixels=112,
        scratch_path=str(tmp_path / "scratch"),
    )

    assert windows == {
        "0_0": (-112, -112, 2160, 2160),
        "1_0": (1936, -112, 4208, 2160),
    }
    assert sorted(reads) == [(0, 0, 2048, 2048), (2048, 0, 4096, 2048)]
    assert sorted(writes) == sorted(reads)
    (marker_fname,) = (tmp_path / "completed").iterdir()
    marker = json.loads(marker_fname.read_text())
    assert sorted(marker["written"]) == [[0, 0], [2048, 0]]
    assert marker["context_pixels"] == 112
    assert marker["code_version"] == "rslp=abc1234 rslearn=def5678"
    provenance = json.loads(pp.get_provenance_fname(UPath(marker_fname)).read_text())
    assert provenance["context_pixels"] == 112
    assert provenance["code_version"] == "rslp=abc1234 rslearn=def5678"
    assert provenance["window_bounds"] == {
        "0_0": [-112, -112, 2160, 2160],
        "1_0": [1936, -112, 4208, 2160],
    }


@pytest.mark.parametrize("context_pixels", [-2, 3])
def test_context_pixels_must_fit_the_token_grid(context_pixels: int) -> None:
    """The margin must be a non-negative multiple of the token size."""
    pp = sys.modules["rslp.large_scale_embeddings.predict_pipeline"]

    with pytest.raises(ValueError, match="context_pixels"):
        pp.predict_pipeline(
            inputs=pp.EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
            projection_json="{}",
            bounds=(0, 0, 2048, 2048),
            time_range=(datetime(2024, 1, 1, tzinfo=UTC),) * 2,
            store_path="unused",
            completed_path="unused",
            checkpoint_path="unused",
            time_index=0,
            context_pixels=context_pixels,
        )


def test_mask_invalid_pixels_masks_pixels_without_input() -> None:
    """Pixels without input become nodata; a single zero coordinate is kept."""
    embeddings = np.ones((4, 1, 3), np.int8)
    embeddings[0, 0, 2] = 0
    valid = np.array([[False, True, True]])
    masked = _mask_invalid_pixels(embeddings, valid)
    assert (masked[:, 0, 0] == NODATA_VALUE).all()
    assert masked[:, 0, 1].tolist() == [1, 1, 1, 1]
    assert masked[:, 0, 2].tolist() == [0, 1, 1, 1]


def test_mask_invalid_pixels_rejects_all_zero_vectors() -> None:
    """An all-zero vector where the inputs had data fails the block."""
    embeddings = np.ones((4, 1, 3), np.int8)
    embeddings[:, 0, 1] = 0
    valid = np.array([[False, True, True]])
    with pytest.raises(AllZeroEmbeddingError, match="1 of 2 valid pixels"):
        _mask_invalid_pixels(embeddings, valid)


def test_mask_invalid_pixels_ignores_all_zero_outside_valid() -> None:
    """An all-zero vector the inputs did not cover is just nodata, not a failure."""
    embeddings = np.ones((4, 1, 2), np.int8)
    embeddings[:, 0, 0] = 0
    valid = np.array([[False, True]])
    masked = _mask_invalid_pixels(embeddings, valid)
    assert (masked[:, 0, 0] == NODATA_VALUE).all()
