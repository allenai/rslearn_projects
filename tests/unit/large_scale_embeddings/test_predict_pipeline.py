"""Unit tests for rslp.large_scale_embeddings.predict_pipeline."""

import json
import pathlib
import sys
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pytest
from rasterio.crs import CRS
from rslearn.dataset import Dataset
from rslearn.dataset.window import WindowLayerData
from rslearn.utils.geometry import Projection
from upath import UPath

from rslp.large_scale_embeddings.predict_pipeline import (
    _collect_provenance,
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
    """A config.json + weights.pth bundle must be passed as model_path.

    The encoder accepts exactly one of model_id/model_path/checkpoint_path, and only
    the model_path loader understands a bundle. Passing a bundle as checkpoint_path
    sends it down the distributed-checkpoint path, which looks for a model_and_optim
    folder that a bundle does not have.
    """
    from rslp.large_scale_embeddings.predict_pipeline import _checkpoint_arg

    bundle = tmp_path / "v1_3_release_v2"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    (bundle / "weights.pth").write_bytes(b"")
    assert _checkpoint_arg(str(bundle)) == "model_path"


def test_a_training_checkpoint_is_loaded_as_checkpoint_path(
    tmp_path: pathlib.Path,
) -> None:
    """A pre-training checkpoint folder keeps the distributed loader."""
    from rslp.large_scale_embeddings.predict_pipeline import _checkpoint_arg

    ckpt = tmp_path / "step667200"
    (ckpt / "model_and_optim").mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    assert _checkpoint_arg(str(ckpt)) == "checkpoint_path"


def test_only_one_loader_argument_survives(tmp_path: pathlib.Path) -> None:
    """Exactly one loader argument may reach the encoder.

    jsonargparse merges this block onto the config file's init_args rather than
    replacing them, so anything left here is what the encoder sees on top of the file.
    Two set is rejected outright, and nulling the unwanted ones does not work either:
    they arrive as the string "None" and fail their own type check. Removal is the
    only form that survives, which is why the config file names no loader at all.
    """
    import json as _json

    import yaml as _yaml

    from rslp.large_scale_embeddings.predict_pipeline import _get_model_extra_args

    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "config.json").write_text("{}")
    (bundle / "weights.pth").write_bytes(b"")

    # A config file carrying the checkpoint_path placeholder, as ours does.
    config = {
        "model": {
            "init_args": {
                "model": {
                    "init_args": {
                        "encoder": [
                            {
                                "class_path": "rslearn.models.olmoearth_pretrain.model.OlmoEarth",
                                "init_args": {
                                    "projected_register_dim": 128,
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
        patch_size=1,
        window_size=16,
        overlap_size=4,
        compile_model=True,
        batch_size=None,
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
    # The unrelated settings must survive the rewrite.
    assert init_args["projected_register_dim"] == 128


def test_materialize_only_then_predict_reuses_the_scratch(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prefetched scratch dataset is materialized once and then only predicted on.

    This is the contract the worker's prefetch relies on: the first call fills
    scratch_path and writes no marker, the second skips straight to inference.
    """
    pp = sys.modules["rslp.large_scale_embeddings.predict_pipeline"]

    calls: list[str] = []

    def fake_materialize(ds_path: UPath, materialize_pipeline_args: object) -> None:
        calls.append("materialize")
        for window in Dataset(ds_path).load_windows(groups=[pp.PREDICTION_GROUP]):
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
    kwargs = dict(
        inputs=pp.EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
        projection_json=json.dumps(projection.serialize()),
        bounds=(0, 0, 2048, 2048),
        time_range=(datetime(2024, 1, 1, tzinfo=UTC),) * 2,
        store_path=str(tmp_path / "store.zarr"),
        completed_path=str(tmp_path / "completed"),
        checkpoint_path=str(tmp_path / "ckpt"),
        time_index=0,
        scratch_path=str(tmp_path / "scratch"),
    )

    pp.predict_pipeline(materialize_only=True, **kwargs)  # type: ignore[arg-type]
    assert calls == ["materialize"]
    assert (tmp_path / "scratch" / pp.MATERIALIZED_SENTINEL).exists()
    assert not (tmp_path / "completed").exists()

    pp.predict_pipeline(**kwargs)  # type: ignore[arg-type]
    assert calls == ["materialize", "predict"]
    marker = json.loads(next((tmp_path / "completed").iterdir()).read_text())
    # The fake predict writes no output, so the crop counts as skipped.
    assert marker["skipped_no_data"] == [[0, 0]]


def test_materialize_only_needs_a_scratch_path() -> None:
    """Without scratch_path the materialized data would be deleted on return."""
    pp = sys.modules["rslp.large_scale_embeddings.predict_pipeline"]

    with pytest.raises(ValueError, match="scratch_path"):
        pp.predict_pipeline(
            inputs=pp.EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
            projection_json="{}",
            bounds=(0, 0, 2048, 2048),
            time_range=(datetime(2024, 1, 1, tzinfo=UTC),) * 2,
            store_path="unused",
            completed_path="unused",
            checkpoint_path="unused",
            time_index=0,
            materialize_only=True,
        )


def test_predict_renders_pca_inline_and_marks_the_render_stage_done(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With pca paths set, each window is rendered from memory and the stage marked.

    The render marker is what keeps the render stage from reading every block back
    out of the store a second time.
    """
    import multiprocessing.dummy

    pp = sys.modules["rslp.large_scale_embeddings.predict_pipeline"]
    rendered: list[tuple[int, int]] = []

    def fake_materialize(ds_path: UPath, materialize_pipeline_args: object) -> None:
        for window in Dataset(ds_path).load_windows(groups=[pp.PREDICTION_GROUP]):
            window.mark_layer_completed(pp.SENTINEL2_LAYER)

    def fake_predict(*args: object, **kwargs: object) -> None:
        for window in Dataset(UPath(tmp_path / "scratch")).load_windows(
            groups=[pp.PREDICTION_GROUP]
        ):
            window.mark_layer_completed(pp.OUTPUT_LAYER)

    def fake_render(**kw: Any) -> bool:
        rendered.append(kw["crop_offset"])
        assert kw["dest_time_index"] == 7
        # The second window has no valid pixels.
        return kw["crop_offset"][0] == 0

    class _InProcess:
        Pool = staticmethod(multiprocessing.dummy.Pool)

    monkeypatch.setattr(pp, "materialize_dataset", fake_materialize)
    monkeypatch.setattr(pp, "run_model_predict", fake_predict)
    monkeypatch.setattr(pp.multiprocessing, "get_context", lambda name: _InProcess)
    monkeypatch.setattr(
        pp, "_read_window_embeddings", lambda d, w, p: np.zeros((128, 4, 4), np.int8)
    )
    monkeypatch.setattr(pp, "write_window_region", lambda **kw: None)
    monkeypatch.setattr(pp, "render_window", fake_render)
    monkeypatch.setattr(pp.PcaArtifact, "load", staticmethod(lambda path: object()))
    monkeypatch.setattr(pp, "pca_time_index", lambda *a: 7)
    monkeypatch.setattr(pp, "get_zone_wedge", lambda crs, res: None)
    monkeypatch.setattr(
        pp,
        "list_kept_crops",
        lambda proj, bounds, size, wedge: [(0, 0, 2048, 2048), (2048, 0, 4096, 2048)],
    )
    monkeypatch.setattr(pp, "_crop_crosses_bad_longitude", lambda proj, b: False)

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
        scratch_path=str(tmp_path / "scratch"),
        pca_artifact_path="gs://bucket/basis",
        pca_store_path=str(tmp_path / "pca.zarr"),
        pca_completed_path=str(tmp_path / "pca_completed"),
        pca_max_level=2,
    )

    assert sorted(rendered) == [(0, 0), (2048, 0)]
    (pca_marker_fname,) = (tmp_path / "pca_completed").iterdir()
    pca_marker = json.loads(pca_marker_fname.read_text())
    assert pca_marker["rendered"] == [[0, 0]]
    assert pca_marker["skipped_empty"] == [[2048, 0]]
    assert pca_marker["max_level"] == 2
    (marker_fname,) = (tmp_path / "completed").iterdir()
    assert pca_marker["source_marker"] == str(marker_fname)
    # Named the way the render stage looks it up, so that stage skips this tile.
    assert pca_marker_fname.name == f"completed_{marker_fname.name}"


def test_pca_paths_must_be_set_together() -> None:
    """A partial pca config would render with nowhere to record it, or vice versa."""
    pp = sys.modules["rslp.large_scale_embeddings.predict_pipeline"]

    with pytest.raises(ValueError, match="must be set together"):
        pp.predict_pipeline(
            inputs=pp.EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
            projection_json="{}",
            bounds=(0, 0, 2048, 2048),
            time_range=(datetime(2024, 1, 1, tzinfo=UTC),) * 2,
            store_path="unused",
            completed_path="unused",
            checkpoint_path="unused",
            time_index=0,
            pca_artifact_path="gs://bucket/basis",
        )
