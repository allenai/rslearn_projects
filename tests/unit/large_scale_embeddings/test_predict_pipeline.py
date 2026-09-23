"""Unit tests for rslp.large_scale_embeddings.predict_pipeline."""

import pathlib
from datetime import UTC, datetime

from rslearn.dataset.window import WindowLayerData
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
    """The unchosen loader arguments must be removed, not merely left unset.

    The model config ships a placeholder for whichever loader it was written against.
    Setting model_path beside that placeholder leaves two of the three set, and the
    encoder rejects that outright -- so a bundle would fail at model construction.
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
                                    "checkpoint_path": "/path/to/checkpoint_dir",
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
    set_args = [
        k
        for k in ("model_id", "model_path", "checkpoint_path")
        if init_args.get(k) is not None
    ]
    assert set_args == ["model_path"], f"expected only model_path set, got {set_args}"
    assert init_args["model_path"] == str(bundle)
    # Explicitly null, not absent: jsonargparse merges this block onto the config
    # file's init_args, so a dropped key lets the file's own placeholder reappear and
    # two loaders end up set, which the encoder rejects outright.
    assert init_args["checkpoint_path"] is None
    assert init_args["model_id"] is None
    # The unrelated settings must survive the rewrite.
    assert init_args["projected_register_dim"] == 128
