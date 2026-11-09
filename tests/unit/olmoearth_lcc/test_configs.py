"""Check that the LCC configs match the class layouts in output_bands.py."""

import json
import pathlib
from typing import Any

import pytest
import yaml

from rslp.olmoearth_lcc.lcc_model import output_bands
from rslp.olmoearth_lcc.lcc_model.output_bands import (
    LC_CLASS_NAMES,
    POST_CHANGE_CATEGORY_NAMES,
    PRE_CHANGE_CATEGORY_NAMES,
)

CONFIG_DIR = pathlib.Path(__file__).parents[3] / "data" / "olmoearth_lcc" / "lcc_model"
MODEL_CONFIGS = sorted(CONFIG_DIR.glob("*.yaml")) + sorted(CONFIG_DIR.glob("emb_v13/*.yaml"))

CLASS_NAMES = {
    "src": LC_CLASS_NAMES,
    "dst": LC_CLASS_NAMES,
    "pre_change": PRE_CHANGE_CATEGORY_NAMES,
    "post_change": POST_CHANGE_CATEGORY_NAMES,
}
CHANGE_MODEL_ARGS = {
    "src": "num_classes_src",
    "dst": "num_classes_dst",
    "pre_change": "num_classes_pre_change",
    "post_change": "num_classes_post_change",
}


def _load_json(fname: str) -> dict[str, Any]:
    with (CONFIG_DIR / fname).open() as f:
        return json.load(f)


def _bands(layer: dict[str, Any]) -> list[str]:
    return layer["band_sets"][0]["bands"]


def test_model_configs_exist() -> None:
    assert len(MODEL_CONFIGS) > 0


@pytest.mark.parametrize("path", MODEL_CONFIGS, ids=lambda p: p.name)
def test_model_config_class_layout(path: pathlib.Path) -> None:
    text = path.read_text()
    assert "label_same_change" not in text
    cfg = yaml.safe_load(text)

    model = cfg["model"]["init_args"]["model"]
    model_args = model["init_args"]
    for task_name, class_names in CLASS_NAMES.items():
        if model["class_path"].endswith("ChangeModel"):
            assert model_args[CHANGE_MODEL_ARGS[task_name]] == len(class_names)
        else:
            convs = [
                layer
                for layer in model_args["decoders"][task_name]
                if "out_channels" in layer.get("init_args", {})
            ]
            assert convs[-1]["init_args"]["out_channels"] == len(class_names)

        task_args = cfg["data"]["init_args"]["task"]["init_args"]["tasks"][task_name][
            "init_args"
        ]
        assert task_args["num_classes"] == len(class_names)
        if "class_names" in task_args:
            assert task_args["class_names"] == class_names


def test_dataset_config_layers() -> None:
    layers = _load_json("config.json")["layers"]
    assert "label_same_change" not in layers
    assert layers["label_src"]["class_names"] == LC_CLASS_NAMES
    assert layers["label_dst"]["class_names"] == LC_CLASS_NAMES
    assert layers["label_pre_change"]["class_names"] == PRE_CHANGE_CATEGORY_NAMES
    assert layers["label_post_change"]["class_names"] == POST_CHANGE_CATEGORY_NAMES
    assert _bands(layers["output_change"]) == output_bands.OUTPUT_BANDS


def test_predict_config_output_bands() -> None:
    layers = _load_json("config_predict.json")["layers"]
    assert _bands(layers["output_change"]) == output_bands.OUTPUT_BANDS


def test_predict_rslearn_config_output_bands() -> None:
    layers = _load_json("config_predict_rslearn.json")["layers"]
    assert (
        _bands(layers["output_src"])
        == output_bands.OUTPUT_BANDS[
            output_bands.SRC_BAND_OFFSET : output_bands.DST_BAND_OFFSET
        ]
    )
    assert (
        _bands(layers["output_dst"])
        == output_bands.OUTPUT_BANDS[
            output_bands.DST_BAND_OFFSET : output_bands.TS_PRE_DAYS_BAND
        ]
    )
    assert _bands(layers["output_pre_change"]) == output_bands.PRE_CHANGE_BANDS
    assert _bands(layers["output_post_change"]) == output_bands.POST_CHANGE_BANDS


def test_output_band_count() -> None:
    assert len(output_bands.OUTPUT_BANDS) == 62
