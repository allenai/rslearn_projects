import json
from pathlib import Path

import yaml

from rslp.forest_loss_driver.monocrop_classifier.create_dataset import (
    LABEL_VECTOR_LAYER,
    MERGED_CLASS_NAMES,
)

CONFIG_DIR = Path("data/forest_loss_driver/monocrop_classifier")
SAMPLER_PATH = (
    "rslp.forest_loss_driver.monocrop_classifier.transforms.PostLossMonthSampler"
)


def _load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open() as f:
        return yaml.safe_load(f)


def test_dataset_config_matches_classes_and_monthly_stack() -> None:
    with (CONFIG_DIR / "config.json").open() as f:
        dataset_config = json.load(f)

    label_vector = dataset_config["layers"][LABEL_VECTOR_LAYER]
    assert label_vector["class_names"] == list(MERGED_CLASS_NAMES)
    assert label_vector["class_property_name"] == "class_name"
    query = dataset_config["layers"]["sentinel2_l2a"]["data_source"]["query_config"]
    assert query["max_matches"] == 23
    assert query["period_duration"] == "30d"
    assert query["space_mode"] == "MOSAIC"


def test_classify_pool_config_matches_dataset_and_temporal_contract() -> None:
    config = _load_yaml("model_classify_pool.yaml")

    data_args = config["data"]["init_args"]
    image_input = data_args["inputs"]["sentinel2_l2a"]
    assert image_input["layers"] == ["sentinel2_l2a"]
    assert image_input["load_all_layers"] is True
    assert image_input["load_all_item_groups"] is True
    assert data_args["inputs"]["targets"]["layers"] == [LABEL_VECTOR_LAYER]

    task_args = data_args["task"]["init_args"]
    assert task_args["property_name"] == "class_name"
    assert task_args["classes"] == list(MERGED_CLASS_NAMES)

    val_sampler = data_args["val_config"]["transforms"][0]
    assert val_sampler["class_path"] == SAMPLER_PATH
    assert val_sampler["init_args"]["num_post_months"] == 6

    test_sampler = data_args["test_config"]["transforms"][0]
    assert test_sampler["class_path"] == SAMPLER_PATH
    assert test_sampler["init_args"]["num_post_months"] == (
        "${MONOCROP_NUM_POST_MONTHS}"
    )
    assert test_sampler["init_args"]["default_num_post_months"] == 6
    assert all(
        transform["class_path"] != SAMPLER_PATH
        for transform in data_args["predict_config"]["transforms"]
    )

    model_args = config["model"]["init_args"]
    decoder = model_args["model"]["init_args"]["decoder"]
    assert decoder[0]["class_path"].endswith("PoolingDecoder")
    assert decoder[0]["init_args"]["out_channels"] == len(MERGED_CLASS_NAMES)

    optimizer = model_args["optimizer"]
    assert optimizer["class_path"].endswith("LayerDecayAdamW")
    assert optimizer["init_args"]["layer_decay_rate"] == 0.8
    assert optimizer["init_args"]["num_layers"] == 12
