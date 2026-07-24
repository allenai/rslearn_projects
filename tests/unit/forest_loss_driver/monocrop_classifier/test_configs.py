import json
from pathlib import Path

import yaml

from rslp.forest_loss_driver.monocrop_classifier.create_dataset import CLASS_NAMES

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

    assert dataset_config["layers"]["label"]["class_names"] == list(CLASS_NAMES)
    query = dataset_config["layers"]["sentinel2_l2a"]["data_source"]["query_config"]
    assert query["max_matches"] == 23
    assert query["period_duration"] == "30d"
    assert query["space_mode"] == "MOSAIC"


def test_model_configs_share_temporal_contract_and_differ_in_optimizer() -> None:
    frozen = _load_yaml("model_frozen.yaml")
    llrd = _load_yaml("model_llrd.yaml")

    for config in (frozen, llrd):
        data_args = config["data"]["init_args"]
        image_input = data_args["inputs"]["sentinel2_l2a"]
        assert image_input["layers"] == ["sentinel2_l2a"]
        assert image_input["load_all_layers"] is True
        assert image_input["load_all_item_groups"] is True
        assert data_args["task"]["init_args"]["class_names"] == list(CLASS_NAMES)

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

    frozen_optimizer = frozen["model"]["init_args"]["optimizer"]
    assert frozen_optimizer["class_path"] == "rslearn.train.optimizer.AdamW"
    freeze_callbacks = [
        callback
        for callback in frozen["trainer"]["callbacks"]
        if callback["class_path"].endswith("FreezeUnfreeze")
    ]
    assert freeze_callbacks == [
        {
            "class_path": "rslearn.train.callbacks.freeze_unfreeze.FreezeUnfreeze",
            "init_args": {"module_selector": ["model", "encoder", 0]},
        }
    ]

    llrd_optimizer = llrd["model"]["init_args"]["optimizer"]
    assert llrd_optimizer["class_path"].endswith("LayerDecayAdamW")
    assert llrd_optimizer["init_args"]["layer_decay_rate"] == 0.8
    assert llrd_optimizer["init_args"]["num_layers"] == 12
