"""Create multi-dataset configuration files.

This script merges multiple dataset configurations into a single multi-dataset
configuration file for training models on multiple tasks simultaneously.
"""

import argparse
import json
import os
import tempfile
from copy import deepcopy
from collections import defaultdict
from ntpath import basename
from typing import Any

import yaml


def apply_template(config_str: str, cfg: dict[str, Any]) -> str:
    """Apply template substitutions to a configuration string.

    Args:
        config_str: The configuration string with template placeholders.
        cfg: Dictionary containing values to substitute.

    Returns:
        Configuration string with placeholders replaced.
    """
    config_str = config_str.replace("{CHECKPOINT_PATH}", cfg["helios_checkpoint_path"])
    config_str = config_str.replace("{PATCH_SIZE}", str(cfg["patch_size"]))
    config_str = config_str.replace("{256/PATCH_SIZE}", str(256 // cfg["patch_size"]))
    config_str = config_str.replace("{128/PATCH_SIZE}", str(128 // cfg["patch_size"]))
    config_str = config_str.replace(
        "{ENCODER_EMBEDDING_SIZE}", str(cfg["encoder_embedding_size"])
    )
    return config_str


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, handling list extensions with '+' suffix.

    Args:
        base: Base dictionary to merge into.
        override: Dictionary with values to merge.

    Returns:
        Merged dictionary.
    """
    for k, v in override.items():
        v_copy = v.copy() if hasattr(v, "copy") else v
        if k.endswith("+"):
            k = k[:-1]
            if k not in base:
                base[k] = []
            base[k].extend(v_copy)
        else:
            if isinstance(v, dict):
                base[k] = deep_merge(base.get(k, {}), v_copy)
            else:
                base[k] = v_copy
    return base


def merge_configs(cfg_list: list[str], maker_cfg: dict[str, Any]) -> str:
    """Merge multiple configuration files into a single YAML string.

    Args:
        cfg_list: List of configuration file paths to merge.
        maker_cfg: Configuration dictionary for template substitution.

    Returns:
        YAML string containing merged configuration.
    """
    dicts = []
    for cfg in cfg_list:
        with open(cfg) as f:
            cfg_str = apply_template(f.read(), maker_cfg.get("substitutions", {}))
            dicts.append(yaml.safe_load(cfg_str))
    merged_dict = dicts[0].copy()
    for d in dicts[1:]:
        merged_dict = deep_merge(merged_dict, d)
    return yaml.dump(merged_dict)


def merge_decoder_heads(
    base_cfg: dict[str, Any], 
    all_dataset_info: dict[str, Any], 
    merge_task_labels: bool = False,
) -> None:
    """Merge decoder heads for a multi-task model.

    Repopulates the decoders key in base_cfg.model.init_args.model.init_args, as well as
    supplying relevant information to base_cfg.model.init_args.task.init_args.

    Args:
        base_cfg: Base configuration dictionary.
        all_dataset_info: Dictionary containing information about the datasets, including
            the number of classes and the final decoder index for each task.
        merge_task_labels: Whether to merge task labels. If so, we change the final decoder's
            number of classes to be the sum of all the classes in the merged tasks.
    """
    decoder_id_to_yaml = {}
    decoder_id_to_task = defaultdict(list)
    decoders = base_cfg["model"]["init_args"]["model"]["init_args"]["decoders"]
    for task_name, decoder_list in decoders.items():
        # Assume all decoders with the same last layer are to be
        # merged and have identical architectures
        decoder_id = decoder_list[-1]["class_path"].split(".")[-1]
        decoder_id_to_yaml[decoder_id] = decoder_list
        decoder_id_to_task[decoder_id].append(task_name)
    
    print("merged decoders:")
    for k, v in decoder_id_to_task.items():
        print(f" - {k}: {v}")
    print()

    if merge_task_labels:
        # Assume all decoders have the layer that determines the number of
        # outputs at index 0 (RegressionHead, ClassificationHead, etc are the 
        # final layers but don't modify the number of channels)
        output_layer_idx = 0
        print(f"merging outputs at layer {output_layer_idx}")
        task_label_offsets = {}
        for decoder_id, decoder_list in decoder_id_to_yaml.items():
            num_classes = 0
            num_outputs_keys = []
            for task_name in decoder_id_to_task[decoder_id]:
                num_task_classes = all_dataset_info[task_name]["num_outputs"] 
                num_outputs_keys.append(
                    all_dataset_info[task_name]["num_outputs_key"]
                )
                task_label_offsets[task_name] = {
                    "offset": num_classes,
                    "outputs_key": all_dataset_info[task_name]["outputs_key"],
                }
                num_classes += num_task_classes
                print(f".... {task_name}: {num_task_classes=}")

            assert len(set(num_outputs_keys)) == 1, \
                "cannot have different num_outputs_keys in the same merged head"
            num_outputs_key = num_outputs_keys[0]
            decoder_list[output_layer_idx]["init_args"][num_outputs_key] = num_classes
            print(f"- decoder {decoder_id} has {num_classes} outputs now")

        task_init_args = base_cfg["model"]["init_args"]["task"]["init_args"]
        task_init_args["task_label_offsets"] = task_label_offsets

        for task_name, task_cfg in base_cfg["data"]["init_args"]["data_modules"].items():
            local_task_init_args = task_cfg["init_args"]["task"]["init_args"]
            local_task_init_args["task_label_offsets"] = task_label_offsets

    model_init_args = base_cfg["model"]["init_args"]["model"]["init_args"]
    model_init_args["decoders"] = decoder_id_to_yaml
    model_init_args["decoder_to_target"] = dict(decoder_id_to_task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, required=True, help="Path to multi-dataset maker config"
    )
    args = parser.parse_args()

    with open(args.cfg) as f:
        maker_cfg: dict[str, Any] = yaml.safe_load(f)
    print(json.dumps(maker_cfg, indent=4))
    print()
    print("=" * 80)
    print()

    base_cfg_path = maker_cfg["base_cfg"]
    global_overrides = maker_cfg.get("global_overrides", {})
    local_overrides = maker_cfg.get("local_overrides", {})
    substitutions = maker_cfg.get("substitutions", {})

    if maker_cfg["output_path"] is None:
        s = ""
        for task_cfg in maker_cfg["dataset_cfgs"]:
            if isinstance(task_cfg, list):
                task_cfg = task_cfg[0]
            basename = os.path.basename(task_cfg).replace(".yaml", "")
            s += f"{os.path.basename(os.path.dirname(task_cfg))}__{basename}__"
        maker_cfg["output_path"] = base_cfg_path.replace(
            ".yaml", f"__{s[:-2]}.yaml"
        )

    to_tmp = {}
    for i, cfg in enumerate([base_cfg_path] + maker_cfg["dataset_cfgs"]):
        if isinstance(cfg, list):
            cfg_base_dir = os.path.basename(os.path.dirname(cfg[0]))
            cfg_fnames = [os.path.splitext(os.path.basename(c))[0] for c in cfg]
            cfg_key = f"{cfg_base_dir}__{'__'.join(cfg_fnames)}.yaml"
            to_tmp[cfg_key] = merge_configs(cfg, maker_cfg)
        else:
            with open(cfg) as f:
                to_tmp[cfg] = apply_template(f.read(), substitutions)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dataset_cfgs = {}
        for i, cfg in enumerate(to_tmp):
            cfg_id = os.path.basename(cfg)
            tmp_dataset_cfgs[cfg] = os.path.join(tmpdir, f"{i}_{cfg_id}.yaml")
        tmp_task_buffers = [open(fp, "w+") for fp in tmp_dataset_cfgs.values()]
        try:
            for cfg in to_tmp:
                with open(tmp_dataset_cfgs[cfg], "w+") as f:
                    f.write(to_tmp[cfg])
                    f.flush()

            with open(tmp_dataset_cfgs[base_cfg_path]) as f:
                base_cfg = yaml.safe_load(f)
            with open(base_cfg.pop("all_dataset_info_path")) as f:
                all_dataset_info = yaml.safe_load(f)

            data_modules = {}
            decoders = {}
            batch_sizes = {}
            task = {
                "class_path": "rslearn.train.tasks.multi_task.MultiTask",
                "init_args": {},
            }
            for task_cfg in tmp_dataset_cfgs.values():
                if task_cfg == tmp_dataset_cfgs[base_cfg_path]:
                    continue
                with open(task_cfg) as f:
                    task_cfg = yaml.safe_load(f)
                    subtasks = list(
                        task_cfg["model"]["init_args"]["model"]["init_args"][
                            "decoders"
                        ].keys()
                    )
                    assert len(subtasks) == 1, "Only one subtask per task is supported"

                    deep_merge(task_cfg, local_overrides)

                    task_name = subtasks[0]
                    data_modules[task_name] = task_cfg["data"]
                    batch_sizes[task_name] = task_cfg["data"]["init_args"]["batch_size"]
                    decoders.update(
                        task_cfg["model"]["init_args"]["model"]["init_args"]["decoders"]
                    )

                    for k, v in task_cfg["data"]["init_args"]["task"][
                        "init_args"
                    ].items():
                        try:
                            task["init_args"][k].update(v.copy())  # type: ignore
                        except KeyError:
                            task["init_args"][k] = v.copy()  # type: ignore

            base_cfg["data"]["init_args"]["data_modules"] = data_modules
            base_cfg["model"]["init_args"]["model"]["init_args"]["decoders"] = decoders
            base_cfg["model"]["init_args"]["task"] = deepcopy(task)
            base_cfg["data"]["init_args"]["batch_sizes"] = batch_sizes
            base_cfg = deep_merge(base_cfg, global_overrides)

            merge_options = maker_cfg.get("merge_options", {})
            merge_heads = merge_options.get("merge_heads", False)
            merge_task_labels = merge_options.get("merge_task_labels", False)
            if merge_heads:
                merge_decoder_heads(base_cfg, all_dataset_info, merge_task_labels)

            with open(maker_cfg["output_path"], "w") as f:  # type: ignore
                yaml.dump(base_cfg, f)

        finally:
            for f in tmp_task_buffers:
                f.close()
