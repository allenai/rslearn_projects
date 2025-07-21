"""Create multi-dataset configuration files.

This script merges multiple dataset configurations into a single multi-dataset
configuration file for training models on multiple tasks simultaneously.
"""

import argparse
import json
import os
import tempfile
from copy import deepcopy
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
        tmp_dataset_cfgs = {
            cfg: os.path.join(tmpdir, f"{os.path.basename(cfg)}") for cfg in to_tmp
        }
        tmp_task_buffers = [open(fp, "w+") for fp in tmp_dataset_cfgs.values()]
        try:
            for cfg in to_tmp:
                with open(tmp_dataset_cfgs[cfg], "w+") as f:
                    f.write(to_tmp[cfg])
                    f.flush()

            with open(tmp_dataset_cfgs[base_cfg_path]) as f:
                base_cfg = yaml.safe_load(f)

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
            base_cfg = deep_merge(base_cfg, global_overrides)

            with open(maker_cfg["output_path"], "w") as f:  # type: ignore
                yaml.dump(base_cfg, f)

        finally:
            for f in tmp_task_buffers:
                f.close()
