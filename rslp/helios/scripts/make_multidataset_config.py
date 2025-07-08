from copy import deepcopy
from ntpath import basename
import tempfile
import os
import json
import yaml
import argparse

def apply_template(config_str, cfg, cfg_path):
    config_str = config_str.replace("{CHECKPOINT_PATH}", cfg["helios_checkpoint_path"])
    config_str = config_str.replace("{PATCH_SIZE}", str(cfg["patch_size"]))
    config_str = config_str.replace("{256/PATCH_SIZE}", str(256 // cfg["patch_size"]))
    config_str = config_str.replace("{128/PATCH_SIZE}", str(128 // cfg["patch_size"]))
    config_str = config_str.replace(
        "{ENCODER_EMBEDDING_SIZE}", str(cfg["encoder_embedding_size"])
    )
    cfg_name = cfg_path.split(os.path.sep)[-2]
    config_str = config_str.replace(
        "target/segment/", f"target/{cfg_name}/segment/"
    )
    return config_str

def deep_merge(base, override):
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

def merge_configs(cfg_list, maker_cfg):
    dicts = []
    for cfg in cfg_list:
        with open(cfg, "r") as f:
            cfg_str = apply_template(f.read(), maker_cfg, cfg)
            dicts.append(yaml.safe_load(cfg_str))
    merged_dict = dicts[0].copy()
    for d in dicts[1:]:
        merged_dict = deep_merge(merged_dict, d)
    return yaml.dump(merged_dict)

def get_dataset_name(cfg):
    if isinstance(cfg, list):
        cfg = cfg[0]
    split = cfg.split(os.path.sep)
    dataset_name, filename = split[-2:]
    if "classifier" in filename:
        return f"{dataset_name}_classify"
    elif "detector" in filename:
        return f"{dataset_name}_detect"
    else:
        # Assume we never use multiple configs from the same dataset, unless
        # we're doing a different task (ex: classifier vs detector in landsat vessels)
        return dataset_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, help="Path to multi-dataset maker config")
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        maker_cfg = yaml.safe_load(f)
    print(json.dumps(maker_cfg, indent=4))
    print()
    print("=" * 80)
    print()

    if maker_cfg["output_path"] is None:
        s = ""
        for task_cfg in maker_cfg["task_cfgs"]:
            if isinstance(task_cfg, list):
                task_cfg = task_cfg[0]
            basename = os.path.basename(task_cfg).replace(".yaml", "")
            s += f"{os.path.basename(os.path.dirname(task_cfg))}__{basename}__"
        maker_cfg["output_path"] = maker_cfg["base_cfg"].replace(".yaml", f"__{s[:-2]}.yaml")

    to_tmp = {}
    for i, cfg in enumerate([maker_cfg["base_cfg"]] + maker_cfg["task_cfgs"]):
        dataset_name = get_dataset_name(cfg) if i > 0 else "base"
        if isinstance(cfg, list):
            to_tmp[dataset_name] = merge_configs(cfg, maker_cfg)
            maker_cfg["task_cfgs"][i - 1] = dataset_name
        else:
            with open(cfg, "r") as f:
                to_tmp[dataset_name] = apply_template(f.read(), maker_cfg, cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_to_fpath = {name: os.path.join(tmpdir, f"{os.path.basename(name)}") for name in to_tmp}
        dataset_buffers = [open(fp, "w+") for fp in dataset_to_fpath.values()]
        try:
            for dataset in to_tmp:
                with open(dataset_to_fpath[dataset], "w+") as f:
                    f.write(to_tmp[dataset])
                    f.flush()

            with open(dataset_to_fpath["base"], "r") as f:
                base_cfg = yaml.safe_load(f)

            task_class_path = "rslearn.train.tasks.multi_task.MultiDatasetTask"
            dataset_configs = {}
            decoders = {}
            task = {
                "class_path": task_class_path,
                "init_args": {
                    "input_mapping": {},
                    "tasks": {},
                }
            }
            for dataset_name, task_cfg_path in dataset_to_fpath.items():
                if dataset_name == "base":
                    continue
                with open(task_cfg_path, "r") as f:  # type: ignore
                    # Assume all the encoders are configured the same
                    task_cfg = yaml.safe_load(f)
                    data_init_args = deepcopy(task_cfg["data"]["init_args"])
                    model_init_args = deepcopy(task_cfg["model"]["init_args"]["model"]["init_args"])
                    subtasks = list(model_init_args["decoders"].keys())

                    if maker_cfg.get("max_train_patches") is not None:
                        data_init_args["train_config"]["num_patches"] = maker_cfg["max_train_patches"]
                    if maker_cfg.get("max_val_patches") is not None:
                        data_init_args["val_config"]["num_patches"] = maker_cfg["max_val_patches"]
                    if maker_cfg.get("batch_size") is not None:
                        data_init_args["batch_size"] = maker_cfg["batch_size"]

                    task["init_args"]["tasks"][dataset_name] = deepcopy(data_init_args["task"]["init_args"]["tasks"])
                    task["init_args"]["input_mapping"][dataset_name] = deepcopy(data_init_args["task"]["init_args"]["input_mapping"])

                    data_init_args["task"]["class_path"] = task_class_path
                    for k, v in data_init_args["task"]["init_args"].items():
                        data_init_args["task"]["init_args"][k] = {
                            dataset_name: v
                        }

                    dataset_configs[dataset_name] = data_init_args
                    decoders[dataset_name] = model_init_args["decoders"]

            base_cfg["data"]["init_args"]["dataset_configs"] = dataset_configs
            base_cfg["data"]["init_args"]["task"] = deepcopy(task)

            base_cfg["model"]["init_args"]["model"]["init_args"]["decoders"] = decoders
            base_cfg["model"]["init_args"]["task"] = deepcopy(task)
            # Shouldn't need this due to argument linking in lightning cli?

            with open(maker_cfg["output_path"], "w") as f:
                yaml.dump(base_cfg, f)

            print(json.dumps(base_cfg, indent=4))
 
        finally:
            for f in dataset_buffers:
                f.close()
