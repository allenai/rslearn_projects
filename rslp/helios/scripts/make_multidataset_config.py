from copy import deepcopy
from ntpath import basename
import tempfile
import os
import json
import yaml
import argparse

def apply_template(config_str, cfg):
    config_str = config_str.replace("{CHECKPOINT_PATH}", cfg["helios_checkpoint_path"])
    config_str = config_str.replace("{PATCH_SIZE}", str(cfg["patch_size"]))
    config_str = config_str.replace("{256/PATCH_SIZE}", str(256 // cfg["patch_size"]))
    config_str = config_str.replace("{128/PATCH_SIZE}", str(128 // cfg["patch_size"]))
    config_str = config_str.replace(
        "{ENCODER_EMBEDDING_SIZE}", str(cfg["encoder_embedding_size"])
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
            cfg_str = apply_template(f.read(), maker_cfg)
            dicts.append(yaml.safe_load(cfg_str))
    merged_dict = dicts[0].copy()
    for d in dicts[1:]:
        merged_dict = deep_merge(merged_dict, d)
    return yaml.dump(merged_dict)


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
        if isinstance(cfg, list):
            cfg_key = "__".join(cfg)
            to_tmp[cfg_key] = merge_configs(cfg, maker_cfg)
            maker_cfg["task_cfgs"][i - 1] = cfg_key
        else:
            with open(cfg, "r") as f:
                to_tmp[cfg] = apply_template(f.read(), maker_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_task_cfgs = {cfg: os.path.join(tmpdir, f"{os.path.basename(cfg)}") for cfg in to_tmp}
        tmp_task_buffers = [open(fp, "w+") for fp in tmp_task_cfgs.values()]
        try:
            for cfg in to_tmp:
                with open(tmp_task_cfgs[cfg], "w+") as f:
                    f.write(to_tmp[cfg])
                    f.flush()

            with open(tmp_task_cfgs[maker_cfg["base_cfg"]], "r") as f:
                base_cfg = yaml.safe_load(f)

            dataset_configs = {}
            decoders = {}
            task = {"class_path": "rslearn.train.tasks.multi_task.MultiTask", "init_args": {}}
            for task_cfg in tmp_task_cfgs.values():
                if task_cfg == tmp_task_cfgs[maker_cfg["base_cfg"]]:
                    continue
                with open(task_cfg, "r") as f:
                    task_cfg = yaml.safe_load(f)
                    subtasks = list(task_cfg["model"]["init_args"]["model"]["init_args"]["decoders"].keys())
                    assert len(subtasks) == 1, "Only one subtask per task is supported"

                    task_name = subtasks[0]
                    dataset_configs[task_name] = task_cfg["data"]
                    decoders.update(task_cfg["model"]["init_args"]["model"]["init_args"]["decoders"])

                    if maker_cfg.get("max_train_patches") is not None:
                        task_cfg["data"]["init_args"]["train_config"]["num_patches"] = maker_cfg["max_train_patches"]
                    if maker_cfg.get("max_val_patches") is not None:
                        task_cfg["data"]["init_args"]["val_config"]["num_patches"] = maker_cfg["max_val_patches"]
                    if maker_cfg.get("batch_size") is not None:
                        task_cfg["data"]["init_args"]["batch_size"] = maker_cfg["batch_size"]
                    
                    for k, v in task_cfg["data"]["init_args"]["task"]["init_args"].items(): 
                        try:
                            task["init_args"][k].update(v.copy())
                        except:
                            task["init_args"][k] = v.copy()

            base_cfg["data"]["init_args"]["dataset_configs"] = dataset_configs
            base_cfg["model"]["init_args"]["model"]["init_args"]["decoders"] = decoders
            base_cfg["data"]["init_args"]["task"] = deepcopy(task)

            # Shouldn't need this due to argument linking in lightning cli?
            base_cfg["model"]["init_args"]["task"] = deepcopy(task)

            with open(maker_cfg["output_path"], "w") as f:
                yaml.dump(base_cfg, f)

            print(json.dumps(base_cfg, indent=4))
 
        finally:
            for f in tmp_task_buffers:
                f.close()
