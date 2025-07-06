from copy import deepcopy
from ntpath import basename
import tempfile
import os
import json
import yaml
import argparse

def apply_template(config_str, helios_checkpoint_path, patch_size, encoder_embedding_size):
    config_str = config_str.replace("{CHECKPOINT_PATH}", helios_checkpoint_path)
    config_str = config_str.replace("{PATCH_SIZE}", str(patch_size))
    config_str = config_str.replace("{256/PATCH_SIZE}", str(256 // patch_size))
    config_str = config_str.replace("{128/PATCH_SIZE}", str(128 // patch_size))
    config_str = config_str.replace(
        "{ENCODER_EMBEDDING_SIZE}", str(encoder_embedding_size)
    )
    return config_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_cfg", type=str, required=True, help="Path to base config")
    parser.add_argument("--task_cfgs", nargs="+", required=True, help="Paths to task configs")
    parser.add_argument("--output_path", type=str, default=None, help="Path to output config")
    parser.add_argument("--helios_checkpoint_path", type=str, required=True, help="Path to Helios checkpoint")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size")
    parser.add_argument("--encoder_embedding_size", type=int, default=768, help="Encoder embedding size")
    parser.add_argument("--max_train_patches", type=int, default=None, help="Maximum number of train samples per task")
    parser.add_argument("--max_val_patches", type=int, default=None, help="Maximum number of val samples per task")
    args = parser.parse_args()

    if args.output_path is None:
        s = ""
        for task_cfg in args.task_cfgs:
            basename = os.path.basename(task_cfg).replace(".yaml", "")
            s += f"{os.path.basename(os.path.dirname(task_cfg))}__{basename}__"
        args.output_path = args.base_cfg.replace(".yaml", f"__{s[:-2]}.yaml")

    to_tmp = {}
    for cfg in [args.base_cfg] + args.task_cfgs:
        with open(cfg, "r") as f:
            config_str = f.read()
        to_tmp[cfg] = apply_template(
            config_str,
            args.helios_checkpoint_path,
            args.patch_size,
            args.encoder_embedding_size,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_paths = [os.path.join(tmpdir, f"{os.path.basename(cfg)}") for cfg in to_tmp]
        file_paths = {cfg: fp for cfg, fp in zip(to_tmp, file_paths)}
        files = [open(fp, "w+") for fp in file_paths.values()]
        try:
            for cfg, f in zip(to_tmp, files):
                f.write(to_tmp[cfg])
                f.flush()

            with open(file_paths[args.base_cfg], "r") as f:
                base_cfg = yaml.safe_load(f)

            dataset_configs = {}
            decoders = {}
            task = {"class_path": "rslearn.train.tasks.multi_task.MultiTask", "init_args": {}}
            for task_cfg in file_paths.values():
                if task_cfg == file_paths[args.base_cfg]:
                    continue
                with open(task_cfg, "r") as f:
                    task_cfg = yaml.safe_load(f)
                    subtasks = list(task_cfg["model"]["init_args"]["model"]["init_args"]["decoders"].keys())
                    assert len(subtasks) == 1, "Only one subtask per task is supported"

                    task_name = subtasks[0]
                    dataset_configs[task_name] = task_cfg["data"]
                    decoders.update(task_cfg["model"]["init_args"]["model"]["init_args"]["decoders"])

                    if args.max_train_patches is not None:
                        task_cfg["data"]["init_args"]["train_config"]["num_patches"] = args.max_train_patches
                    if args.max_val_patches is not None:
                        task_cfg["data"]["init_args"]["val_config"]["num_patches"] = args.max_val_patches

                    for k, v in task_cfg["data"]["init_args"]["task"]["init_args"].items(): 
                        try:
                            task["init_args"][k].update(v)
                        except:
                            task["init_args"][k] = v

            base_cfg["data"]["init_args"]["dataset_configs"] = dataset_configs
            base_cfg["model"]["init_args"]["model"]["init_args"]["decoders"] = decoders
            base_cfg["data"]["init_args"]["task"] = deepcopy(task)

            # Shouldn't need this due to argument linking in lightning cli?
            base_cfg["model"]["init_args"]["task"] = deepcopy(task)

            with open(args.output_path, "w") as f:
                yaml.dump(base_cfg, f)

            print(json.dumps(base_cfg, indent=4))
            
        finally:
            for f in files:
                f.close()
