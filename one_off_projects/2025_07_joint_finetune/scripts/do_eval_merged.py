"""
Run all evals for a label-merged model.

Supports old weight format (ie model.task_embedding without DecoderTrunks).
"""

import argparse
import os
import sys
import tempfile
import yaml
import json
import torch


def load_yaml_config(config_path, substitutions=None):
    with open(config_path, 'r') as f:
        config_str = f.read()
    if substitutions:
        for key, value in substitutions.items():
            if value is not None:
                config_str = config_str.replace(f"{{{key}}}", str(value))
    return yaml.safe_load(config_str)


def resolve_key_diff(d1, d2, prefix=""):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if set(d1.keys()) != set(d2.keys()):
            print(f"{prefix}: ", d1.keys() - d2.keys())
            for k in list(d1.keys()):
                if k in d2:
                    resolve_key_diff(d1[k], d2[k], f"{prefix}.{k}")
                elif "limit_val_batches" not in k:
                    print(f"deleting {prefix}.{k}")
                    del d1[k]


def replace_key(d, key, value):
    if key in d:
        print(f"replaced {key} with {value}, it was {d[key]}")
        d[key] = value
    iter_ = d.values() if isinstance(d, dict) else d
    if hasattr(iter_, "__iter__"):
        for item in d.values():
            if isinstance(item, dict):
                replace_key(item, key, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model name (dir)")
    parser.add_argument("--ckpt", default="last.ckpt", help="ckpt file in {model}/checkpoints/")
    parser.add_argument("--project", default="helios_finetune_cosine_lr", help="project dir for model")
    parser.add_argument("--full", action="store_true", help="set load_all_patches to true")
    args = parser.parse_args()

    base_model = args.model
    ckpt_file = args.ckpt
    project = args.project
    full = args.full

    ckpt_path = f"/weka/dfive-default/rslearn-eai/projects/{project}/{base_model}"
    ckpt_cfg_path = os.path.join(ckpt_path, "checkpoints", "config.yaml")

    helios_ckpt_path = (
        "/weka/dfive-default/helios/checkpoints" \
        "/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000"
    )
    cmd = [
        "python", "-m", "rslp.main", "helios", "launch_finetune",
        "--helios_checkpoint_path", helios_ckpt_path,
        "--patch_size", "8",
        "--encoder_embedding_size", "768",
        "--image_name", "henryh/rslp_multidataset_dev",
        "--cluster+=ai2/titan-cirrascale",
        "--cluster+=ai2/saturn-cirrascale",
        "--cluster+=ai2/ceres-cirrascale",
        "--rslp_project", "helios-debug",
        "--experiment_id", "eval",
        "--local", "true",
        "--do_eval", "true"
    ]
    substitutions = {
        "PATCH_SIZE": 8,
        "ENCODER_EMBEDDING_SIZE": 768,
        "256/PATCH_SIZE": 256 // 8,
        "128/PATCH_SIZE": 128 // 8,
    }

    with tempfile.NamedTemporaryFile(mode="w") as f:
        cfg = load_yaml_config(ckpt_cfg_path, substitutions=substitutions)

        # Get the task label offsets and link tasks
        old_cfg_style = True
        for callback in cfg["trainer"]["callbacks"]:
            config_str = callback["init_args"].get("config_str")
            if config_str is not None:
                full_cfg = json.loads(config_str)
                cfg["model"]["init_args"]["task"] = full_cfg["model"]["init_args"]["task"].copy()
                try:
                    task_label_offsets = full_cfg["model"]["init_args"]["task"]["init_args"]["task_label_offsets"]
                except KeyError:
                    # we are already a new config style
                    old_cfg_style = False
                break
 
        # If full, set load_all_patches and use the map-style all patches dataset
        if full:
            for data_module in cfg["data"]["init_args"]["data_modules"].values():
                data_module["init_args"]["use_in_memory_all_patches_dataset"] = True
                data_module["init_args"]["val_config"]["init_args"]["load_all_patches"] = True
                data_module["init_args"]["test_config"]["init_args"]["load_all_patches"] = True
        else:
            # Ensure that load_all_patches is false everywhere
            replace_key(cfg, "load_all_patches", False)

        # Change the old configs to match the new style
        if old_cfg_style:
            del cfg["model"]["init_args"]["task"]["init_args"]["task_label_offsets"]
            for k, v in cfg["data"]["init_args"]["data_modules"].items():
                if "task_label_offsets" in v["init_args"]["task"]["init_args"]:
                    del v["init_args"]["task"]["init_args"]["task_label_offsets"]
            cfg["model"]["init_args"]["model"]["class_path"] = "rslearn.models.multitask.MultiTaskMergedModel"
            cfg["model"]["init_args"]["model"]["init_args"]["task_label_offsets"] = task_label_offsets

            mm_init = cfg["model"]["init_args"]["model"]["init_args"]
            if "task_embedding" in mm_init:
                del mm_init["task_embedding"]

            if "task_embedding" in mm_init:
                mm_init["trunk"] = {
                    "class_path": "rslearn.models.trunk.DecoderTrunk",
                    "init_args": {
                        "task_embedding": mm_init.pop("task_embedding")
                    }
                }
            elif "task_embedding" in mm_init.get("trunk", {}).get("init_args", {}):
                trunk_args = mm_init.pop("trunk")["init_args"]
                trunk_args["disable_moe"] = (
                    not trunk_args.pop("use_moe") 
                    if "use_moe" in trunk_args else False
                )
                mm_init["trunk"] = {
                    "class_path": "rslearn.models.trunk.DecoderTrunk",
                    "init_args": {
                        "task_embedding": trunk_args.pop("task_embedding"),
                        "layers": [
                            {
                                "class_path": "rslp.helios.moe.MoETransformer",
                                "init_args": trunk_args
                            }
                        ]
                    }
                }

        # Set limit on validation batches
        cfg["trainer"] = {"callbacks": []}
        cfg["trainer"]["limit_val_batches"] = None

        print("=" * 30 + " MODEL CONFIG " + "=" * 30)
        print(json.dumps(cfg["model"], indent=2))
        print("=" * 80)
        print()

        yaml.dump(cfg, f)
        f.flush()
        cmd.append(f"--config_paths+={f.name}")

        # Patch trunk weights for older version
        with tempfile.NamedTemporaryFile(mode="w") as f:
            ckpt_path = os.path.join(ckpt_path, 'checkpoints', ckpt_file)
            sd = torch.load(ckpt_path)
            if any(k.startswith("model.task_embedding") for k in sd["state_dict"]):
                for k, v in list(sd["state_dict"].items()):
                    if k.startswith("model.task_embedding"):
                        sd["state_dict"][k.replace("model.task_embedding", "model.trunk.task_embedding")] = v
                        del sd["state_dict"][k]
                torch.save(sd, f.name)
                f.flush()
                cmd.append(f"--ckpt_path={f.name}")
                print(f"patched weights for older version, saved to {f.name}")

            elif any(k.startswith("model.trunk.transformer") for k in sd["state_dict"]):
                for k, v in list(sd["state_dict"].items()):
                    if k.startswith("model.trunk.transformer"):
                        sd["state_dict"][k.replace("model.trunk.transformer", "model.trunk.layers.0")] = v
                        del sd["state_dict"][k]
                torch.save(sd, f.name)
                f.flush()
                cmd.append(f"--ckpt_path={f.name}")
                print(f"patched weights for older version, saved to {f.name}")

            else:
                print("no need to patch weights")
                cmd.append(f"--ckpt_path={ckpt_path}")

            print()
            print("=" * 20 + " CMD " + "=" * 20)
            print(" ".join(cmd))
            print("=" * 45)
            print()

            os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
            os.system(" ".join(cmd))
