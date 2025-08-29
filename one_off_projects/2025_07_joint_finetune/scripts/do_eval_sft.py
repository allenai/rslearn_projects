import os
import json
import argparse
import tempfile
import yaml

base_dir = "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs"
all_cfgs = {}
for task in os.listdir(base_dir):
    if task.startswith("v2_") and task != "v2_landsat_vessels":
        all_cfgs[task] = [
            os.path.join(base_dir, task, cfg)
            for cfg in os.listdir(os.path.join(base_dir, task))
            if cfg.endswith(".yaml") and "soup" not in cfg
        ]

all_cfgs["vessel_classification"] = [
    os.path.join(base_dir, "v2_landsat_vessels", "finetune_classifier_cosinelr.yaml"),
]
all_cfgs["vessel_detection"] = [
    os.path.join(base_dir, "v2_landsat_vessels", "finetune_detector_cosinelr.yaml"),
]
print("available tasks:")
print(json.dumps(all_cfgs, indent=2))
print()

parser = argparse.ArgumentParser()
parser.add_argument("ckpt_path", type=str, help="Path to the checkpoint")
parser.add_argument("task", type=str, help="Task to evaluate")
parser.add_argument("old_or_new", type=str, help="old or new helios checkpoint")
parser.add_argument("--full", action="store_true", help="Run eval on all patches")
parser.add_argument("--save_eval_path", type=str, help="Path to save eval results")
args = parser.parse_args()

ckpt_cfg_paths = all_cfgs[args.task]
if args.old_or_new.lower() == "old":
    helios_path = "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000"
else:
    helios_path = "/weka/dfive-default/helios/checkpoints/yawenzzzz/latent_mim_cross_random_per_modality_patchdisc_add_contrastive_0.1_1/step400000"
print("using helios path", helios_path)
print()
cmd_template = [
    "RSLP_PREFIX=/weka/dfive-default/rslearn-eai",
    "python", "-m", "rslp.main", "helios", "launch_finetune",
    "--helios_checkpoint_path", helios_path,
    "--patch_size", "8",
    "--encoder_embedding_size", "768",
    "--image_name", "henryh/rslp_multidataset_dev",
    "--cluster+=ai2/titan-cirrascale",
    "--cluster+=ai2/saturn-cirrascale",
    "--cluster+=ai2/ceres-cirrascale",
    "--rslp_project", "helios-debug",
    "--experiment_id", "eval",
    "--local", "true",
    "--do_eval", "true",
    "--allow_missing_weights", "true"
]

substitutions = {
    "PATCH_SIZE": 8,
    "ENCODER_EMBEDDING_SIZE": 768,
    "256/PATCH_SIZE": 256 // 8,
    "128/PATCH_SIZE": 128 // 8,
}

def load_yaml_config(config_path, substitutions=None):
    with open(config_path, 'r') as f:
        config_str = f.read()
    if substitutions:
        for key, value in substitutions.items():
            if value is not None:
                config_str = config_str.replace(f"{{{key}}}", str(value))
    return yaml.safe_load(config_str)

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

cmd_template.append(f"--ckpt_path={args.ckpt_path}")
with tempfile.NamedTemporaryFile(mode="w") as f:
    cmd = cmd_template.copy()
    cmd.append(f"--config_paths+={f.name}")
    if args.save_eval_path:
        cmd.extend(["--save_eval_path", args.save_eval_path])

    cfg = {}
    for ckpt_cfg_path in ckpt_cfg_paths:
        cfg = deep_merge(cfg, load_yaml_config(ckpt_cfg_path, substitutions))

    if args.full:
        cfg["data"]["init_args"]["use_in_memory_all_patches_dataset"] = True
        for split in ["val", "test"]:
            cfg["data"]["init_args"][f"{split}_config"]["load_all_patches"] = True
            cfg["data"]["init_args"][f"{split}_config"]["patch_size"] = substitutions["PATCH_SIZE"]

    print("patched config to have in_memory_all_patches_dataset=True")
    yaml.dump(cfg, f)
    f.flush()

    print(f"Evaluating {args.task} with checkpoint {args.ckpt_path}")
    print()
    print("=" * 80)
    print(" ".join(cmd))
    print("=" * 80)
    print()

    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
    os.system(" ".join(cmd))
