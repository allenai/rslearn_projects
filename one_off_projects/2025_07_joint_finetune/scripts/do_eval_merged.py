import sys
import os
import tempfile
import yaml
import json

def load_yaml_config(config_path, substitutions=None):
    """Load a YAML config file with optional string substitutions."""
    with open(config_path, 'r') as f:
        config_str = f.read()
    
    # Apply substitutions if provided
    if substitutions:
        for key, value in substitutions.items():
            if value is not None:
                config_str = config_str.replace(f"{{{key}}}", str(value))

    return yaml.safe_load(config_str)

base_model = "detect_all_v2"

ckpt_path = f"/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/{base_model}"
ckpt_cfg_path = os.path.join(ckpt_path, "checkpoints", "config.yaml")
substitutions = {
    "PATCH_SIZE": 8,
    "ENCODER_EMBEDDING_SIZE": 768,
    "256/PATCH_SIZE": 256 // 8,
    "128/PATCH_SIZE": 128 // 8,
}
cmd = [
    "python", "-m", "rslp.main", "helios", "launch_finetune",
    "--helios_checkpoint_path", "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
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

with tempfile.NamedTemporaryFile(mode="w") as f:
    cfg = load_yaml_config(ckpt_cfg_path, substitutions=substitutions)

    for callback in cfg["trainer"]["callbacks"]:
        config_str = callback["init_args"].get("config_str")
        if config_str is not None:
            full_cfg = json.loads(config_str)
            cfg["model"]["init_args"]["task"] = full_cfg["model"]["init_args"]["task"].copy()
            print("INFO: linked multi task")
            break
    else:
        raise ValueError("No multi task config found")

    cfg["model"]["init_args"]["model"]["init_args"]["restore_config"] = {
        "restore_path": os.path.join(ckpt_path, "checkpoints", "last.ckpt"),
        "selector": ["state_dict"],
        "remap_prefixes": [["model.", ""]]
    }
    cfg["trainer"]["limit_val_batches"] = 1024

    def print_key_diff(d1, d2, prefix=""):
        if isinstance(d1, dict) and isinstance(d2, dict):
            if set(d1.keys()) != set(d2.keys()):
                print(f"{prefix}: ", d1.keys() - d2.keys())
                for k in list(d1.keys()):
                    if k in d2:
                        print_key_diff(d1[k], d2[k], f"{prefix}.{k}")
                    elif "limit_val_batches" not in k:
                        print(f"deleting {prefix}.{k}")
                        del d1[k]
    
    with open(
        '/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_landsat_vessels/finetune_detector_cosinelr.yaml',
        'r'
    ) as f2:
        ex = yaml.safe_load(f2)

    print_key_diff(cfg, ex)
    print("====")
    print_key_diff(cfg, ex)
    print("==")

    yaml.dump(cfg, f)
    f.flush()
    cmd.append(f"--config_paths+={f.name}")

    print(" ".join(cmd))
    print()

    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
    os.system(" ".join(cmd))
