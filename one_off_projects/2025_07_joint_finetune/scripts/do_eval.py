import os
import tempfile
import yaml

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

ckpt_path = "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/detect_all_v2__unmerged__vessel_detection"
ckpt_cfg_path = "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_landsat_vessels/finetune_detector_cosinelr.yaml"
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

    cfg["model"]["init_args"]["model"]["init_args"]["restore_config"] = {
        "restore_path": os.path.join(ckpt_path, "checkpoints", "last.ckpt"),
        "selector": ["state_dict"],
        "remap_prefixes": [["model.", ""]]
    }
    cfg["model"]["init_args"]["model"]["init_args"]["task_embedding"] = {
        "class_path": "rslearn.models.task_embedding.TaskMHAEmbedding",
        "init_args": {
            "encoder_embedding_size": 768,
            "num_heads": 12,
        }
    }

    yaml.dump(cfg, f)
    f.flush()
    cmd.append(f"--config_paths+={f.name}")

    print(" ".join(cmd))
    print()

    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
    os.system(" ".join(cmd))
