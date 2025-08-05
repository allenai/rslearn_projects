import sys
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

base_model = "detect_all_v2"
task = sys.argv[1]

task2cfgs = {
    "v2_satlas_wind_turbine_128": [
        f"/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_satlas_wind_turbine_128/basecfg_cosinelr.yaml",
        f"/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_satlas_wind_turbine_128/basecfg_helios_mm.yaml",
    ],
    "v2_satlas_marine_infra_128": [
        f"/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_satlas_marine_infra_128/basecfg_cosinelr.yaml",
        f"/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_satlas_marine_infra_128/basecfg_helios_mm.yaml",
    ],
    "v2_sentinel2_vessels_128": [
        f"/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_sentinel2_vessels_128/basecfg_cosinelr.yaml",
        f"/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_sentinel2_vessels_128/basecfg_helios.yaml",
    ],
    "v2_sentinel1_vessels_128": [
        f"/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_sentinel1_vessels_128/basecfg_cosinelr.yaml",
        f"/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_sentinel1_vessels_128/basecfg_helios.yaml",
    ],
    "vessel_detection": [
        f"/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_landsat_vessels/finetune_detector_cosinelr.yaml",
    ],
}

task2ckpt = {
    "v2_satlas_wind_turbine_128": "detect_satlas_wind_turbine",
    "v2_satlas_marine_infra_128": "detect_satlas_marine_infra",
    "v2_sentinel2_vessels_128": "detect_sentinel2_vessels",
    "v2_sentinel1_vessels_128": "detect_sentinel1_vessels",
    "vessel_detection": "vessel_detection",
}

ckpt_path = f"/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/{base_model}__unmerged__{task2ckpt[task]}"
ckpt_cfg_paths = task2cfgs[task]
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
    for i, ckpt_cfg_path in enumerate(ckpt_cfg_paths):
        if i == 0:
            cfg = load_yaml_config(ckpt_cfg_path, substitutions=substitutions)
            cfg["trainer"]["limit_val_batches"] = 204
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
        else:
            cmd.append(f"--config_paths+={ckpt_cfg_path}")

    print(" ".join(cmd))
    print()

    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
    os.system(" ".join(cmd))
