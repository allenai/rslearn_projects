"""Do evals on unmerged MoE models.

Note: this will usually give terrible performance on detection models with FasterRCNN,
since the decoder weights have been spliced, so the cls_scores are over-represented 
without a corresponding change in the nms thresholding.

TODO: get segmentation working (strange issues with the checkpoints?)
"""

import argparse
import os
import tempfile
import yaml
import torch
import json

def load_yaml_config(config_path, substitutions=None):
    with open(config_path, 'r') as f:
        config_str = f.read()
    if substitutions:
        for key, value in substitutions.items():
            if value is not None:
                config_str = config_str.replace(f"{{{key}}}", str(value))
    return yaml.safe_load(config_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("base_model", help="Base model name")
    parser.add_argument("task_name", help="Task name to evaluate")
    parser.add_argument("--max_batches", type=int, default=None, help="Maximum number of batches to evaluate")
    args = parser.parse_args()

    base_model = args.base_model
    task_name = args.task_name
    max_batches = args.max_batches

    base_dir = "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs"
    all_cfgs = {}
    for task in os.listdir(base_dir):
        if task.startswith("v2_"):
            all_cfgs[task] = [
                os.path.join(base_dir, task, cfg)
                for cfg in os.listdir(os.path.join(base_dir, task))
                if cfg.endswith(".yaml")
            ]
    
    all_cfgs["vessel_classification"] = [
        os.path.join(base_dir, "v2_landsat_vessels", "finetune_classifier_cosinelr.yaml"),
    ]
    all_cfgs["vessel_detection"] = [
        os.path.join(base_dir, "v2_landsat_vessels", "finetune_detector_cosinelr.yaml"),
    ]

    ckpt2task = {
        "detect_satlas_wind_turbine": "v2_satlas_wind_turbine_128",
        "detect_satlas_marine_infra": "v2_satlas_marine_infra_128", 
        "detect_sentinel2_vessels": "v2_sentinel2_vessels_128",
        "detect_sentinel1_vessels": "v2_sentinel1_vessels_128",
        "vessel_detection": "vessel_detection",
        "crop_type_classification": "v2_nandi_crop_type",
        "cropland_classification": "v2_worldcereal_cropland",
        "vessel_classification": "vessel_classification",
        "segment": "v2_pastis",
        "segment_satlas_solar_farm": "v2_satlas_solar_farm_128",
    }

    ckpt_path = f"/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/{base_model}__unmerged__{task_name}"
    ckpt_cfg_paths = all_cfgs[ckpt2task[task_name]]
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

    with open(os.path.join(ckpt_path, "checkpoints", "config.yaml")) as cf:
        trunk_cfg = yaml.safe_load(cf)["model"]["init_args"]["model"]["init_args"]["trunk"]
        trunk_layer_init_args = trunk_cfg.pop("init_args")
        trunk_layer_init_args["disable_moe"] = not trunk_layer_init_args.pop("use_moe")
        trunk_cfg["init_args"] = {
            "task_embedding": trunk_layer_init_args.pop("task_embedding"),
            "layers": [
                {
                    "class_path": "rslp.helios.moe.MoETransformer",
                    "init_args": trunk_layer_init_args
                }
            ]
        }
    print(json.dumps(trunk_cfg, indent=2))
    print("=======================")

    with tempfile.NamedTemporaryFile(mode="w") as tmp_ckpt_file:
        with tempfile.NamedTemporaryFile(mode="w") as cfg_file:
            for i, ckpt_cfg_path in enumerate(ckpt_cfg_paths):
                if i == 0:
                    cfg = load_yaml_config(ckpt_cfg_path, substitutions=substitutions)
                    cfg["trainer"]["limit_val_batches"] = max_batches
                    cfg["model"]["init_args"]["model"]["init_args"]["trunk"] = trunk_cfg
                    cfg["model"]["init_args"]["restore_config"] = {
                        "restore_path": tmp_ckpt_file.name,
                        "selector": ["state_dict"],
                        "remap_prefixes": [["model.", ""]]
                    }
                    yaml.dump(cfg, cfg_file)
                    cfg_file.flush()
                    cmd.append(f"--config_paths+={cfg_file.name}")

                else:
                    cmd.append(f"--config_paths+={ckpt_cfg_path}")

            # Patch trunk weights for older version
            ckpt = torch.load(os.path.join(ckpt_path, "checkpoints", "last.ckpt"))
            sd = ckpt["state_dict"]
            for k, v in list(sd.items()):
                new_k = k.replace("model.trunk.transformer", "model.trunk.layers.0")
                sd[new_k] = v
                if new_k != k:
                    del sd[k]
            print("Replaced trunk.transformer with trunk.layers.0")
            torch.save(ckpt, tmp_ckpt_file.name)
            tmp_ckpt_file.flush()

            cmd.append(f"--ckpt_path={tmp_ckpt_file.name}")

            print(" ".join(cmd))
            print()

            os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
            os.system(" ".join(cmd))
