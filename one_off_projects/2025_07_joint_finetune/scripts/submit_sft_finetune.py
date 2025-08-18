#!/usr/bin/env python3
"""
Submit finetuning jobs.
"""

import subprocess
import os
import tempfile
import json
import yaml

RUN = True
DEBUG = False
PROJECT_NAME = 'helios-debug' if DEBUG else "2025_07_29_helios_finetune"


def submit_job(task_dir: str, task_name: str, cfgs: list[str], model_name: str, ckpt_path: str) -> bool:
    """Submit a single helios finetune job."""
    exp_id = model_name + "__" + task_name
    cmd = [
        "python", "-m", "rslp.main", "helios", "launch_finetune",
        "--helios_checkpoint_path", ckpt_path,
        "--patch_size", "8",
        "--encoder_embedding_size", "768",
        "--image_name", "henryh/rslp_multidataset_dev",
        "--cluster+=ai2/titan-cirrascale",
        "--cluster+=ai2/saturn-cirrascale", 
        "--cluster+=ai2/ceres-cirrascale",
        "--rslp_project", PROJECT_NAME,
        "--experiment_id", exp_id,
    ]
    for cfg in cfgs:
        cmd.append(f"--config_paths+=/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/{task_dir}/{cfg}")
    cmd.append("--config_paths+=/weka/dfive-default/ryanp/rslearn_projects/data/helios/v2_shared/helios_freeze_then_lowlr.yaml")

    if DEBUG:
        cmd.append("--local")
        cmd.append("true")

    with tempfile.NamedTemporaryFile(mode="w") as f:
        restore_config = {
            "model": {
                "init_args": {
                    "restore_config": {
                        "restore_path": os.path.join(ckpt_path, "checkpoints", "last.ckpt"),
                        "selector": ["state_dict"],
                        "remap_prefixes": [["model.", ""]]
                    },
                }
            }
        }
        yaml.dump(restore_config, f)
        f.flush()
        cmd.append(f"--config_paths+={f.name}")

        print()
        print("RESTORE_CONFIG")
        print(json.dumps(restore_config, indent=4))
        print(" ".join(cmd))

        env = os.environ.copy()
        env["RSLP_PREFIX"] = "/weka/dfive-default/rslearn-eai"
        try:
            if RUN:
                subprocess.run(cmd, check=True, env=env)
            print(f"✅ {exp_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {exp_id}: {e}")
            return False


def main():
    """Submit jobs."""
    CKPT_PATHS = {
        "medium_v1": "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/medium_bsVAR_gradaccum8_freeze5_refill_nolfmc_limit50k"
    }
    TASK_CFG_PAIRS = [
        ("v2_pastis", "pastis", "basecfg_cosinelr.yaml", "basecfg_helios_mm.yaml"),
        ("v2_lfmc", "lfmc", "finetune_s1_s2_srtm_cosinelr.yaml"),
        ("v2_nandi_crop_type", "nandi_crop_type", "finetune_s1_s2_cosinelr.yaml"),
        ("v2_worldcereal_cropland", "worldcereal_cropland", "finetune_s1_s2_cosinelr.yaml"),
        ("v2_landsat_vessels", "landsat_vessel_classify", "finetune_classifier_cosinelr.yaml"),
        ("v2_landsat_vessels", "landsat_vessel_detect", "finetune_detector_cosinelr.yaml"),
    ]

    print(f"Submitting {len(TASK_CFG_PAIRS)} jobs...")
    success = 0
    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
    for model_name, ckpt_path in CKPT_PATHS.items():
        for task_dir, task_name, *cfgs in TASK_CFG_PAIRS:
            success += int(submit_job(task_dir, task_name, cfgs, model_name, ckpt_path))
            if DEBUG:
                break
        if DEBUG:
            break
    print(f"\nCompleted: {success}/{len(TASK_CFG_PAIRS)} jobs successful")


if __name__ == "__main__":
    main() 
