#!/usr/bin/env python3
"""
Minimal script to submit helios finetune jobs.
"""

import subprocess
import os
import tempfile
import yaml


def submit_job(ckpt_path: str, task_dir: str, task_name: str, cfgs: list[str], model_name: str) -> bool:
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
        "--rslp_project", "helios_finetune_cosine_lr",
        "--experiment_id", exp_id,
    ]
    for cfg in cfgs:
        cmd.append(f"--config_paths+=/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/{task_dir}/{cfg}")

    with tempfile.NamedTemporaryFile(mode="w") as f:
        restore_config = {
            "model": {
                "init_args": {
                    "restore_config": {
                        "restore_path": os.path.join(ckpt_path, "checkpoints", "last.ckpt"),
                        "selector": ["state_dict"],
                        "remap_prefixes": [["model.", ""]]
                    },
                    "model": {
                        "init_args": {
                            "task_embedding": {
                                "class_path": "rslearn.models.task_embedding.TaskMHAEmbedding",
                                "init_args": {
                                    "encoder_embedding_size": 768,
                                    "num_heads": 12,
                                }
                            }
                        }
                    }
                }
            }
        }
        yaml.dump(restore_config, f)
        f.flush()
        cmd.append(f"--config_paths+={f.name}")

        print(" ".join(cmd))

        env = os.environ.copy()
        env["RSLP_PREFIX"] = "/weka/dfive-default/rslearn-eai"
        try:
            subprocess.run(cmd, check=True, env=env)
            print(f"✅ {exp_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {exp_id}: {e}")
            return False


def main():
    """Submit jobs."""
    CKPT_PATHS = {
        "classify_all_fixmetrics": (
            "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/classify_all_fixmetrics__cropland_classification",
            "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/classify_all_fixmetrics__crop_type_classification",
            "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/classify_all_fixmetrics__vessel_classification",
        ),
        "segment_all_fixmetrics": (
            "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/segment_all_fixmetrics__segment",
            "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr/segment_all_fixmetrics__segment_satlas_solar_farm",
        )
    }
 
    TASK_CFG_PAIRS = [
        ("v2_landsat_vessels", "vessel_classification", "finetune_classifier_cosinelr.yaml"),
        ("v2_pastis", "segment", "basecfg_cosinelr.yaml", "basecfg_helios_mm.yaml"),
        ("v2_nandi_crop_type", "crop_type_classification", "finetune_s1_s2_cosinelr.yaml"),
        ("v2_worldcereal_cropland", "cropland_classification", "finetune_s1_s2_cosinelr.yaml"),
        ("v2_landsat_vessels", "vessel_detection", "finetune_detector_cosinelr.yaml"),
    ]

    total_jobs = len(CKPT_PATHS) * len(TASK_CFG_PAIRS)
    print(f"Submitting {total_jobs} jobs...")

    success = 0
    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
    for name, ckpt_paths in CKPT_PATHS.items():
        for task_dir, task_name, *cfgs in TASK_CFG_PAIRS:
            for ckpt_path in ckpt_paths:
                if f"{name}__{task_name}" in ckpt_path:
                    success += int(submit_job(ckpt_path, task_dir, task_name, cfgs, name))
                    break
            else:
                success += int(submit_job(ckpt_paths[0], task_dir, task_name, cfgs, name))
    
    print(f"\nCompleted: {success}/{total_jobs} jobs successful")


if __name__ == "__main__":
    main() 
