#!/usr/bin/env python3
"""
Minimal script to submit helios finetune jobs for dataset percentage sweep.
"""

import subprocess
import os


def submit_job(ckpt_path: str, task: str, cfgs: list[str], name: str) -> bool:
    """Submit a single helios finetune job."""
    # Generate experiment ID: {CKPT_PATH}__{TASK}__{CFG}
    exp_id = f"{name}__{task}__{cfgs[-1]}".replace(".yaml", "")
    
    cmd = [
        "python", "-m", "rslp.main", "helios", "launch_finetune",
        "--helios_checkpoint_path", ckpt_path,
        "--patch_size", "8",
        "--encoder_embedding_size", "768",
        "--image_name", "henryh/rslp_multidataset_dev",
        "--cluster+=ai2/titan-cirrascale",
        "--cluster+=ai2/saturn-cirrascale", 
        "--cluster+=ai2/ceres-cirrascale",
        "--rslp_project", "helios_finetune_dataset_percentage",
        "--experiment_id", exp_id
    ]
    for cfg in cfgs:
        cmd.append(f"--config_paths+=/weka/dfive-default/ryanp/rslearn_projects/data/helios/{task}/{cfg}")
 
    print(f"Submitting: {cmd}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {exp_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {exp_id}: {e}")
        return False


def main():
    """Submit jobs."""
    CKPT_PATHS = {
        "0.00004": "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_0.0004/step380250",
        "0.0004": "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0004/step421750",
        "0.004": "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.004/step460250",
        "0.0625": "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.0625/step467750",
        "0.125": "/weka/dfive-default/helios/checkpoints/henryh/latent_mim_cross_only_dec_wc_osm_srtm_dataset_percentage_sweep_.125/step468000"
    }
    
    TASK_CFG_PAIRS = [
        ("v2_landsat_vessels", "finetune_detector_cosinelr.yaml"),
        ("v2_pastis", "basecfg.yaml", "basecfg_helios_mm.yaml"),
        ("v2_nandi_crop_type", "finetune_s1_s2_cosinelr.yaml")
    ]
    
    total_jobs = len(CKPT_PATHS) * len(TASK_CFG_PAIRS)
    print(f"Submitting {total_jobs} jobs...")
    
    successful = 0
    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
    for name, ckpt_path in CKPT_PATHS.items():
        for task, *cfgs in TASK_CFG_PAIRS:
            if submit_job(ckpt_path, task, cfgs, name):
                successful += 1
    
    print(f"\nCompleted: {successful}/{total_jobs} jobs successful")


if __name__ == "__main__":
    main() 
