#!/usr/bin/env python3
"""
Submit finetuning jobs.
"""

import subprocess
import os

RUN = True
DEBUG = False
#PROJECT_NAME = "helios-debug" if DEBUG else "2025_07_29_helios_finetune" #"2025_07_29_helios_joint_finetune_debug"
PROJECT_NAME = "helios_finetune_cosinelr"#"2025_07_30_joint_finetune_sweep"
CKPT_PATH = "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000"
IMAGE_NAME = "henryh/rslp_multidataset_dev_0.05w" if "joint_finetune_debug" in PROJECT_NAME else "henryh/rslp_multidataset_dev"


def submit_job(task_dir: str, task_name: str, cfgs: list[str]) -> bool:
    """Submit a single helios finetune job."""
    exp_id = task_name + "_0.05w" if "joint_finetune_debug" in PROJECT_NAME else task_name + "__fewbatcheval"
    cmd = [
        "python", "-m", "rslp.main", "helios", "launch_finetune",
        "--helios_checkpoint_path", CKPT_PATH,
        "--patch_size", "8",
        "--encoder_embedding_size", "768",
        "--image_name", IMAGE_NAME,
        "--cluster+=ai2/titan-cirrascale",
        "--cluster+=ai2/saturn-cirrascale", 
        "--cluster+=ai2/ceres-cirrascale",
        "--rslp_project", PROJECT_NAME,
        "--experiment_id", exp_id,
    ]
    for cfg in cfgs:
        print(f"Adding config: {cfg}")
        if cfg == "few_batch_eval.yaml":
            cfg_task_dir = "v3_multitask"
        else:
            cfg_task_dir = task_dir
        cmd.append(f"--config_paths+=/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/{cfg_task_dir}/{cfg}")

    cmd.append("--config_paths+=/weka/dfive-default/ryanp/rslearn_projects/data/helios/v2_shared/helios_freeze_then_lowlr.yaml")

    print()
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
    TASK_CFG_PAIRS = [
        # ("v2_pastis", "pastis", "basecfg_cosinelr.yaml", "basecfg_helios_mm.yaml"),
        # ("v2_nandi_crop_type", "nandi_crop_type", "finetune_s1_s2_cosinelr.yaml"),
        # ("v2_worldcereal_cropland", "worldcereal_cropland", "finetune_s1_s2_cosinelr.yaml"),
        # "v2_landsat_vessels", "landsat_vessel_classify", "finetune_classifier_cosinelr.yaml"),
        ("v2_landsat_vessels", "landsat_vessel_detect", "finetune_detector_cosinelr.yaml", "few_batch_eval.yaml"),
        # ("v2_lfmc", "lfmc", "finetune_s1_s2_srtm_cosinelr.yaml"),
        ("v2_satlas_marine_infra_128", "marine_infra", "basecfg_cosinelr.yaml", "basecfg_helios_mm.yaml", "few_batch_eval.yaml"),
        ("v2_satlas_wind_turbine_128", "wind_turbine", "basecfg_cosinelr.yaml", "basecfg_helios_mm.yaml", "few_batch_eval.yaml"),
        ("v2_sentinel1_vessels_128", "vessel_sentinel1", "basecfg_cosinelr.yaml", "basecfg_helios.yaml", "few_batch_eval.yaml"),
        ("v2_sentinel2_vessels_128", "vessel_sentinel2", "basecfg_cosinelr.yaml", "basecfg_helios.yaml", "few_batch_eval.yaml"),
        # ("v2_satlas_solar_farm_128", "solar_farm", "basecfg_cosinelr.yaml", "basecfg_helios.yaml"),
    ]

    print(f"Submitting {len(TASK_CFG_PAIRS)} jobs...")
    success = 0
    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
    for task_dir, task_name, *cfgs in TASK_CFG_PAIRS:
        success += int(submit_job(task_dir, task_name, cfgs))
    print(f"\nCompleted: {success}/{len(TASK_CFG_PAIRS)} jobs successful")


if __name__ == "__main__":
    main() 
