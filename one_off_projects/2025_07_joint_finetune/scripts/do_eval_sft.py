import os
import argparse

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Path to the checkpoint")
    parser.add_argument("task", type=str, help="Task to evaluate")
    args = parser.parse_args()

    ckpt_cfg_paths = task2cfgs[args.task]
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
        "--do_eval", "true",
        "--allow_missing_weights", "true"
    ]

    cmd.append(f"--ckpt_path={args.ckpt_path}")
    for ckpt_cfg_path in ckpt_cfg_paths:
        cmd.append(f"--config_paths+={ckpt_cfg_path}")

    print(f"Evaluating {args.task} with checkpoint {args.ckpt_path}")
    print()
    print("=" * 80)
    print(" ".join(cmd))
    print("=" * 80)
    print()

    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
    os.system(" ".join(cmd))
