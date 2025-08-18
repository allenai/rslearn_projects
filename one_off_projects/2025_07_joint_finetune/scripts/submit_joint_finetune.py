import os
import subprocess
import tempfile
import yaml
import itertools


image_name = "henryh/rslp_multidataset_dev"#_0.05w"
project_name =  "2025_07_30_joint_finetune_sweep"#"2025_07_29_helios_joint_finetune_debug"
# project_name = "2025_07_29_helios_joint_finetune_debug"
template = {
    "base_cfg": "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/2025_07_29_debug/base.yaml",
    "substitutions": {
        "patch_size": 8,
        "encoder_embedding_size": 768,
        "helios_checkpoint_path": "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
    },
}
dataset_cfgs = {
    "vessel_detect": "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_landsat_vessels/finetune_detector_cosinelr.yaml",
    "cropland": "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_worldcereal_cropland/finetune_s1_s2_cosinelr.yaml",
    "croptype": "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_nandi_crop_type/finetune_s1_s2_cosinelr.yaml",
    "vessel_classify": "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_landsat_vessels/finetune_classifier_cosinelr.yaml",
    "pastis": [
        "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_pastis/basecfg_cosinelr.yaml",
        "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_pastis/basecfg_helios_mm.yaml",
    ]
}
combos = [
    # adding new tokens - what happens?
    # ["vessel_detect", "cropland"],
    # ["vessel_detect", "cropland", "croptype"],
    # ["vessel_detect", "cropland", "croptype", "vessel_classify"],
    # ["vessel_detect", "cropland", "croptype", "vessel_classify", "pastis"],
    # # pretty sure cropland is biggest, so what if we add in the opposite order?
    # # maybe perf increases with more ood tokens, but decreases with more ood task heads?
    # ["vessel_detect", "pastis"],
    # ["vessel_detect", "pastis", "vessel_classify"],
    # ["vessel_detect", "pastis", "vessel_classify", "croptype"],
]

all_tasks = ["vessel_detect", "cropland", "croptype", "vessel_classify", "pastis"]
all_combos = list(itertools.combinations(all_tasks, 2))
for combo in list(all_combos):
    if list(combo) in combos:
        all_combos.remove(combo)
combos += all_combos

for combo in combos:
    with tempfile.NamedTemporaryFile(mode="w") as maker:
        with tempfile.NamedTemporaryFile(mode="w") as cfg:
            template["dataset_cfgs"] = [dataset_cfgs[cfg] for cfg in combo]
            template["output_path"] = cfg.name
            exp_id = "_".join(combo) + "_norefill" # + "_0.05w_norefill"

            yaml.dump(template, maker)
            maker.flush()

            os.chdir("/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/scripts")
            os.system(f"python make_multidataset_config.py --cfg {maker.name}")

            os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
            cmd = [
                "python", "-m", "rslp.main", "helios", "launch_finetune",
                "--helios_checkpoint_path", "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
                "--patch_size", "8",
                "--encoder_embedding_size", "768",
                "--image_name", image_name,
                "--cluster+=ai2/titan-cirrascale",
                "--cluster+=ai2/saturn-cirrascale", 
                "--cluster+=ai2/ceres-cirrascale",
                "--rslp_project", project_name,
                "--experiment_id", exp_id,
                "--config_paths+=" + cfg.name,
                "--config_paths+=/weka/dfive-default/ryanp/rslearn_projects/data/helios/v2_shared/helios_freeze_then_lowlr.yaml",
            ]
            print(" ".join(cmd))

            env = os.environ.copy()
            env["RSLP_PREFIX"] = "/weka/dfive-default/rslearn-eai"
            try:
                subprocess.run(cmd, check=True, env=env)
                print(f"✅ {exp_id}")
            except subprocess.CalledProcessError as e:
                print(f"❌ {exp_id}: {e}")
            print()
