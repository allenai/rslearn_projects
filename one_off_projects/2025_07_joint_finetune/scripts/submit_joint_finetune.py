import os
import subprocess
import tempfile
import yaml
import itertools


image_name = "henryh/rslp_multidataset_dev"#_0.05w"
project_name = "2025_09_04_loo_evals" #"2025_08_29_finetune_scaling_laws"
template = {
    "base_cfg": "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/2025_09_02_scaling/base.yaml",
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
    ],
    "sentinel1": [
        "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_sentinel1_vessels_128/basecfg_cosinelr.yaml",
        "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_sentinel1_vessels_128/basecfg_helios.yaml"
    ],
    "sentinel2": [
        "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_sentinel2_vessels_128/basecfg_cosinelr.yaml",
        "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_sentinel2_vessels_128/basecfg_helios.yaml"
    ]
}
combos = [
    # adding new tokens - what happens?
    ["cropland", "croptype"],
    ["cropland", "croptype", "pastis"],
    ["cropland", "croptype", "pastis", "vessel_detect"],
    ["cropland", "croptype", "pastis", "vessel_detect", "sentinel1"],
    ["sentinel1", "vessel_detect"],
    ["sentinel1", "vessel_detect", "sentinel2"],
    ["sentinel1", "vessel_detect", "sentinel2", "pastis"],
    ["sentinel1", "vessel_detect", "sentinel2", "pastis", "cropland"],
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

def extra_cfg(length):
    return {"global_overrides": {
            "model": {
                "class_path": "rslearn.train.lightning_module.RslearnLightningModule",
                "init_args": {
                    "model": {
                        "init_args": {
                            "trunk": {
                                "class_path": "rslearn.models.trunk.DecoderTrunk",
                                "init_args": {
                                    "task_embedding": {
                                        "class_path": "rslearn.models.task_embedding.TaskChannelEmbedding",
                                        "init_args": {
                                            "encoder_embedding_size": 768,
                                            "add_spatial_embed": True,
                                        },
                                    },
                                    "layers": [
                                        {
                                            "class_path": "rslp.helios.moe.MoETransformer",
                                            "init_args": {
                                                "dim": 768,
                                                "n_layers": 1,
                                                "n_heads": 12,
                                                "num_experts": 4,
                                                "num_slots": 4,
                                            },
                                        }
                                    ],
                                },
                            }
                        }
                    }
                },
            },
            "trainer": {
                "accumulate_grad_batches": length,
            },
        },
        "merge_options": {
            "merge_heads": True,
            "merge_task_labels": True,
            "same_label_groups": [
                ["detect_sentinel1_vessels", "detect_sentinel2_vessels", "vessel_detection"]
            ],
        },
    }


"""
all_tasks = ["vessel_detect", "cropland", "croptype", "vessel_classify", "pastis"]
all_combos = list(itertools.combinations(all_tasks, 2))
for combo in list(all_combos):
    if list(combo) in combos:
        all_combos.remove(combo)
combos += all_combos
"""

for combo in combos:
    with tempfile.NamedTemporaryFile(mode="w") as maker:
        with tempfile.NamedTemporaryFile(mode="w") as cfg:
            template["dataset_cfgs"] = [dataset_cfgs[cfg] for cfg in combo]
            template["output_path"] = cfg.name
            template.update(extra_cfg(len(combo)))
            exp_id = "_".join(combo)

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
