"""Finetuning on top of label-merged MoE models.

Sort of a hacky way to get around having to splice label-merged models and relearn
roi head weights for detection tasks.

Use a small lr since these models are not changed at all from the previous sft phase,
so it's quite easy to overfit very quickly. No need to do freezing, probably.

Uses so far
===========
python3 submit_isolate_finetune.py

Experiments to run
==================
1. SFT on task-matched models with no freezing (ie classify->classify__moe) [OUTDATED]
    - python3 submit_isolate_finetune.py --models classify__moe_v2 segment__moe_v2 detect__moe_v2
2. LOO experiments (for final presentation)
    - python3 submit_isolate_finetune.py --loo --models no_landsat no_s1 no_s2 no_turbine no_marine
    - python3 submit_isolate_finetune.py --loo --models no_marine --percents 0.01 0.1 0.5
3. OOD experiments (for final presentation)
    - python3 submit_isolate_finetune.py --models segment_v3 detect --ckpt_project 2025_08_29_finetune_benchmarks
    - python3 submit_isolate_finetune.py --models segment_v3 detect --ckpt_project 2025_08_29_finetune_benchmarks --tasks crop_type_classification --percents 0.01 0.1 0.5 1.0
"""

import os
import tempfile
import subprocess
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="*", help="Models to finetune")
parser.add_argument("--ckpt_project", default="2025_09_04_loo", help="Project directory with checkpoints")
parser.add_argument("--project", default="2025_09_08_loo_evals", help="Output project directory")
parser.add_argument("--exp_id", default="{model}__{task}__{percent}__v3", help="Experiment ID template")
parser.add_argument("--loo", action="store_true", help="Run LOO experiments")
parser.add_argument("--debug", action="store_true", help="Run in debug mode")
parser.add_argument("--percents", type=float, nargs="*", default=[1.0], help="Percentages of dataset to sweep over")
parser.add_argument("--ckpt", default="last.ckpt", help="Checkpoint name to finetune from")
parser.add_argument("--tasks", nargs="*", help="Override the predefined tasks and finetune on these for all models")
parser.add_argument(
    "--template_cfg_path",
    default="/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/2025_09_04_loo/template.yaml",
    help="Template config path"
)
parser.add_argument(
    "--freeze_cfg_path",
    default="/weka/dfive-default/ryanp/rslearn_projects/data/helios/v2_shared/helios_freeze_then_lowlr.yaml",
    help="Freeze config path"
)
parser.add_argument(
    "--helios_ckpt_path",
    default="/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
    help="Helios checkpoint path"
)
args = parser.parse_args()

grouped_tasks = {
    "detect": ["crop_type_classification", "cropland_classification", "segment"],
    "classify": ["vessel_detection", "segment", "detect_satlas_marine_infra"],
    "segment": ["vessel_classification", "detect_satlas_wind_turbine", "cropland_classification", "crop_type_classification"],
}
# grouped_tasks = {
#     "detect": [
#         "detect_sentinel1_vessels",
#         "detect_sentinel2_vessels",
#         "vessel_detection",
#         "detect_satlas_marine_infra",
#         "detect_satlas_wind_turbine",
#     ],
#     "classify": [
#         "crop_type_classification",
#         "cropland_classification",
#         "vessel_classification",
#     ],
#     "segment": [
#         "segment_satlas_solar_farm",
#         "segment",
#     ],
# }

task2ckpt = {
    "vessel_detection": "v2_landsat_vessels",
    "detect_sentinel1_vessels": "v2_sentinel1_vessels_128",
    "detect_sentinel2_vessels": "v2_sentinel2_vessels_128",
    "detect_satlas_wind_turbine": "v2_satlas_wind_turbine_128",
    "detect_satlas_marine_infra": "v2_satlas_marine_infra_128",
    "crop_type_classification": "v2_nandi_crop_type",
    "cropland_classification": "v2_worldcereal_cropland",
    "vessel_classification": "v2_landsat_vessels",
    "segment_satlas_solar_farm": "v2_satlas_solar_farm_128",
    "segment": "v2_pastis",
}
dir2cfg = {
    "vessel_detection": ["finetune_detector_cosinelr.yaml"],
    "vessel_classification": ["finetune_classifier_cosinelr.yaml"],
}

loo_tasks = {
    "no_landsat": ["vessel_detection"],
    "no_s1": ["detect_sentinel1_vessels"],
    "no_s2": ["detect_sentinel2_vessels"],
    "no_turbine": ["detect_satlas_wind_turbine"],
    "no_marine": ["detect_satlas_marine_infra"],
}

max_epochs = {
    "vessel_detection": 100,
    "vessel_classification": 100,
    "segment_satlas_solar_farm": 500,
    "segment": 500,
    "detect_sentinel1_vessels": 200,
    "detect_sentinel2_vessels": 100,
    "detect_satlas_wind_turbine": 100,
    "detect_satlas_marine_infra": 100,
    "crop_type_classification": 100,
    "cropland_classification": 100,
}

ckpt_scratch_dir = "/weka/dfive-default/ryanp/scratch/__tmp_ckpts"
base_cfg_dir = "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs"
base_project_dir = os.path.join(
    "/weka/dfive-default/rslearn-eai/projects", 
    args.ckpt_project
)

if args.debug:
    args.project = "helios-debug"

for base_model in args.models:
    if args.tasks:
        tasks = args.tasks
    elif args.loo:
        tasks = loo_tasks[base_model]
    else:
        tasks = grouped_tasks[base_model.split("_")[0]]
    ckpt_path = os.path.join(base_project_dir, base_model)
    cfg_path = os.path.join(ckpt_path, "checkpoints/config.yaml")
    cmd = [
        "python", "-m", "rslp.main", "helios", "launch_finetune",
        "--helios_checkpoint_path", args.helios_ckpt_path,
        "--patch_size", "8",
        "--encoder_embedding_size", "768",
        "--image_name", "henryh/rslp_multidataset_dev",
        "--cluster+=ai2/titan-cirrascale",
        "--cluster+=ai2/saturn-cirrascale",
        "--cluster+=ai2/ceres-cirrascale",
        "--rslp_project", args.project,
    ]

    wd = "/weka/dfive-default/ryanp/rslearn_projects/"
    env = os.environ.copy()
    env["RSLP_PREFIX"] = "/weka/dfive-default/rslearn-eai"

    for percent in args.percents:
        for task in tasks:
            with tempfile.NamedTemporaryFile(mode="w") as run_cfg_file:
                with tempfile.NamedTemporaryFile(mode="w") as maker_cfg_file:
                    with open(args.template_cfg_path, "r") as f:
                        template = yaml.safe_load(f)

                        template["output_path"] = run_cfg_file.name
                        all_cfg_files = sorted(os.listdir(os.path.join(base_cfg_dir, task2ckpt[task])))
                        template["dataset_cfgs"] = [[
                            os.path.join(base_cfg_dir, task2ckpt[task], cfg)
                            for cfg in dir2cfg.get(task, all_cfg_files)
                            if cfg.endswith(".yaml")
                        ]]
                        restore_cfg = template["global_overrides"]["model"]["init_args"]["restore_config"]["init_args"]
                        restore_cfg["restore_path"] = os.path.join(ckpt_path, "checkpoints", args.ckpt)

                        trainer = template["global_overrides"].get("trainer", {})
                        trainer['limit_train_batches'] = percent
                        trainer['max_epochs'] = max_epochs[task]
                        if 'max_epochs' in trainer:
                            original_epochs = trainer['max_epochs']
                            adjusted_epochs = int(original_epochs / percent)
                            trainer['max_epochs'] = adjusted_epochs
                            print(f"adjusted epochs: {original_epochs} -> {adjusted_epochs} (factor: {1/percent:.2f})")

                        yaml.dump(template, maker_cfg_file)
                
                    maker_cfg_file.flush()
                    print("\nMaking run config from maker config...")

                    make_cmd = [
                        "python", "make_multidataset_config.py",
                        "--cfg", maker_cfg_file.name,
                    ]
                    print(" ".join(make_cmd))
                    subprocess.run(make_cmd, check=True, env=env)
                    print()

                    exp_id = args.exp_id.format(model=base_model, task=task, percent=percent)
                    print(f"Experiment ID: {args.project}/{exp_id}")

                    cmd_copy = cmd.copy()
                    cmd_copy.append(f"--config_paths+={run_cfg_file.name}")
                    cmd_copy.append(f"--experiment_id={exp_id}")
                    if args.debug:
                        cmd_copy.extend(["--local", "true"])

                    print("*" * 30 + " RUN COMMAND " + "*" * 30)
                    print(" ".join(cmd_copy))
                    print("*" * 80)
                    print()

                    subprocess.run(cmd_copy, check=True, env=env, cwd=wd)
                    print()

                    if args.debug:
                        break
