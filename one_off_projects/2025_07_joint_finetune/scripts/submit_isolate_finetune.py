"""Finetuning on top of label-merged MoE models.

Sort of a hacky way to get around having to splice label-merged models and relearn
roi head weights for detection tasks.

Use a small lr since these models are not changed at all from the previous sft phase,
so it's quite easy to overfit very quickly. No need to do freezeing, probably.

Uses so far
===========
python3 submit_isolate_finetune.py

Experiments to run
==================
1. SFT on task-matched models with no freezing (ie classify->classify__moe)
    - python3 submit_isolate_finetune.py classify__moe_v2 segment__moe_v2 detect__moe_v2

"""

import os
import tempfile
import subprocess
import yaml
import sys

rslp_prefix = "/weka/dfive-default/rslearn-eai"
rslp_project = "2025_08_07_helios_moe_finetune"
experiment_id = "{model}__{task}__isolate"
debug = (rslp_project == "helios-debug")

grouped_tasks = {
    "detect": [
        "detect_sentinel1_vessels",
        "detect_sentinel2_vessels",
        "vessel_detection",
        "detect_satlas_marine_infra",
        "detect_satlas_wind_turbine",
    ],
    "classify": [
        "crop_type_classification",
        "cropland_classification",
        "vessel_classification",
    ],
    "segment": [
        "segment_satlas_solar_farm",
        "segment",
    ],
}

freeze_cfg_path = (
    "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects" \
    "/2025_07_joint_finetune/configs/2025_08_07_freeze/none.yaml"
)

helios_ckpt_path = (
    "/weka/dfive-default/helios/checkpoints" \
    "/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000"
)

for base_model in sys.argv[1:]:
    model_name = base_model.split("__")[0]
    tasks = grouped_tasks[model_name]
    cfg_template_path = (
        "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects" \
        f"/2025_07_joint_finetune/configs/2025_08_06_isolate_sft/{model_name}.yaml"
    )
    ckpt_path = (
        f"/weka/dfive-default/rslearn-eai/projects/" \
        f"helios_finetune_cosine_lr/{base_model}"
    )
    cmd = [
        "python", "-m", "rslp.main", "helios", "launch_finetune",
        "--helios_checkpoint_path", helios_ckpt_path,
        "--patch_size", "8",
        "--encoder_embedding_size", "768",
        "--image_name", "henryh/rslp_multidataset_dev_moe",
        "--cluster+=ai2/titan-cirrascale",
        "--cluster+=ai2/saturn-cirrascale",
        "--cluster+=ai2/ceres-cirrascale",
        "--rslp_project", rslp_project,
        "--ckpt_path", os.path.join(ckpt_path, "checkpoints/last.ckpt"),
    ]

    wd = "/weka/dfive-default/ryanp/rslearn_projects/"
    env = os.environ.copy()
    env["RSLP_PREFIX"] = rslp_prefix

    for task in tasks:
        with tempfile.NamedTemporaryFile(mode="w") as maker_cfg_file:
            with tempfile.NamedTemporaryFile(mode="w") as run_cfg_file:
                with open(cfg_template_path, "r") as f:
                    cfg = yaml.safe_load(f)
                    disabled = [other for other in tasks if other != task]
                    cfg["global_overrides"]["data"] = {
                        "init_args": {
                            "disabled_datasets": disabled
                        }
                    }
                    cfg["output_path"] = run_cfg_file.name
                    yaml.dump(cfg, maker_cfg_file)

                maker_cfg_file.flush()
                print("\nMaking run config from maker config...")

                make_cmd = [
                    "python", "make_multidataset_config.py",
                    "--cfg", maker_cfg_file.name,
                ]
                print(" ".join(make_cmd))
                subprocess.run(make_cmd, check=True, env=env)
                print()

                exp_id = experiment_id.format(model=base_model, task=task)
                print(f"Experiment ID: {rslp_project}/{exp_id}")

                cmd_copy = cmd.copy()
                cmd_copy.append(f"--config_paths+={run_cfg_file.name}")
                cmd_copy.append(f"--config_paths+={freeze_cfg_path}")
                cmd_copy.append(f"--experiment_id={exp_id}")
                if debug:
                    cmd_copy.extend(["--local", "true"])

                print("*" * 30 + " RUN COMMAND " + "*" * 30)
                print(" ".join(cmd_copy))
                print("*" * 80)
                print()

                subprocess.run(cmd_copy, check=True, env=env, cwd=wd)
                print()

                if debug:
                    break
