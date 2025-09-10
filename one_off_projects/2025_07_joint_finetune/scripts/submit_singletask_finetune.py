"""Submit finetuning jobs on top of label-unmerged single task MoE models.

Freezing options
================
standard: 20 epochs, 10x low lr
- use if just finetuning random-init decoder, chopping off moe

decoder: 10 epochs, 5x low lr
- use if finetuning on top of moe + sft1 init decoder
- train only decoder+moe for 10, then unfreeze encoder

decoder_moe: 5 epochs, 2x low lr -> 5 epochs, 5x low lr
- use if finetuning on top of moe + sft1 init decoder
- train only decoder for 5, then unfreeze moe for 5, then full finetune
- BROKEN SINCE TORCH DON'T SUPPORT MULTIPLE CALLBACKS OF THE SAME TYPE

none: small lr (2e-5)
- use if training non-spliced moe + sft1 (no need to relearn roi head)
- already pretty overfit so don't freeze
- use this in submit_isolate_finetune.py

Uses so far
===========
python3 submit_singletask_finetune.py --encoder_only --freezer standard
python3 submit_singletask_finetune.py --freezer decoder_moe
python3 submit_singletask_finetune.py --freezer decoder

Experiments to run
==================
1. SFT (freezer decoder) on all tasks based on detect model
    python3 submit_singletask_finetune.py --models detect__moe_v2 --freezer decoder

2. Freezing strategy ablations across classification tasks
    python3 submit_singletask_finetune.py \
        --encoder_only \
        --freezer standard \
        --models detect__moe_v2 \
        --types classify

3. Dataset ablation experiments
    python3 submit_singletask_finetune.py \
        --dataset_percents 0.01 0.1 0.2 0.5 1.0 \
        --freezer decoder \
        --models detect__moe_v2 \
        --tasks crop_type_classification cropland_classification \
        --project 2025_08_07_helios_moe_finetune_dataset_ablation

"""

import torch
import subprocess
import os
import tempfile
import json
import yaml
import argparse
from typing import Any


def submit_job(
    ckpt_path: str | None,
    entry: dict[str, Any],
    model_name: str,
    dataset_percent: float,
    args: argparse.Namespace,
) -> bool:
    """Submit job.
    
    Args:
        ckpt_path: Path to checkpoint (none = use base)
        entry: Task configuration
        model_name: Model name
        dataset_percent: Percentage of dataset to use
        args: Command line arguments
    
    Returns:
        Boolean indicating success
    """
    task_dir = entry["task_dir"]
    task_name = entry["task_name"]
    cfgs = entry["cfgs"]
    freezer = args.freezer
    encoder_only = args.encoder_only
    debug = args.debug
    project = args.project

    exp_id = model_name + "__" + task_name + "__" + f"freezer_{freezer}"
    if encoder_only:
        exp_id += "__encoder_only"
    if debug:
        exp_id += "__debug"
    if dataset_percent != 1.0 or "ablation" in project:
        exp_id += f"__{dataset_percent}"

    cmd = [
        "python", "-m", "rslp.main", "helios", "launch_finetune",
        "--helios_checkpoint_path", args.helios,
        "--patch_size", "8",
        "--encoder_embedding_size", "768",
        "--image_name", "henryh/rslp_multidataset_dev_moe",
        "--cluster+=ai2/titan-cirrascale",
        "--cluster+=ai2/saturn-cirrascale", 
        "--cluster+=ai2/ceres-cirrascale",
        "--rslp_project", "helios-debug" if debug else project,
        "--experiment_id", exp_id,
    ]

    num_epochs = None
    for cfg in cfgs:
        cfg_path = (
            f"/weka/dfive-default/ryanp/rslearn_projects/one_off_projects" \
            f"/2025_07_joint_finetune/configs/{task_dir}/{cfg}"
        )
        cmd.append(f"--config_paths+={cfg_path}")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
            try:
                if num_epochs is None:
                    num_epochs = cfg["trainer"]["max_epochs"]
                assert num_epochs == cfg["trainer"]["max_epochs"]
            except KeyError:
                pass
    if num_epochs is None:
        raise ValueError("No num_epochs found in any config")

    cmd.append(
        f"--config_paths+=/weka/dfive-default/ryanp/rslearn_projects/one_off_projects" \
        f"/2025_07_joint_finetune/configs/2025_08_07_freeze/{freezer}.yaml"
    )

    if debug:
        cmd.append("--local")
        cmd.append("true")

    with tempfile.NamedTemporaryFile(mode="w") as f:
        if ckpt_path is not None:
            # Support yaml conversion of old format with trunk
            with open(os.path.join(ckpt_path, "checkpoints", "config.yaml")) as cf:
                old_cfg = yaml.safe_load(cf)["model"]["init_args"]["model"]["init_args"]["trunk"]
                trunk_layer_init_args = old_cfg.pop("init_args")
                trunk_layer_init_args["disable_moe"] = not trunk_layer_init_args.pop("use_moe")
                old_cfg["init_args"] = {
                    "task_embedding": trunk_layer_init_args.pop("task_embedding"),
                    "layers": [
                        {
                            "class_path": "rslp.helios.moe.MoETransformer",
                            "init_args": trunk_layer_init_args
                        }
                    ]
                }
                override_cfg = {
                    "model": {
                        "init_args": {
                            "model": {
                                "class_path": "rslearn.models.multitask.MultiTaskModel",
                                "init_args": {
                                    "trunk": old_cfg
                                }
                            }
                        }
                    }
                }

            # Might have to change keys in state_dict slightly
            # Don't support really old format without trunk
            ckpt = torch.load(os.path.join(ckpt_path, "checkpoints", "last.ckpt"))
            sd = ckpt["state_dict"]
            for k, v in list(sd.items()):
                if encoder_only:
                    if "model.encoder" not in k:
                        del sd[k]
                else:
                    new_k = k.replace("model.trunk.transformer", "model.trunk.layers.0")
                    sd[new_k] = v
                    if new_k != k:
                        del sd[k]

            # Use a persistent "temporary" checkpoint dir so that when we upload to beaker,
            # we can still access the checkpoint as it isn't loaded immediately
            ckpt_f = os.path.join(args.tmp_ckpt_dir, os.path.basename(ckpt_path))
            override_cfg["model"]["init_args"]["restore_config"] = {
                "restore_path": ckpt_f,
                "selector": ["state_dict"],
                "remap_prefixes": [["model.", ""]]
            }
            os.makedirs(args.tmp_ckpt_dir, exist_ok=True)
            torch.save(ckpt, ckpt_f)
            print(f"saved temporary checkpoint to {ckpt_f}")
        
        else:
            print("Not loading checkpoint, using base model")
            override_cfg = {}

        # Add dataset percent if it's requested
        if dataset_percent != 1.0:
            if "trainer" not in override_cfg:
                override_cfg["trainer"] = {}
            override_cfg["trainer"]["limit_train_batches"] = dataset_percent
            override_cfg["trainer"]["max_epochs"] = int(num_epochs / dataset_percent)
            print(
                f"Adjusted epochs: {num_epochs} -> {override_cfg['trainer']['max_epochs']} " \
                f"(factor: {1/dataset_percent:.2f})"
            )

        # Finish override config and start the job
        print(f"=" * 30 + " OVERRIDE_CONFIG " + "=" * 30)
        print(json.dumps(override_cfg, indent=2))
        print("=" * 70)
        print()
        yaml.dump(override_cfg, f)
        f.flush()

        if encoder_only:
            cmd.append("--allow_missing_weights")
            cmd.append("true")
            assert dataset_percent == 1.0, "cannot do encoder_only and dataset ablation"
        else:
            cmd.append(f"--config_paths+={f.name}")

        print("=" * 70)
        print(" ".join(cmd))
        print("=" * 70)

        env = os.environ.copy()
        env["RSLP_PREFIX"] = "/weka/dfive-default/rslearn-eai"
        try:
            subprocess.run(cmd, check=True, env=env)
            print(f"✅ {exp_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {exp_id}: {e}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="Models to finetune")
    parser.add_argument("--tasks", nargs="*", help="Tasks to finetune on")
    parser.add_argument("--types", nargs="*", help="Types of tasks to finetune on")
    parser.add_argument("--match_type", action="store_true", help="Match types of tasks to finetune on")
    parser.add_argument("--encoder_only", action="store_true", help="Don't load decoder/MoE weights")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--freezer", default="decoder", help="Freezing strategy")
    parser.add_argument("--dataset_percents", type=float, nargs="*", default=[1.0], help="Percentage of dataset to sweep over")
    parser.add_argument("--ckpt_project_dir", default="helios_finetune_cosine_lr", help="Project directory")
    parser.add_argument("--project", default="2025_08_07_helios_moe_finetune", help="Wandb project")
    parser.add_argument("--tmp_ckpt_dir", default="/weka/dfive-default/ryanp/cache/helios_ckpts", help="Temporary checkpoint directory")
    parser.add_argument("--helios", default="/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000")
    args = parser.parse_args()

    models = [
        "detect__moe_v2",
        "classify__moe_v2",
        "segment__moe_v2",
        "base"
    ]

    ckpt_paths = {model: [] for model in models}
    ckpt_paths["base"] = None
    base_path = os.path.join("/weka/dfive-default/rslearn-eai/projects", args.ckpt_project_dir)
    for model_dir in os.listdir(base_path):
        for model in models:
            if model_dir.startswith(model + "__unmerged__"):
                ckpt_paths[model].append(os.path.join(base_path, model_dir))
    
    print("available models:")
    print(json.dumps(ckpt_paths, indent=2))
    print()

    task_cfg_pairs = [
        {"task_dir": "v2_satlas_solar_farm_128", "task_name": "segment_satlas_solar_farm", "cfgs": ["basecfg_cosinelr.yaml", "basecfg_helios_mm.yaml"], "type": "segment"},
        {"task_dir": "v2_pastis", "task_name": "segment", "cfgs": ["basecfg_cosinelr.yaml", "basecfg_helios_mm.yaml"], "type": "segment"},
        {"task_dir": "v2_landsat_vessels", "task_name": "vessel_classification", "cfgs": ["finetune_classifier_cosinelr.yaml"], "type": "classify"},
        {"task_dir": "v2_nandi_crop_type", "task_name": "crop_type_classification", "cfgs": ["finetune_s1_s2_cosinelr.yaml"], "type": "classify"},
        {"task_dir": "v2_worldcereal_cropland", "task_name": "cropland_classification", "cfgs": ["finetune_s1_s2_cosinelr.yaml"], "type": "classify"},
        {"task_dir": "v2_landsat_vessels", "task_name": "vessel_detection", "cfgs": ["finetune_detector_cosinelr.yaml"], "type": "detect"},
        {"task_dir": "v2_satlas_wind_turbine_128", "task_name": "detect_satlas_wind_turbine", "cfgs": ["basecfg_cosinelr.yaml", "basecfg_helios_mm.yaml"], "type": "detect"},
        {"task_dir": "v2_satlas_marine_infra_128", "task_name": "detect_satlas_marine_infra", "cfgs": ["basecfg_cosinelr.yaml", "basecfg_helios_mm.yaml"], "type": "detect"},
        {"task_dir": "v2_sentinel1_vessels_128", "task_name": "detect_sentinel1_vessels", "cfgs": ["basecfg_cosinelr.yaml", "basecfg_helios.yaml"], "type": "detect"},
        {"task_dir": "v2_sentinel2_vessels_128", "task_name": "detect_sentinel2_vessels", "cfgs": ["basecfg_cosinelr.yaml", "basecfg_helios.yaml"], "type": "detect"},
    ]
    print("available tasks:")
    print(json.dumps(task_cfg_pairs, indent=2))
    print()

    models = {}
    for model_name, ckpt_paths in ckpt_paths.items():
        if args.models and model_name not in args.models:
            continue
        models[model_name] = ckpt_paths

    tasks = []
    for entry in task_cfg_pairs:
        if args.tasks and entry["task_name"] not in args.tasks:
            continue
        if args.types and entry["type"] not in args.types:
            continue
        tasks.append(entry)
    
    if args.match_type:
        tasks = [entry for entry in tasks if entry["type"] in args.types]

    num_models = len(models)
    num_tasks = len(tasks)
    print("================")
    print("MODELS:", num_models)
    print("TASKS:", num_tasks)
    print("DATASET PERCENTS:", args.dataset_percents)
    print("TOTAL JOBS:", num_models * num_tasks * len(args.dataset_percents))
    print("================")
    print()

    success = 0
    os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
    for dataset_percent in args.dataset_percents:
        for model_name, ckpt_paths in models.items():
            for entry in tasks:
                if ckpt_paths is None:
                    # we are using base model
                    success += int(submit_job(None, entry, model_name, dataset_percent, args))
                else:
                    for ckpt_path in ckpt_paths:
                        if f"{model_name}__unmerged__{entry['task_name']}" in ckpt_path:
                            success += int(submit_job(ckpt_path, entry, model_name, dataset_percent, args))
                            break
                    else:
                        success += int(submit_job(ckpt_paths[0], entry, model_name, dataset_percent, args))
                if args.debug:
                    exit()

    print(f"\nCompleted: {success} jobs successful")
