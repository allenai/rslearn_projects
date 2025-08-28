import os
import json
import argparse

base_dir = "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs"
all_cfgs = {}
for task in os.listdir(base_dir):
    if task.startswith("v2_"):
        all_cfgs[task] = [
            os.path.join(base_dir, task, cfg)
            for cfg in os.listdir(os.path.join(base_dir, task))
            if cfg.endswith(".yaml")
        ]

all_cfgs["vessel_classification"] = [
    os.path.join(base_dir, "v2_landsat_vessels", "finetune_classifier_cosinelr.yaml"),
]
all_cfgs["vessel_detection"] = [
    os.path.join(base_dir, "v2_landsat_vessels", "finetune_detector_cosinelr.yaml"),
]
print("available tasks:")
print(json.dumps(all_cfgs, indent=2))
print()

parser = argparse.ArgumentParser()
parser.add_argument("ckpt_path", type=str, help="Path to the checkpoint")
parser.add_argument("task", type=str, help="Task to evaluate")
parser.add_argument("old_or_new", type=str, help="old or new helios checkpoint")
args = parser.parse_args()

ckpt_cfg_paths = all_cfgs[args.task]
if args.old_or_new.lower() == "old":
    helios_path = "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000"
else:
    helios_path = "/weka/dfive-default/helios/checkpoints/yawenzzzz/latent_mim_cross_random_per_modality_patchdisc_add_contrastive_0.1_1/step400000"
print("using helios path", helios_path)
print()
cmd = [
    "RSLP_PREFIX=/weka/dfive-default/rslearn-eai",
    "python", "-m", "rslp.main", "helios", "launch_finetune",
    "--helios_checkpoint_path", helios_path,
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
