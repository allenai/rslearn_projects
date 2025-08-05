import os
import subprocess
import argparse

ckpt_path = "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000"
cmd = [
    "python", "-m", "rslp.main", "helios", "launch_finetune",
    "--helios_checkpoint_path", ckpt_path,
    "--patch_size", "8",
    "--encoder_embedding_size", "768",
    "--image_name", "henryh/rslp_multidataset_dev",
    "--cluster+=ai2/titan-cirrascale",
    "--cluster+=ai2/saturn-cirrascale", 
    "--cluster+=ai2/ceres-cirrascale",
    "--gpus", "4"
]

parser = argparse.ArgumentParser()
parser.add_argument("--jobs", type=str, required=True, nargs="+")
parser.add_argument("--tag", type=str, required=True)
parser.add_argument("--project", type=str, default="helios_finetune_cosine_lr")
args = parser.parse_args()

print("="*100)
print(f"Using project: {args.project}")
print(f"Using checkpoint: {ckpt_path}")
print(f"Submitting jobs: {args.jobs}")
print("="*100)
print()

os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
for job in args.jobs:
    run_cmd = cmd.copy()
    run_cmd.extend(["--rslp_project", args.project])
    run_cmd.extend(["--experiment_id", job + "__" + args.tag])
    run_cmd.append(f"--config_paths+=/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/2025_07_31_moe/OUT_{job}.yaml")
    print("-"*100)
    print(f"Running: {' '.join(run_cmd)}")
    env = os.environ.copy()
    env["RSLP_PREFIX"] = "/weka/dfive-default/rslearn-eai"
    subprocess.run(run_cmd, env=env, check=True)
    print("-"*100)
    print()
