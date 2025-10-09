import argparse
import os
import subprocess
from datetime import datetime

cmd = [
    "python", "-m", "rslp.main", "helios", "launch_finetune",
    "--helios_checkpoint_path", "$CKPT_PATH",
    "--patch_size", "8",
    "--encoder_embedding_size", "768",
    "--image_name", "$IMAGE",
    "--rslp_project", "$PROJECT_NAME",
    "--experiment_id", "$EXP_ID",
]

parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="detect")
parser.add_argument("--ckpt_path", type=str, default="/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000")
parser.add_argument("--exp_id", type=str, default="debug")
parser.add_argument("--project_name", type=str, default="helios-debug")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--image", type=str, default="dev")
parser.add_argument("--clusters", type=str, nargs="*", default=["saturn", "ceres", "titan"])
args = parser.parse_args()

args.image = f"henryh/rslp_multidataset_{args.image}"
if args.project_name == "s":
    args.project_name = "helios_finetune_cosine_lr"

for cluster in args.clusters:
    cmd.append(f"--cluster+=ai2/{cluster}-cirrascale")

if args.gpu == 0:
    RLSP_PREFIX = "/weka/dfive-default/ryanp/rslearn_projects/project_data"
    cmd.extend(["--local", "true"])
else:
    RLSP_PREFIX = "/weka/dfive-default/rslearn-eai"
    cmd.extend(["--gpus", str(args.gpu)])
    if args.project_name == "helios-debug":
        args.project_name = "helios_finetune_cosine_lr"

if args.cfg == "vessel":
    args.cfg = "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_landsat_vessels/finetune_detector_cosinelr.yaml"
    cmd.append("--config_paths+=" + args.cfg)
elif args.cfg == "detect":
    args.cfg = "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/2025_07_31_moe/OUT_detect.yaml"
    cmd.append("--config_paths+=" + args.cfg)
elif args.cfg == "pastis":
    cfgs = [
        '/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_pastis/basecfg_cosinelr.yaml',
        '/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_pastis/basecfg_helios_mm.yaml'
    ]
    for cfg in cfgs:
        cmd.append("--config_paths+=" + cfg)
elif args.cfg == "turbine":
    cfgs = [
        '/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_satlas_wind_turbine_128/basecfg_cosinelr.yaml',
        '/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_satlas_wind_turbine_128/basecfg_helios_mm.yaml',
        '/weka/dfive-default/ryanp/rslearn_projects/data/helios/v2_shared/helios_freeze_then_lowlr.yaml'
    ]
    for cfg in cfgs:
        cmd.append("--config_paths+=" + cfg)
elif args.cfg == "solar":
    cfgs = [
        '/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_satlas_solar_farm_128/basecfg_cosinelr.yaml',
        '/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/v2_satlas_solar_farm_128/basecfg_helios_mm.yaml'
    ]
    for cfg in cfgs:
        cmd.append("--config_paths+=" + cfg)
elif args.cfg == "lora":
    args.cfg = "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/2025_08_12_embeds/OUT_classify_plain.yaml"
    args.exp_id = "debug_task_lora_classify"
    cmd.append("--config_paths+=" + args.cfg)
elif args.cfg == "moe":
    args.cfg = "/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs/2025_08_15_helios_moe/OUT_classify.yaml"
    args.exp_id = "debug_task_moe_base"
    cmd.append("--config_paths+=" + args.cfg)
else:
    cmd.append("--config_paths+=" + args.cfg)


if args.exp_id == "debug":
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.exp_id = f"debug_{now}"

print(args)

env = os.environ.copy()
env["RSLP_PREFIX"] = RLSP_PREFIX

run_cmd = " ".join(cmd)
run_cmd = run_cmd.replace("$CKPT_PATH", args.ckpt_path)
run_cmd = run_cmd.replace("$PROJECT_NAME", args.project_name)
run_cmd = run_cmd.replace("$EXP_ID", args.exp_id)
run_cmd = run_cmd.replace("$IMAGE", args.image)

print(f"Running: {run_cmd}")
os.chdir("/weka/dfive-default/ryanp/rslearn_projects/")
subprocess.run(run_cmd, shell=True, env=env)
