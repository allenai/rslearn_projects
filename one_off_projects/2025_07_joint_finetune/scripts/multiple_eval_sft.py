import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--version", type=str, required=True, help="options are [v1, v2, cosine]")
parser.add_argument("--full", action="store_true", help="Run eval on all patches")
parser.add_argument("--save_dir", type=str, help="Directory to save eval results", default="/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/data/evals")
args = parser.parse_args()

if args.version in ["v1", "v2"]:
    ckpt_dir = "/weka/dfive-default/rslearn-eai/projects/2025_08_27_helios_cmp/"
else:
    ckpt_dir = "/weka/dfive-default/rslearn-eai/projects/helios_finetune_cosine_lr"
ckpt = "checkpoints/last.ckpt"
task_to_directory = {
    "v2_nandi_crop_type": "nandi_crop_type",
    "v2_worldcereal_cropland": "worldcereal_cropland",
    "v2_satlas_marine_infra_128": "marine_infra",
    "v2_satlas_wind_turbine_128": "wind_turbine",
    "v2_sentinel1_vessels_128": "vessel_sentinel1",
    "v2_sentinel2_vessels_128": "vessel_sentinel2",
    "v2_pastis": "pastis",
    "v2_satlas_solar_farm_128": "solar_farm",
    "vessel_classification": "landsat_vessel_classify",
    "vessel_detection": "landsat_vessel_detect"
}
tasks = {v: k for k, v in task_to_directory.items()}

for d in os.listdir(ckpt_dir):
    if d.endswith("_v1") and args.version == "v1":
        old_or_new = "old"
    elif d.endswith("_v2") and args.version == "v2":
        old_or_new = "new"
    elif args.version == "cosine":
        old_or_new = "old"

    ckpt_path = os.path.join(ckpt_dir, d, ckpt)
    try:
        task = tasks[d.replace("_" + args.version, "")]
    except KeyError:
        continue
    cmd = f"python do_eval_sft.py {ckpt_path} {task} {old_or_new}"
    if args.full:
        cmd += " --full"
    if args.save_dir:
        cmd += f" --save_eval_path {os.path.join(args.save_dir, d + '.json')}"
    print(cmd)
    os.system(cmd)
    print()
