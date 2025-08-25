import os
import argparse

ckpt_paths = {
    "v1": "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000",
    "v2": "/weka/dfive-default/helios/checkpoints/yawenzzzz/latent_mim_cross_random_per_modality_patchdisc_add_contrastive_0.1_1/step400000",
}


def find_dir(base_dir, postfix_dir):
    for d in os.listdir(base_dir):
        if d.endswith(postfix_dir):
            return d
    raise ValueError(f"Directory {postfix_dir} not found in {base_dir}")

parser = argparse.ArgumentParser()
parser.add_argument("--dirs", type=str, required=True, nargs="+")
parser.add_argument("--tasks", type=str, nargs="*", default=["classify", "segment", "detect", "all"])
parser.add_argument("--base_dir", type=str, default="/weka/dfive-default/ryanp/rslearn_projects/one_off_projects/2025_07_joint_finetune/configs")
parser.add_argument("--project", type=str, default="2025_08_12_task_embeds")
parser.add_argument("--clusters", type=str, nargs="*", default=["saturn", "ceres", "titan"])
parser.add_argument("--ckpt_path", type=str, default=ckpt_paths["v1"])
parser.add_argument("--dry", action="store_true")
args = parser.parse_args()

template = "python3 run.py --cfg {cfg} --exp_id {exp} --gpu 4 --image dev --project {project} --clusters {clusters} --ckpt_path {ckpt_path}"
if args.ckpt_path in ckpt_paths:
    args.ckpt_path = ckpt_paths[args.ckpt_path]

for postfix_dir in args.dirs:
    d = find_dir(args.base_dir, postfix_dir)
    with open(os.path.join(args.base_dir, d, "exp_id.txt"), "r") as f:
        base_exp_id = f.read().strip()
    for f in os.listdir(os.path.join(args.base_dir, d)):
        go = False
        for task in args.tasks:
            if task in f:
                go = True
                break
        if go and f.startswith("OUT") and f.endswith(".yaml"):
            cfg = os.path.join(args.base_dir, d, f)
            exp_id = base_exp_id.format(task=os.path.splitext(f)[0])
            exp_id = exp_id.replace("OUT_", "")
            cmd = template.format(
                cfg=cfg,
                exp=exp_id,
                project=args.project,
                clusters=" ".join(args.clusters),
                ckpt_path=args.ckpt_path,
            )
            print(cmd)
            if not args.dry:
                os.system(cmd)
            print("-" * 100)
            print()
