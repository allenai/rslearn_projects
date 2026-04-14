import argparse
import subprocess
from pathlib import Path

# Script lives on Weka so the job sees it after mount.
DRIVER = "/weka/dfive-default/piperw/rslearn_projects/beaker/install.sh"


def launch(*, name: str, gpus: str = "1") -> None:
    cmd = [
        "gantry", "run",
        "--name", name,
        "--budget", "ai2/es-platform",       # match your org / Beaker budget
        "--workspace", "ai2/earth-systems",  # or your workspace
        "--priority", "urgent",
        "--cluster", "ai2/ceres",
        "--cluster", "ai2/jupiter",
        "--weka", "dfive-default:/weka/dfive-default",
        "--gpus", gpus,
        "--shared-memory", "256GiB",         # tune; rslearn data loaders can be heavy
        "--beaker-image", "ai2/cuda12.8-ubuntu22.04-torch2.6.0",  # base; install.sh replaces torch
        "--gh-token-secret", "PIPERW_GITHUB_TOKEN",  # drop if installs are from local trees only
        "--env-secret", "WANDB_API_KEY=PIPERW_WANDB_TOKEN",  # or RSLEARN_WANDB / whatever you use
        "--no-python",
        "--allow-dirty"
        "--",
        DRIVER,
    ]
    print("[Launching]", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="pw_landslide_finetune")
    p.add_argument("--gpus", default="1")
    args = p.parse_args()
    launch(name=args.name, gpus=args.gpus)
