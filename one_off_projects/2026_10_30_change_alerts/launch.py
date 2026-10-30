"""Launch the training runs on Beaker.

Each run is one rslp.common.beaker_train job on a configs/{source}/{run}.yaml config
written by make_model_configs.py. The Beaker image must include the rslearn version
with rslearn.change_alerts.layers.
"""

import argparse
import os
from pathlib import Path

from common import SOURCE_DATASETS
from make_model_configs import CONFIG_DIR, RUNS_BY_NAME

from rslp.common.beaker_train import beaker_train
from rslp.utils.beaker import WekaMount

REPO_ROOT = Path(__file__).resolve().parents[2]
WEKA_MOUNT = WekaMount(bucket_name="dfive-default", mount_path="/weka/dfive-default")


def main() -> None:
    """Launch the jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_name", required=True, help="Beaker image name")
    parser.add_argument(
        "--clusters",
        default="ai2/jupiter,ai2/ceres",
        help="Comma-separated Beaker clusters",
    )
    parser.add_argument("--sources", default=",".join(SOURCE_DATASETS))
    parser.add_argument("--runs", default=",".join(RUNS_BY_NAME))
    parser.add_argument("--priority", default="high")
    parser.add_argument(
        "--dry_run", action="store_true", help="Only print the configs to launch"
    )
    args = parser.parse_args()

    # beaker_train uploads the current directory and takes config paths relative to it.
    os.chdir(REPO_ROOT)
    for source in args.sources.split(","):
        for run_name in args.runs.split(","):
            config_path = (CONFIG_DIR / source / f"{run_name}.yaml").resolve()
            if not config_path.exists():
                raise FileNotFoundError(
                    f"{config_path} does not exist, run make_model_configs.py first"
                )
            rel_path = str(config_path.relative_to(REPO_ROOT))
            print(f"launching {rel_path}", flush=True)
            if args.dry_run:
                continue
            beaker_train(
                image_name=args.image_name,
                cluster=args.clusters.split(","),
                config_path=rel_path,
                weka_mounts=[WEKA_MOUNT],
                priority=args.priority,
            )


if __name__ == "__main__":
    main()
