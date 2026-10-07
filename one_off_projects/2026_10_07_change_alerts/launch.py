"""Launch the training runs of the experiment grid on Beaker.

Each run is one `rslp.main common beaker_train` job on a configs/{source}/{run}.yaml
config written by make_model_configs.py. The Beaker image must include the rslearn
version with rslearn.change_alerts.
"""

import argparse
import json
import shlex
import subprocess  # nosec
from pathlib import Path

from common import SOURCE_DATASETS
from make_model_configs import CONFIG_DIR, RUNS_BY_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
WEKA_MOUNT = {"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}


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
    parser.add_argument(
        "--dry_run", action="store_true", help="Only print the commands"
    )
    args = parser.parse_args()

    clusters = args.clusters.split(",")
    for source in args.sources.split(","):
        for run_name in args.runs.split(","):
            config_path = (CONFIG_DIR / source / f"{run_name}.yaml").resolve()
            if not config_path.exists():
                raise FileNotFoundError(
                    f"{config_path} does not exist, run make_model_configs.py first"
                )
            cmd = [
                "python",
                "-m",
                "rslp.main",
                "common",
                "beaker_train",
                "--image_name",
                args.image_name,
                f"--cluster={json.dumps(clusters)}",
                "--config_path",
                str(config_path.relative_to(REPO_ROOT)),
                f"--weka_mounts+={json.dumps(WEKA_MOUNT)}",
            ]
            print(shlex.join(cmd))
            if not args.dry_run:
                subprocess.check_call(cmd, cwd=REPO_ROOT)  # nosec


if __name__ == "__main__":
    main()
