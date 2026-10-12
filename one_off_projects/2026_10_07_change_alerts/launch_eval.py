"""Launch eval_test.py on Beaker, one job per source dataset and run.

The current code is uploaded like rslp.common.beaker_train does, and each job writes
its metrics to {out_dir}/{source}_{run}.csv. The Beaker image must include the
rslearn version with rslearn.change_alerts.
"""

import argparse
import os
import shlex
import uuid
from pathlib import Path

from beaker import (
    Beaker,
    BeakerConstraints,
    BeakerExperimentSpec,
    BeakerTaskResources,
    BeakerTaskSpec,
)
from common import SOURCE_DATASETS
from make_model_configs import RUNS_BY_NAME

from rslp import launcher_lib
from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_WORKSPACE,
    WekaMount,
    create_gcp_credentials_mount,
    get_base_env_vars,
)

PROJECT_ID = "2026_10_07_change_alerts"
CODE_EXPERIMENT_ID = "eval_code"
REPO_ROOT = Path(__file__).resolve().parents[2]
WEKA_MOUNT = WekaMount(bucket_name="dfive-default", mount_path="/weka/dfive-default")


def main() -> None:
    """Launch the jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_name", required=True, help="Beaker image name")
    parser.add_argument("--out_dir", required=True, help="Directory for the CSVs")
    parser.add_argument(
        "--clusters",
        default="ai2/jupiter,ai2/ceres",
        help="Comma-separated Beaker clusters",
    )
    parser.add_argument("--sources", default=",".join(SOURCE_DATASETS))
    parser.add_argument("--runs", default=",".join(RUNS_BY_NAME))
    parser.add_argument("--priority", default="high")
    args = parser.parse_args()

    # upload_code archives the current directory.
    os.chdir(REPO_ROOT)
    launcher_lib.upload_code(PROJECT_ID, CODE_EXPERIMENT_ID)

    download = (
        "python -c 'from rslp.launcher_lib import download_code; "
        f'download_code("{PROJECT_ID}", "{CODE_EXPERIMENT_ID}")\''
    )
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        for source in args.sources.split(","):
            for run_name in args.runs.split(","):
                out = f"{args.out_dir}/{source}_{run_name}.csv"
                eval_cmd = shlex.join(
                    [
                        "python",
                        "eval_test.py",
                        "--sources",
                        source,
                        "--runs",
                        run_name,
                        "--out",
                        out,
                    ]
                )
                script = (
                    f"set -e; {download}; "
                    "cd one_off_projects/2026_10_07_change_alerts; "
                    f"mkdir -p {shlex.quote(args.out_dir)}; {eval_cmd}"
                )
                spec = BeakerExperimentSpec(
                    budget=DEFAULT_BUDGET,
                    description=f"{PROJECT_ID}/eval/{source}_{run_name}",
                    tasks=[
                        BeakerTaskSpec.new(
                            name="main",
                            beaker_image=args.image_name,
                            priority=args.priority,
                            command=["bash", "-c", script],
                            constraints=BeakerConstraints(
                                cluster=args.clusters.split(",")
                            ),
                            datasets=[
                                create_gcp_credentials_mount(),
                                WEKA_MOUNT.to_data_mount(),
                            ],
                            env_vars=get_base_env_vars(),
                            resources=BeakerTaskResources(
                                gpu_count=1, shared_memory="256GiB"
                            ),
                        )
                    ],
                )
                name = f"{PROJECT_ID}_eval_{source}_{run_name}_{str(uuid.uuid4())[:8]}"
                workload = beaker.experiment.create(name=name, spec=spec)
                print(f"{name}: {workload.experiment.id}", flush=True)


if __name__ == "__main__":
    main()
