"""Launch eval_test.py on Beaker, one job per source dataset and model.

The current code is uploaded like rslp.common.beaker_train does, and each job writes
its metrics to {out_dir}/{source}_{model}.csv, where round 1/2 models are written as
{date}_{run}, e.g. 2026_10_12_fixed4_history_pool.
The Beaker image must include the rslearn version with rslearn.change_alerts.layers.
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
from eval_test import OLD_MODELS
from make_model_configs import PROJECT_NAME, RUNS_BY_NAME

from rslp import launcher_lib
from rslp.utils.beaker import (
    DEFAULT_BUDGET,
    DEFAULT_MIN_RUNTIME,
    DEFAULT_WORKSPACE,
    WekaMount,
    create_gcp_credentials_mount,
    get_base_env_vars,
)

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
    parser.add_argument("--models", default=",".join(list(RUNS_BY_NAME) + OLD_MODELS))
    parser.add_argument("--priority", default="high")
    args = parser.parse_args()

    # upload_code archives the current directory.
    os.chdir(REPO_ROOT)
    launcher_lib.upload_code(PROJECT_NAME, CODE_EXPERIMENT_ID)

    download = (
        "python -c 'from rslp.launcher_lib import download_code; "
        f'download_code("{PROJECT_NAME}", "{CODE_EXPERIMENT_ID}")\''
    )
    with Beaker.from_env(default_workspace=DEFAULT_WORKSPACE) as beaker:
        for source in args.sources.split(","):
            for model in args.models.split(","):
                model_slug = model.replace("_change_alerts/", "_")
                out = f"{args.out_dir}/{source}_{model_slug}.csv"
                eval_cmd = shlex.join(
                    [
                        "python",
                        "eval_test.py",
                        "--sources",
                        source,
                        "--models",
                        model,
                        "--out",
                        out,
                    ]
                )
                script = (
                    f"set -e; {download}; "
                    "cd one_off_projects/2026_10_30_change_alerts; "
                    f"mkdir -p {shlex.quote(args.out_dir)}; {eval_cmd}"
                )
                spec = BeakerExperimentSpec(
                    budget=DEFAULT_BUDGET,
                    description=f"{PROJECT_NAME}/eval/{source}_{model_slug}",
                    tasks=[
                        BeakerTaskSpec.new(
                            name="main",
                            beaker_image=args.image_name,
                            priority=args.priority,
                            command=["bash", "-c", script],
                            constraints=BeakerConstraints(
                                cluster=args.clusters.split(",")
                            ),
                            # Without a minimum runtime the job is not allocated, and
                            # may never be scheduled.
                            min_runtime=DEFAULT_MIN_RUNTIME,
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
                name = (
                    f"{PROJECT_NAME}_eval_{source}_{model_slug}_{str(uuid.uuid4())[:8]}"
                )
                workload = beaker.experiment.create(name=name, spec=spec)
                print(f"{name}: {workload.experiment.id}", flush=True)


if __name__ == "__main__":
    main()
