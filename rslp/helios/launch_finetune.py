"""Launch Helios fine-tuning experiments."""

import json
import os
import subprocess  # nosec
import tempfile
from pathlib import Path

from rslp.log_utils import get_logger

DEFAULT_RSLP_PROJECT = "helios_finetuning"
CONFIG_BASE_DIR = Path("data/helios")
DEFAULT_CLUSTER = [
    "ai2/jupiter-cirrascale-2",
    "ai2/saturn-cirrascale",
    "ai2/neptune-cirrascale",
]

logger = get_logger(__name__)


def launch_finetune(
    helios_checkpoint_path: str,
    experiment_prefix: str,
    image_name: str,
    tasks: list[str] | None = None,
    rslp_project: str = DEFAULT_RSLP_PROJECT,
    cluster: list[str] = DEFAULT_CLUSTER,
) -> None:
    """Launch Helios fine-tuning experiments.

    Args:
        helios_checkpoint_path: path to Helios checkpoint to fine-tune from.
        experiment_prefix: prefix for the run name on W&B.
        image_name: what Beaker image to use.
        tasks: optional list of tasks to launch, e.g. ["eurosat",
            "satlas_marine_infra"]. Default is to launch all tasks.
        rslp_project: optional override for W&B project to use.
        cluster: see beaker_train.
    """
    if tasks is None:
        task_dirs = list(CONFIG_BASE_DIR.iterdir())
    else:
        task_dirs = [CONFIG_BASE_DIR / task_name for task_name in tasks]

    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        # Need to use relative path from rslearn_projects folder since the config file
        # will be copied into the Beaker experiment's rslearn_projects copy.
        tmp_dir = os.path.relpath(tmp_dir)

        for task_dir in task_dirs:
            for config_fname in task_dir.iterdir():
                experiment_id = (
                    f"{experiment_prefix}_{task_dir.name}_{config_fname.name}"
                )

                # I can't figure out how to override Helios checkpoint_path from
                # command-line since it appears in a list, so instead we create a copy
                # of the configuration file in a temporary directory.
                with config_fname.open() as f:
                    config_str = f.read()
                config_str = config_str.replace(
                    "{CHECKPOINT_PATH}", helios_checkpoint_path
                )

                tmp_config_fname = os.path.join(tmp_dir, f"{experiment_id}.yaml")
                with open(tmp_config_fname, "w") as f:
                    f.write(config_str)

                weka_mounts = [
                    dict(bucket_name="dfive-default", mount_path="/weka/dfive-default")
                ]

                args = [
                    "python",
                    "-m",
                    "rslp.main",
                    "common",
                    "beaker_train",
                    "--config_path",
                    tmp_config_fname,
                    "--image_name",
                    image_name,
                    "--cluster",
                    json.dumps(cluster),
                    "--weka_mounts",
                    json.dumps(weka_mounts),
                    "--project_id",
                    rslp_project,
                    "--experiment_id",
                    experiment_id,
                ]
                logger.info(f"Launching job by running: {args}")
                subprocess.check_call(args)  # nosec
