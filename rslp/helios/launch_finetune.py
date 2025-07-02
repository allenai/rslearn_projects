"""Launch Helios fine-tuning experiments."""

import json
import os
import subprocess  # nosec
import tempfile
from pathlib import Path

from rslp.log_utils import get_logger

DEFAULT_RSLP_PROJECT = "helios_finetuning"
CONFIG_BASE_DIR = Path("data/helios")

logger = get_logger(__name__)


def launch_finetune(
    helios_checkpoint_path: str,
    experiment_id: str,
    image_name: str,
    encoder_embedding_size: int,
    patch_size: int,
    cluster: list[str],
    config_paths: list[str],
    rslp_project: str = DEFAULT_RSLP_PROJECT,
    gpus: int = 1,
    priority: str = "high",
    retries: int = 0,
) -> None:
    """Launch Helios fine-tuning experiments.

    Args:
        helios_checkpoint_path: path to Helios checkpoint to fine-tune from.
        experiment_id: the experiment name.
        image_name: what Beaker image to use.
        encoder_embedding_size: the embedding size of the encoder.
        patch_size: the patch size to use.
        cluster: see beaker_train.
        config_paths: list of configuration files to use. Later config files override
            earlier configs in the list.
        rslp_project: optional override for W&B project to use.
        gpus: how many GPUs to assign in the Beaker job.
        priority: what priority to use.
        retries: Beaker job retries.
    """
    # Go into each config file (including the base ones) and make replacements as
    # needed.
    # I can't figure out how to override Helios checkpoint_path from
    # command-line since it appears in a list, so instead we create a copy
    # of all these configuration files in a temporary directory.
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        # Need to use relative path from rslearn_projects folder since the config file
        # will be copied into the Beaker experiment's rslearn_projects copy.
        tmp_dir = os.path.relpath(tmp_dir)

        tmp_config_fnames: list[str] = []
        for config_idx, cur_config_fname in enumerate(config_paths):
            with open(cur_config_fname) as f:
                config_str = f.read()
            config_str = config_str.replace("{CHECKPOINT_PATH}", helios_checkpoint_path)
            config_str = config_str.replace("{PATCH_SIZE}", str(patch_size))
            config_str = config_str.replace("{256/PATCH_SIZE}", str(256 // patch_size))
            config_str = config_str.replace("{128/PATCH_SIZE}", str(128 // patch_size))
            config_str = config_str.replace(
                "{ENCODER_EMBEDDING_SIZE}", str(encoder_embedding_size)
            )

            tmp_config_fname = os.path.join(
                tmp_dir, f"{experiment_id}_{config_idx}.yaml"
            )
            with open(tmp_config_fname, "w") as f:
                f.write(config_str)
            tmp_config_fnames.append(tmp_config_fname)

        weka_mounts = [
            dict(bucket_name="dfive-default", mount_path="/weka/dfive-default")
        ]

        # OK now we can prepare all the command-line arguments to beaker_train.
        args = [
            "python",
            "-m",
            "rslp.main",
            "common",
            "beaker_train",
            "--config_paths",
            json.dumps(tmp_config_fnames),
            "--image_name",
            image_name,
            "--cluster",
            json.dumps(cluster),
            "--weka_mounts",
            json.dumps(weka_mounts),
            "--gpus",
            str(gpus),
            "--project_id",
            rslp_project,
            "--experiment_id",
            experiment_id,
            "--priority",
            priority,
            "--retries",
            str(retries),
        ]
        logger.info(f"Launching job by running: {args}")
        subprocess.check_call(args)  # nosec
