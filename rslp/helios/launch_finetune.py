"""Launch Helios fine-tuning experiments."""

import json
import os
import subprocess  # nosec
import tempfile
from pathlib import Path

import yaml

from rslp.log_utils import get_logger

DEFAULT_RSLP_PROJECT = "helios_finetuning"
CONFIG_BASE_DIR = Path("data/helios")
EVAL_BASE_DIR = "helios/eval_sweeps"

logger = get_logger(__name__)


def launch_finetune(
    experiment_id: str,
    config_paths: list[str],
    image_name: str | None = None,
    cluster: list[str] | None = None,
    helios_checkpoint_path: str | None = None,
    encoder_embedding_size: int | None = None,
    patch_size: int | None = None,
    rslp_project: str = DEFAULT_RSLP_PROJECT,
    gpus: int = 1,
    priority: str = "high",
    retries: int = 0,
    mode: str = "fit",
    profiler: str | None = None,
    local: bool = False,
    do_eval: bool = False,
    ckpt_path: str | None = None,
    allow_missing_weights: bool = False,
) -> None:
    """Launch Helios fine-tuning experiments.

    Args:
        experiment_id: the experiment name.
        config_paths: list of configuration files to use. Later config files override
            earlier configs in the list.
        image_name: what Beaker image to use. Must be specified if not local.
        cluster: see beaker_train. Must be specified if not local.
        helios_checkpoint_path: path to Helios checkpoint to fine-tune from. If none, assume
            it's already specified in the config.
        encoder_embedding_size: the embedding size of the encoder. If none, assume
            it's already specified in the config.
        patch_size: the patch size to use. If none, assume it's already specified in the config.
        rslp_project: optional override for W&B project to use. By default, uses DEFAULT_RSLP_PROJECT.
        gpus: how many GPUs to assign in the Beaker job. By default, uses 1.
        priority: what priority to use. By default, uses "high".
        retries: Beaker job retries. By default, uses 0.
        mode: Mode to run the model ('fit', 'validate', 'test', or 'predict').
        profiler: Profiler to use for training. Can be 'simple' or 'advanced' or None.
        local: Whether to run the command locally instead of spawning a Beaker job.
        do_eval: Whether to just run evals.
        ckpt_path: Optionally specify checkpoint path to load from if do_eval.
        allow_missing_weights: Whether to allow missing weights in checkpoint specified in --ckpt_path.
    """
    # Go into each config file (including the base ones) and make replacements as
    # needed.
    # I can't figure out how to override Helios checkpoint_path from
    # command-line since it appears in a list, so instead we create a copy
    # of all these configuration files in a temporary directory.
    with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
        weka_mounts = [
            dict(bucket_name="dfive-default", mount_path="/weka/dfive-default")
        ]
        full_eval_dir = os.path.join(weka_mounts[0]["mount_path"], EVAL_BASE_DIR)
        os.makedirs(full_eval_dir, exist_ok=True)

        # Need to use relative path from rslearn_projects folder since the config file
        # will be copied into the Beaker experiment's rslearn_projects copy.
        tmp_dir = os.path.relpath(tmp_dir)

        tmp_config_fnames: list[str] = []
        for config_idx, cur_config_fname in enumerate(config_paths):
            # Load the config file as string for template substitution
            with open(cur_config_fname) as f:
                config_str = f.read()

            if helios_checkpoint_path is not None:
                config_str = config_str.replace(
                    "{CHECKPOINT_PATH}", helios_checkpoint_path
                )
            if patch_size is not None:
                config_str = config_str.replace("{PATCH_SIZE}", str(patch_size))
                config_str = config_str.replace(
                    "{256/PATCH_SIZE}", str(256 // patch_size)
                )
                config_str = config_str.replace(
                    "{128/PATCH_SIZE}", str(128 // patch_size)
                )
            if encoder_embedding_size is not None:
                config_str = config_str.replace(
                    "{ENCODER_EMBEDDING_SIZE}", str(encoder_embedding_size)
                )

            # Save the config file to the temporary directory
            tmp_config_fname = os.path.join(
                tmp_dir, f"{experiment_id}_{config_idx}.yaml"
            )
            with open(tmp_config_fname, "w") as f:
                config = yaml.safe_load(config_str)
                yaml.dump(config, f, default_flow_style=False)
            tmp_config_fnames.append(tmp_config_fname)

        if local:
            # If running locally, assume we are in a gpu session
            # NOTE: assuming that all the args are passed through to the config file and do NOT get
            # passed through the final call to rslp.rslearn_main (except for profiler)
            # Keeping the same behavior as before.
            if mode == "fit" and do_eval:
                mode = "validate"
            args = [
                "python",
                "-m",
                "rslp.rslearn_main",
                "model",
                mode,
            ]
            paths = []
            for i, _ in enumerate(config_paths):
                args.append("--config")
                path = f"{tmp_dir}/{experiment_id}_{i}.yaml"
                paths.append(path)
                args.append(path)

            args.extend(
                ["--rslp_experiment", experiment_id, "--rslp_project", rslp_project]
            )

            if allow_missing_weights:
                args.append("--allow_missing_weights")

            if profiler:
                args.append("--profiler")
                args.append(profiler)
            args.append("--autoresume=true")

            if ckpt_path:
                args.extend(["--ckpt_path", ckpt_path])

            s = "\n" + "=" * 80
            s += "\nNOTE: Command being spawned:\n"
            s += " ".join(args)
            s += "\n" + "=" * 80 + "\n"
            logger.info(s)

            # Monkeypatch paths that are hardcoded in the config files
            for path in paths:
                with open(path) as f:
                    string = f.read()
                string = string.replace("/opt/", "./docker_build/")
                with open(path, "w") as f:
                    f.write(string)

            subprocess.check_call(args)  # nosec

        else:
            if do_eval:
                raise NotImplementedError("Eval mode not supported for Beaker job")
            if image_name is None:
                raise ValueError("image_name must be specified if not local")
            if cluster is None:
                raise ValueError("cluster must be specified if not local")

            extra_args = []
            if profiler:
                extra_args.extend(["--profiler", profiler])
            if allow_missing_weights:
                extra_args.append("--allow_missing_weights")

            args = [
                "python",
                "-m",
                "rslp.main",
                "common",
                "beaker_train",
                "--mode",
                mode,
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
            if extra_args:
                args.extend(["--extra_args", json.dumps(extra_args)])
            logger.info(f"Launching job by running: {args}")
            subprocess.check_call(args)  # nosec
