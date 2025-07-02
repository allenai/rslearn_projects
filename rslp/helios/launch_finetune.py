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
    helios_checkpoint_path: str,
    experiment_id: str,
    image_name: str,
    encoder_embedding_size: int,
    patch_size: int,
    cluster: "list[str]",
    config_paths: "list[str]",
    rslp_project: str = DEFAULT_RSLP_PROJECT,
    gpus: int = 1,
    priority: str = "high",
    retries: int = 0,
    mode: str = "fit",
    profiler: "str | None" = None,
    local: bool = False,
    do_eval: bool = False,
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
        mode: Mode to run the model ('fit', 'validate', 'test', or 'predict').
        profiler: Profiler to use for training. Can be 'simple' or 'advanced'.
        local: Whether to run the command locally instead of spawning a Beaker job.
        do_eval: Whether to just run evals.
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

            config_str = config_str.replace("{CHECKPOINT_PATH}", helios_checkpoint_path)
            config_str = config_str.replace("{PATCH_SIZE}", str(patch_size))
            config_str = config_str.replace("{256/PATCH_SIZE}", str(256 // patch_size))
            config_str = config_str.replace("{128/PATCH_SIZE}", str(128 // patch_size))
            config_str = config_str.replace(
                "{ENCODER_EMBEDDING_SIZE}", str(encoder_embedding_size)
            )

            # String to yaml to add test metrics file key
            config = yaml.safe_load(config_str)
            if do_eval and "model" in config and "init_args" in config["model"]:
                model_name = "_".join(helios_checkpoint_path.split(os.path.sep)[-2:])  # "modelname_stepX"
                eval_task = config_paths[0].split(os.path.sep)[-2]
                path = os.path.join(full_eval_dir, f"{model_name}__{eval_task}.json")
                config["model"]["init_args"]["metrics_file"] = path
                logger.info(f"Saving test metrics to {path}")

            # Save the config file to the temporary directory
            tmp_config_fname = os.path.join(
                tmp_dir, f"{experiment_id}_{config_idx}.yaml"
            )
            with open(tmp_config_fname, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            tmp_config_fnames.append(tmp_config_fname)

        if local:
            # If running locally, assume we are in a gpu session
            # NOTE: assuming that all the args are passed through to the config file and do NOT get 
            # passed through the final call to rslp.rslearn_main (except for profiler)
            args = [
                "python",
                "-m",
                "rslp.rslearn_main",
                "model",
                "fit" if not do_eval else "test"
            ]
            paths = []
            for i, _ in enumerate(config_paths):
                args.append("--config")
                path = f"{tmp_dir}/{experiment_id}_{i}.yaml"
                paths.append(path)
                args.append(path)

            args.extend([
                "--rslp_experiment",
                experiment_id,
                "--rslp_project",
                rslp_project
            ])

            if profiler:
                args.append("--profiler")
                args.append(profiler)
            args.append("--autoresume=true")

            print("=" * 80)
            print("DEBUG: Command being spawned:")
            print(" ".join(args))
            print("=" * 80)

            # Need to monkeypatch configs to fix helios config path for local run
            # not great but works for now
            for path in paths:
                with open(path, "r") as f:
                    string = f.read()
                string = string.replace(
                    "/opt/helios/data/norm_configs/computed.json",
                    "./helios/data/norm_configs/computed.json"
                )
                with open(path, "w") as f:
                    f.write(string)
            input("Press Enter to continue...")
            subprocess.check_call(args)

        else:
            if do_eval:
                raise NotImplementedError("Eval mode not supported for Beaker job")

            extra_args = []
            if profiler:
                extra_args.extend(["--profiler", profiler])
                
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
