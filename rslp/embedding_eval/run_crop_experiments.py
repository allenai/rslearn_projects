"""Run embedding + knn/linear_probe experiments across tasks and crop configs.

For each task, computes OlmoEarth embeddings at various crop sizes and label positions,
then evaluates with knn and linear probe. Also evaluates GSE embeddings.

Results are saved incrementally to a JSON file.

Usage (from rslearn_projects root):
    python -m rslp.embedding_eval.run_crop_experiments \
        --experiment_config rslp/embedding_eval/crop_experiment_results.json

The script is designed so it can be run on multiple machines in parallel.
"""

import argparse
import csv
import io
import json
import os
import random
import re
import subprocess  # nosec
import sys
from dataclasses import dataclass

from pydantic import BaseModel, Field, model_validator
from rslearn.utils.fsspec import open_atomic
from upath import UPath


@dataclass
class TaskConfig:
    """Configuration for one task (dataset) to evaluate on."""

    ds_path: str
    label_key: str = "label"
    split_key: str = "group"


DS_ROOT = "/weka/dfive-default/rslearn-eai/datasets/alphaearth_supplemental_evaluations"
TASKS: dict[str, TaskConfig] = {
    "africa_crop_mask": TaskConfig(ds_path=os.path.join(DS_ROOT, "africa_crop_mask")),
    "canada_crops_fine": TaskConfig(ds_path=os.path.join(DS_ROOT, "canada_crops_fine")),
    "descals": TaskConfig(ds_path=os.path.join(DS_ROOT, "descals")),
    "glance": TaskConfig(ds_path=os.path.join(DS_ROOT, "glance")),
    "lcmap_lu": TaskConfig(ds_path=os.path.join(DS_ROOT, "lcmap_lu")),
    "us_trees": TaskConfig(ds_path=os.path.join(DS_ROOT, "us_trees")),
    "nandi": TaskConfig(
        ds_path="/weka/dfive-default/rslearn-eai/datasets/nandi/",
        label_key="category",
        split_key="split",
    ),
    "awf": TaskConfig(
        ds_path="/weka/dfive-default/rslearn-eai/datasets/awf/",
        label_key="category",
        split_key="split",
    ),
}

BATCH_SIZE = 16
KNN_K = 3


class CropConfig(BaseModel):
    """A setting of input size and cropping method to evaluate on."""

    input_size: int
    pos_name: str
    pos_args: list[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    """ExperimentConfig specifies the different variations to evaluate."""

    crop_configs: list[CropConfig]
    methods: list[str]
    patch_size: int
    results_file: str
    model_id: str | None = None
    checkpoint_dir: str | None = None
    h5_suffix: str = ""

    @model_validator(mode="after")
    def validate_model_source(self) -> "ExperimentConfig":
        """Validate the ExperimentConfig has one of model_id and checkpoint_dir."""
        if (self.model_id is None) == (self.checkpoint_dir is None):
            raise ValueError(
                "Experiment config must specify exactly one of model_id or checkpoint_dir"
            )
        return self


def load_experiment_config(experiment_config_path: str) -> ExperimentConfig:
    """Load experiment configuration from JSON filename."""
    with open(experiment_config_path) as f:
        raw = json.load(f)
    return ExperimentConfig.model_validate(raw)


def load_results(results_file: str) -> dict:
    """Load results dict from JSON file."""
    if os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)
    return {}


def save_results(results_file: str, results: dict) -> None:
    """Save results dict to JSON file."""
    with open_atomic(UPath(results_file), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [saved] {results_file}")


def parse_mean_accuracy(output: str) -> float | None:
    """Parse the mean accuracy from get_balanced_accuracy.py output."""
    for line in output.splitlines():
        m = re.search(r"Mean:\s+([\d.]+)", line)
        if m:
            return float(m.group(1))
    return None


def h5_name(input_size: int, pos_name: str, patch_size: int, h5_suffix: str) -> str:
    """Get the H5 name under which to store embeddings for these conditions."""
    suffix = h5_suffix
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    return f"embeddings_ps{patch_size}_crop{input_size}_{pos_name}{suffix}.h5"


def run_embeddings(
    ds_path: str,
    input_size: int,
    pos_name: str,
    pos_args: list[str],
    exp_cfg: ExperimentConfig,
) -> str:
    """Compute embeddings if H5 doesn't exist. Returns the H5 path."""
    out_path = os.path.join(
        ds_path,
        h5_name(
            input_size=input_size,
            pos_name=pos_name,
            patch_size=exp_cfg.patch_size,
            h5_suffix=exp_cfg.h5_suffix,
        ),
    )
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists")
        return out_path

    cmd = [
        sys.executable,
        "-m",
        "rslp.embedding_eval.compute_olmoearth_embeddings",
        "--ds_path",
        ds_path,
        "--patch_size",
        str(exp_cfg.patch_size),
        "--input_size",
        str(input_size),
        "--batch_size",
        str(BATCH_SIZE),
        "--mode",
        "center",
        "--out_path",
        out_path,
    ] + pos_args
    if exp_cfg.checkpoint_dir is not None:
        cmd.extend(["--checkpoint_dir", exp_cfg.checkpoint_dir])
    else:
        assert exp_cfg.model_id is not None
        cmd.extend(["--model_id", exp_cfg.model_id])
    print(f"  [embed] {' '.join(cmd)}")
    subprocess.check_call(cmd)  # nosec
    return out_path


def run_eval(cfg: TaskConfig, embed_fname: str, method: str) -> float | None:
    """Run get_balanced_accuracy.py and return mean accuracy."""
    cmd = [
        sys.executable,
        "-m",
        "rslp.embedding_eval.get_balanced_accuracy",
        "--ds_path",
        cfg.ds_path,
        "--embed_fname",
        embed_fname,
        "--repeats",
        "1",
        "--samples",
        "0",
        "--k",
        str(KNN_K),
        "--split_key",
        cfg.split_key,
        "--label_key",
        cfg.label_key,
        "--method",
        method,
    ]
    print(f"  [eval {method}] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)  # nosec
    print(result.stdout)
    if result.returncode != 0:
        print(f"  [error] {result.stderr}")
        return None
    return parse_mean_accuracy(result.stdout)


def result_exists(
    results: dict, task: str, method_key: str, embed_key: str, crop_key: str
) -> bool:
    """Check whether a result already exists in the results dict."""
    val = results.get(task, {}).get(method_key, {}).get(embed_key)
    if val is None:
        return False
    if not isinstance(val, dict):
        # Flat value (e.g. gse: 0.87) — result exists.
        return True
    return crop_key in val


def store_result(
    results_file: str,
    task: str,
    method_key: str,
    embed_key: str,
    crop_key: str,
    acc: float,
) -> None:
    """Re-read JSON, merge in the new result, and atomically write it back.

    If two processes call store_result simultaneously, we might lose one result, but
    the result file will be in a consistent state so eventually we should still get the
    other result.
    """
    results = load_results(results_file)
    results.setdefault(task, {}).setdefault(method_key, {}).setdefault(embed_key, {})[
        crop_key
    ] = acc
    save_results(results_file, results)


def print_results_csv(results: dict) -> None:
    """Print results as CSV to stdout. Rows=(method, task), columns=model/crop combos."""
    # Discover all methods and (embed_key, crop_key) columns.
    methods: set[str] = set()
    col_set: set[tuple[str, str]] = set()
    for task_data in results.values():
        for method_key, method_data in task_data.items():
            methods.add(method_key)
            for embed_key, embed_data in method_data.items():
                if isinstance(embed_data, dict):
                    for crop_key in embed_data:
                        col_set.add((embed_key, crop_key))
                else:
                    col_set.add((embed_key, ""))

    columns = sorted(col_set)
    methods_sorted = sorted(methods)
    tasks = sorted(results.keys())

    buf = io.StringIO()
    writer = csv.writer(buf)

    header = ["method", "task"] + [
        f"{embed}/{crop}" if crop else embed for embed, crop in columns
    ]
    writer.writerow(header)

    for method_key in methods_sorted:
        for task in tasks:
            row = [method_key, task]
            for embed_key, crop_key in columns:
                val = results.get(task, {}).get(method_key, {}).get(embed_key, {})
                if isinstance(val, dict):
                    cell = val.get(crop_key, "")
                else:
                    cell = val if crop_key == "" else ""
                row.append(cell)
            writer.writerow(row)

    print(buf.getvalue(), end="")


def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        help="Path to JSON file describing crop experiment settings",
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    """Main entrypoint."""
    args = parse_args()
    exp_cfg = load_experiment_config(args.experiment_config)

    # Build flat list of all experiments and shuffle so parallel processes spread out.
    # Each entry: (task, method, embed_key, crop_key, input_size, pos_name, pos_args)
    experiments: list[
        tuple[str, str, str, str, int | None, str | None, list[str] | None]
    ] = []
    for task in TASKS:
        for method in exp_cfg.methods:
            # GSE baseline
            experiments.append((task, method, "gse", "center", None, None, None))
            # OlmoEarth crop configs
            for crop_cfg in exp_cfg.crop_configs:
                crop_key = f"{crop_cfg.input_size}_{crop_cfg.pos_name}"
                experiments.append(
                    (
                        task,
                        method,
                        "olmoearth",
                        crop_key,
                        crop_cfg.input_size,
                        crop_cfg.pos_name,
                        crop_cfg.pos_args,
                    )
                )

    random.shuffle(experiments)

    for (
        task,
        method,
        embed_key,
        crop_key,
        input_size,
        pos_name,
        pos_args,
    ) in experiments:
        cfg = TASKS[task]
        method_key = "lp" if method == "linear_probe" else "knn"

        if result_exists(
            load_results(exp_cfg.results_file), task, method_key, embed_key, crop_key
        ):
            print(f"  [skip] {task} {embed_key} {crop_key} method={method}")
            continue

        label = f"{task} {embed_key} {crop_key} method={method}"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        if embed_key == "gse":
            embed_fname = "gse"
        else:
            assert input_size is not None
            assert pos_name is not None
            assert pos_args is not None
            h5_path = run_embeddings(
                cfg.ds_path,
                input_size,
                pos_name,
                pos_args,
                exp_cfg,
            )
            embed_fname = h5_path

        acc = run_eval(cfg, embed_fname, method)
        if acc is not None:
            store_result(
                exp_cfg.results_file,
                task,
                method_key,
                embed_key,
                crop_key,
                acc,
            )

    print(f"\n\nAll done! Results saved to {exp_cfg.results_file}\n")
    print_results_csv(load_results(exp_cfg.results_file))


if __name__ == "__main__":
    main()
