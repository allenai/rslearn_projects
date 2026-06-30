"""Summarize W&B eval metrics across many runs as a (method x task) table.

Given a single regex that matches many run names of the form
``{method_id}_{task}_{model}``, this script:

* infers the task for each run from the known ``TASK_CONFIGS`` in
  ``rslp.olmoearth_evals.launch``,
* looks up the monitored metric and mode (min/max) for each task by parsing
  the task's YAML chain in ``data/olmoearth_evals/tasks/``,
* drops any run whose history hasn't reached ``--required_max_epoch`` (i.e.
  training did not finish),
* and prints two TSV tables to stdout: one with the metric value at the
  latest (=required) epoch, and one with the best metric value over training.

Rows are method ids (the run name with the ``_{task}_{model}`` suffix
stripped), columns are tasks.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import wandb
import yaml

from rslp.olmoearth_evals.launch import TASK_CONFIGS

DEFAULT_WANDB_ENTITY = "eai-ai2"
EPOCH_COLUMN = "epoch"
MANAGED_CHECKPOINT_CLASS = (
    "rslearn.train.callbacks.checkpointing.ManagedBestLastCheckpoint"
)
# Repo root is rslearn_projects/ (rslp/olmoearth_evals/scripts/X.py -> parents[3]).
REPO_ROOT = Path(__file__).resolve().parents[3]
TASKS_DIR = REPO_ROOT / "data" / "olmoearth_evals" / "tasks"


def get_task_metric(task: str) -> tuple[str, str]:
    """Return (metric_name, mode) for a task by parsing its YAML chain.

    Walks the YAMLs in ``TASK_CONFIGS[task]`` in order, taking the last
    ``ManagedBestLastCheckpoint`` callback's ``monitor`` / ``mode``. Later
    configs in the chain override earlier ones, matching the jsonargparse
    behavior used at launch time.
    """
    monitor: str | None = None
    mode: str | None = None
    for cfg_name in TASK_CONFIGS[task]:
        yaml_path = TASKS_DIR / f"{cfg_name}.yaml"
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        callbacks = (cfg.get("trainer") or {}).get("callbacks") or []
        for cb in callbacks:
            if not isinstance(cb, dict):
                continue
            if cb.get("class_path") != MANAGED_CHECKPOINT_CLASS:
                continue
            init_args = cb.get("init_args") or {}
            if "monitor" in init_args:
                monitor = init_args["monitor"]
            if "mode" in init_args:
                mode = init_args["mode"]
    if monitor is None or mode is None:
        raise ValueError(
            f"task {task}: could not find ManagedBestLastCheckpoint in YAML chain"
        )
    return monitor, mode


def parse_run_name(run_name: str, tasks: list[str]) -> tuple[str, str, str] | None:
    """Split ``{method_id}_{task}_{model}`` using the known task list.

    Tasks are tried longest-first so that overlapping names (e.g. ``pastis_ts``
    vs ``pastis_uni``) are matched correctly.
    """
    for task in sorted(tasks, key=len, reverse=True):
        marker = f"_{task}_"
        idx = run_name.rfind(marker)
        if idx < 0:
            continue
        method_id = run_name[:idx]
        model = run_name[idx + len(marker) :]
        if not method_id or not model:
            continue
        return method_id, task, model
    return None


def metric_at_max_epoch(
    history: pd.DataFrame, metric: str, required_epoch: int
) -> float | None:
    """Return metric at ``required_epoch`` if the run reached it, else None."""
    if EPOCH_COLUMN not in history or metric not in history:
        return None
    epochs = history[EPOCH_COLUMN].dropna()
    if epochs.empty or int(epochs.max()) < required_epoch:
        return None
    rows = history[history[EPOCH_COLUMN] == required_epoch]
    values = rows[metric].dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def best_metric(history: pd.DataFrame, metric: str, mode: str) -> float | None:
    """Return min/max of metric over history, ignoring NaNs."""
    if metric not in history:
        return None
    series = history[metric].dropna()
    if series.empty:
        return None
    op = getattr(series, mode)
    return float(op())


def print_table(
    title: str,
    methods: list[str],
    tasks: list[str],
    values: dict[tuple[str, str], float],
) -> None:
    """Print a TSV table with method_id rows and task columns."""
    print(f"=== {title} ===")
    print("\t".join(["method_id", *tasks]))
    for m in methods:
        cells = [m]
        for t in tasks:
            v = values.get((m, t))
            cells.append(f"{v:.4f}" if v is not None else "")
        print("\t".join(cells))


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize W&B eval metrics across many runs as a (method x task)"
            " table. Prints two TSV tables: latest-epoch values and best values."
        )
    )
    parser.add_argument("--entity", default=DEFAULT_WANDB_ENTITY, help="W&B entity")
    parser.add_argument("--project", required=True, help="W&B project")
    parser.add_argument(
        "--run_regex",
        required=True,
        help="Regex matched against run name (re.match semantics).",
    )
    parser.add_argument(
        "--runs_after",
        default=None,
        help=(
            "ISO timestamp; only consider runs whose startedAt is >= this"
            " (lexicographic comparison)."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=(
            "Restrict to this subset of tasks (default: all tasks in"
            " TASK_CONFIGS that have at least one matching finished run)."
        ),
    )
    parser.add_argument(
        "--required_max_epoch",
        type=int,
        default=99,
        help="Drop runs whose max epoch is below this (default 99).",
    )
    args = parser.parse_args()

    known_tasks = list(TASK_CONFIGS.keys())
    if args.tasks:
        unknown = set(args.tasks) - set(known_tasks)
        if unknown:
            raise ValueError(f"unknown task(s): {sorted(unknown)}")
        candidate_tasks = list(args.tasks)
    else:
        candidate_tasks = known_tasks

    task_metrics: dict[str, tuple[str, str]] = {
        t: get_task_metric(t) for t in candidate_tasks
    }

    api = wandb.Api()
    pattern = re.compile(args.run_regex)
    runs = []
    for run in api.runs(f"{args.entity}/{args.project}"):
        if not pattern.match(run.name):
            continue
        if args.runs_after is not None:
            started_at = (run.metadata or {}).get("startedAt")
            if started_at is None or started_at < args.runs_after:
                continue
        runs.append(run)

    # (method_id, task) -> (run, history). Keep the most recently created if
    # there are duplicates.
    latest_values: dict[tuple[str, str], float] = {}
    best_values: dict[tuple[str, str], float] = {}
    seen_methods: set[str] = set()
    seen_tasks: set[str] = set()
    chosen_run_id: dict[tuple[str, str], str] = {}
    chosen_created_at: dict[tuple[str, str], str] = {}

    for run in runs:
        parsed = parse_run_name(run.name, candidate_tasks)
        if parsed is None:
            print(f"skip {run.name}: no known task suffix matched")
            continue
        method_id, task, _model = parsed
        if task not in task_metrics:
            continue

        created_at = run.created_at or ""
        key = (method_id, task)
        prev = chosen_created_at.get(key)
        if prev is not None and created_at <= prev:
            print(
                f"skip {run.name} (id={run.id}): older duplicate for"
                f" method={method_id} task={task}, keeping {chosen_run_id[key]}"
            )
            continue

        print(f"fetching {run.name}")
        history = run.history()
        metric_name, mode = task_metrics[task]
        latest = metric_at_max_epoch(history, metric_name, args.required_max_epoch)
        if latest is None:
            print(
                f"skip {run.name}: did not reach epoch {args.required_max_epoch}"
                f" or missing metric {metric_name}"
            )
            continue
        best = best_metric(history, metric_name, mode)
        if best is None:
            print(f"skip {run.name}: no values for metric {metric_name}")
            continue

        latest_values[key] = latest
        best_values[key] = best
        chosen_run_id[key] = run.id
        chosen_created_at[key] = created_at
        seen_methods.add(method_id)
        seen_tasks.add(task)

    if not seen_methods:
        print("no finished runs matched")
        return

    methods = sorted(seen_methods)
    # Preserve TASK_CONFIGS order, restricted to tasks we actually saw.
    tasks = [t for t in candidate_tasks if t in seen_tasks]

    print()
    print_table(
        f"latest epoch (={args.required_max_epoch})",
        methods,
        tasks,
        latest_values,
    )
    print()
    print_table("best over training", methods, tasks, best_values)


if __name__ == "__main__":
    main()
