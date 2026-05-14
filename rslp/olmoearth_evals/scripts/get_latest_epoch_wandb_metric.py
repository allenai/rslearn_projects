"""Get metric values across W&B runs at the latest common epoch.

We look at the history of all matching runs, find the latest epoch that is
present in every matching run (i.e. the minimum of the per-run max epochs), and
then read each run's metric value at that epoch.
"""

import argparse
import re

import pandas as pd
import wandb

DEFAULT_WANDB_ENTITY = "eai-ai2"
EPOCH_COLUMN = "epoch"


def _max_epoch(history_df: pd.DataFrame) -> int | None:
    """Return the largest epoch present in the history, or None if unavailable."""
    if EPOCH_COLUMN not in history_df:
        return None
    epochs = history_df[EPOCH_COLUMN].dropna()
    if epochs.empty:
        return None
    return int(epochs.max())


def get_metric_at_latest_epoch(
    entity_name: str,
    project_name: str,
    metric_name: str,
    run_id: str | None = None,
    run_regex: str | None = None,
    runs_after: str | None = None,
) -> tuple[int | None, dict[str, float]]:
    """Get the metric value at the latest epoch common to all matching runs.

    Args:
        entity_name: the W&B entity containing the runs.
        project_name: the W&B project containing the runs.
        metric_name: the name of the metric to retrieve.
        run_id: the W&B run ID. Exactly one of run_id and run_regex must be specified.
        run_regex: a regex for matching the W&B run name.
        runs_after: only consider runs after this time.

    Returns:
        a tuple (latest_common_epoch, values), where values is a map from run
        name to the metric value at that epoch. Returns (None, {}) if no runs
        match or no run has an epoch column.
    """
    if (run_id is None and run_regex is None) or (
        run_id is not None and run_regex is not None
    ):
        raise ValueError("exactly one of run_id and run_regex must be specified")

    api = wandb.Api()
    runs = []

    if run_id is not None:
        run = api.run(f"{entity_name}/{project_name}/{run_id}")
        runs.append(run)

    elif run_regex is not None:
        pattern = re.compile(run_regex)
        for run in api.runs(f"{entity_name}/{project_name}"):
            if not pattern.match(run.name):
                continue
            # We can just use string comparison for the date, which is of the form:
            # "2025-06-27T21:49:03.775779Z"
            if runs_after is not None and run.metadata["startedAt"] < runs_after:
                continue
            runs.append(run)

    # First pass: fetch histories and compute per-run max epoch.
    histories: dict[str, pd.DataFrame] = {}
    max_epochs: dict[str, int] = {}
    for run in runs:
        print(f"getting history for {run.name}")
        history_df = run.history()
        if metric_name not in history_df:
            print(f"warning: run {run.name} does not have metric {metric_name}")
            continue
        run_max_epoch = _max_epoch(history_df)
        if run_max_epoch is None:
            print(f"warning: run {run.name} has no '{EPOCH_COLUMN}' values")
            continue
        histories[run.name] = history_df
        max_epochs[run.name] = run_max_epoch

    if not max_epochs:
        return None, {}

    latest_common_epoch = min(max_epochs.values())
    limiting_runs = sorted(
        run_name
        for run_name, run_max_epoch in max_epochs.items()
        if run_max_epoch == latest_common_epoch
    )
    print(
        f"latest common epoch across {len(max_epochs)} runs: {latest_common_epoch} "
        f"(limited by: {', '.join(limiting_runs)})"
    )
    for run_name, run_max_epoch in max_epochs.items():
        if run_max_epoch != latest_common_epoch:
            print(
                f"  note: run {run_name} reached epoch {run_max_epoch} "
                f"(> {latest_common_epoch}); using value at {latest_common_epoch}"
            )

    # Second pass: read metric value at the latest common epoch.
    values: dict[str, float] = {}
    for run_name, history_df in histories.items():
        rows = history_df[history_df[EPOCH_COLUMN] == latest_common_epoch]
        metric_values = rows[metric_name].dropna()
        if metric_values.empty:
            print(
                f"warning: run {run_name} has no {metric_name} value at "
                f"epoch {latest_common_epoch}"
            )
            continue
        # If the metric is logged multiple times in the same epoch, take the last.
        values[run_name] = float(metric_values.iloc[-1])

    return latest_common_epoch, values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Get metric values at the latest epoch common to all matching W&B runs"
        )
    )
    parser.add_argument(
        "--entity",
        type=str,
        required=False,
        help="W&B entity",
        default=DEFAULT_WANDB_ENTITY,
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="W&B project",
    )
    parser.add_argument(
        "--run",
        type=str,
        help="W&B run ID",
        default=None,
    )
    parser.add_argument(
        "--run_regex",
        type=str,
        help="Regex for matching W&B run name",
        default=None,
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        help="The name of the metric",
    )
    parser.add_argument(
        "--runs_after",
        type=str,
        help="Only process runs after this date",
        default=None,
    )
    args = parser.parse_args()

    latest_epoch, values = get_metric_at_latest_epoch(
        args.entity,
        args.project,
        metric_name=args.metric,
        run_regex=args.run_regex,
        run_id=args.run,
        runs_after=args.runs_after,
    )
    if latest_epoch is None:
        print("no runs with usable history found")
    else:
        print(f"metric values at epoch {latest_epoch}:")
        for run_name in sorted(values.keys()):
            print(run_name, values[run_name])
