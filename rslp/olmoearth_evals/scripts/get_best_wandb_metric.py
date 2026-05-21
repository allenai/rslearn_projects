"""Get best value of a metric in W&B over a run.

We look at the history of the run since we assume the summary metric mode may not be
set correctly.
"""

import argparse
import re

import wandb

DEFAULT_WANDB_ENTITY = "eai-ai2"


def get_best_metric(
    entity_name: str,
    project_name: str,
    metric_name: str,
    mode: str = "max",
    run_id: str | None = None,
    run_regex: str | None = None,
    runs_after: str | None = None,
) -> dict[str, float]:
    """Get the best value of the metric over the history of the run.

    Args:
        entity_name: the W&B entity containing the run.
        project_name: the W&B project containing the run.
        metric_name: the name of the metric to get best value of. It must correspond to
            a supported operation on pandas DataFrame.
        mode: how to aggregate the metric values over the history. It must be a
            supported operation on pandas dataframe.
        run_id: the W&B run ID. Exactly one of run_id and run_regex must be specified.
        run_regex: a regex for matching the W&B run name.
        runs_after: only consider runs after this time.

    Returns:
        a map from run name to the best value of the metric for that run.
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
            if run.metadata["startedAt"] < runs_after:
                continue
            runs.append(run)

    best_values = {}
    for run in runs:
        print(f"getting metric for {run.name}")
        history_df = run.history()
        if metric_name not in history_df:
            print(f"warning: run {run.name} does not have metric {metric_name}")
            continue
        op = getattr(history_df[metric_name], mode)
        best_values[run.name] = op()

    return best_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get best metric value for a W&B run")
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
        "--mode",
        type=str,
        required=False,
        help="Mode to determine what metric value is best (one of max or min)",
        default="max",
    )
    parser.add_argument(
        "--runs_after",
        type=str,
        help="Only process runs after this date",
        default=None,
    )
    args = parser.parse_args()

    best_values = get_best_metric(
        args.entity,
        args.project,
        metric_name=args.metric,
        mode=args.mode,
        run_regex=args.run_regex,
        run_id=args.run,
        runs_after=args.runs_after,
    )
    for run_name in sorted(best_values.keys()):
        print(run_name, best_values[run_name])
