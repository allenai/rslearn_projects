"""Get best value of a metric in W&B over a run.

We look at the history of the run since we assume the summary metric mode may not be
set correctly.
"""

import argparse

import wandb

DEFAULT_WANDB_ENTITY = "eai-ai2"


def get_best_metric(
    entity_name: str,
    project_name: str,
    run_id: str,
    metric_name: str,
    mode: str = "max",
) -> float:
    """Get the best value of the metric over the history of the run.

    Args:
        entity_name: the W&B entity containing the run.
        project_name: the W&B project containing the run.
        run_id: the W&B run ID.
        metric_name: the name of the metric to get best value of. It must correspond to
            a supported operation on pandas DataFrame.
        mode: how to aggregate the metric values over the history. It must be a
            supported operation on pandas dataframe.

    Returns:
        the best value of the metric.
    """
    api = wandb.Api()
    run = api.run(f"{entity_name}/{project_name}/{run_id}")
    history_df = run.history()
    op = getattr(history_df[metric_name], mode)
    best_value = op()
    return best_value


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
        required=True,
        help="W&B run ID",
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
    args = parser.parse_args()

    best_value = get_best_metric(
        args.entity, args.project, args.run, args.metric, args.mode
    )
    print(best_value)
