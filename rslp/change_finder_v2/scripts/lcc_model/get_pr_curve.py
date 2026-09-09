"""Plot precision-recall curves for LCC model runs in a W&B project.

For each run in a W&B project we read the per-threshold precision and recall
metrics logged by the segmentation F1 metric (with report_per_class enabled).
The metric keys look like:

    val_binary/precision_0,9_binary/f1/cls_2
    val_binary/recall_0,9_binary/f1/cls_2
    val_binary/F1_0,9_binary/f1/cls_2

where ``binary`` is the task name, ``0,9`` is the score threshold (a "." in the
threshold is logged as "," since metric names can't contain "."), and ``cls_2``
is the change class. We then plot one PR curve per run.

Two modes select which epoch the precision/recall values come from:
- ``last``: the last recorded value of each metric.
- ``best``: the epoch where the max F1 (across thresholds) is highest; all
  precision/recall values then come from that single epoch.
"""

import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd
import wandb

DEFAULT_WANDB_ENTITY = "eai-ai2"
EPOCH_COLUMN = "epoch"
AUROC_COLUMN = "val_binary/auroc"
# The change class index in the binary change task.
CHANGE_CLASS_INDEX = 2

PRECISION_RE = re.compile(
    rf"^val_binary/precision_(?P<thr>[0-9,]+)_binary/f1/cls_{CHANGE_CLASS_INDEX}$"
)
RECALL_RE = re.compile(
    rf"^val_binary/recall_(?P<thr>[0-9,]+)_binary/f1/cls_{CHANGE_CLASS_INDEX}$"
)
F1_RE = re.compile(
    rf"^val_binary/F1_(?P<thr>[0-9,]+)_binary/f1/cls_{CHANGE_CLASS_INDEX}$"
)


def _threshold_columns(
    history_df: pd.DataFrame,
) -> dict[float, dict[str, str]]:
    """Map each threshold to its precision/recall/f1 column names."""
    columns: dict[float, dict[str, str]] = {}
    for column in history_df.columns:
        for key, metric_re in (
            ("precision", PRECISION_RE),
            ("recall", RECALL_RE),
            ("f1", F1_RE),
        ):
            match = metric_re.match(column)
            if match is None:
                continue
            threshold = float(match.group("thr").replace(",", "."))
            columns.setdefault(threshold, {})[key] = column
    return columns


def _last_value(history_df: pd.DataFrame, column: str) -> float | None:
    """Return the last non-NaN value of a column, or None if unavailable."""
    values = history_df[column].dropna()
    if values.empty:
        return None
    return float(values.iloc[-1])


def _points_last(
    history_df: pd.DataFrame, threshold_columns: dict[float, dict[str, str]]
) -> list[tuple[float, float, float]]:
    """Get (threshold, recall, precision) using each metric's last value."""
    points = []
    for threshold, cols in threshold_columns.items():
        if "precision" not in cols or "recall" not in cols:
            continue
        precision = _last_value(history_df, cols["precision"])
        recall = _last_value(history_df, cols["recall"])
        if precision is None or recall is None:
            continue
        points.append((threshold, recall, precision))
    return points


def _best_epoch_scores(
    history_df: pd.DataFrame,
    threshold_columns: dict[float, dict[str, str]],
    best_metric: str,
) -> pd.Series | None:
    """Per-row score used to pick the best epoch, or None if unavailable.

    For "f1" this is the max F1 across thresholds; for "auroc" it is the
    ``val_binary/auroc`` column.
    """
    if best_metric == "auroc":
        if AUROC_COLUMN not in history_df:
            return None
        return history_df[AUROC_COLUMN]

    f1_cols = [cols["f1"] for cols in threshold_columns.values() if "f1" in cols]
    if not f1_cols:
        return None
    return history_df[f1_cols].max(axis=1)


def _points_best(
    history_df: pd.DataFrame,
    threshold_columns: dict[float, dict[str, str]],
    best_metric: str,
) -> tuple[int | None, list[tuple[float, float, float]]]:
    """Get points from the epoch with the highest ``best_metric`` score.

    Returns (epoch, points).
    """
    if EPOCH_COLUMN not in history_df:
        return None, []

    scores = _best_epoch_scores(history_df, threshold_columns, best_metric)
    if scores is None or scores.dropna().empty:
        return None, []
    best_idx = scores.idxmax()
    best_row = history_df.loc[best_idx]
    epoch = None if pd.isna(best_row[EPOCH_COLUMN]) else int(best_row[EPOCH_COLUMN])

    points = []
    for threshold, cols in threshold_columns.items():
        if "precision" not in cols or "recall" not in cols:
            continue
        precision = best_row[cols["precision"]]
        recall = best_row[cols["recall"]]
        if pd.isna(precision) or pd.isna(recall):
            continue
        points.append((threshold, float(recall), float(precision)))
    return epoch, points


def get_pr_points(
    entity_name: str,
    project_name: str,
    mode: str,
    best_metric: str,
) -> dict[str, list[tuple[float, float, float]]]:
    """Get (threshold, recall, precision) points per run from W&B history.

    Args:
        entity_name: the W&B entity containing the runs.
        project_name: the W&B project containing the runs.
        mode: "last" or "best" (see module docstring).
        best_metric: in "best" mode, the metric used to pick the epoch, either
            "f1" (max F1 across thresholds) or "auroc" (val_binary/auroc).

    Returns:
        a map from run name to a list of (threshold, recall, precision) tuples
        sorted by recall.
    """
    api = wandb.Api()

    results: dict[str, list[tuple[float, float, float]]] = {}
    for run in api.runs(f"{entity_name}/{project_name}"):
        print(f"getting history for {run.name}")
        history_df = run.history()
        threshold_columns = _threshold_columns(history_df)

        if mode == "best":
            epoch, points = _points_best(history_df, threshold_columns, best_metric)
            if points:
                print(f"  {run.name}: using epoch {epoch} (best {best_metric})")
        else:
            points = _points_last(history_df, threshold_columns)

        if not points:
            print(f"warning: run {run.name} has no precision/recall columns")
            continue

        points.sort(key=lambda p: p[1])  # sort by recall for a clean curve
        results[run.name] = points

    return results


def plot_pr_curves(
    points_per_run: dict[str, list[tuple[float, float, float]]],
    output_path: str,
) -> None:
    """Plot one PR curve per run and save to a PNG."""
    fig, ax = plt.subplots(figsize=(10, 8))
    for run_name in sorted(points_per_run):
        points = points_per_run[run_name]
        recalls = [p[1] for p in points]
        precisions = [p[2] for p in points]
        ax.plot(recalls, precisions, marker="o", markersize=3, label=run_name)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Precision-Recall Curves")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot precision-recall curves for W&B runs in a project"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=DEFAULT_WANDB_ENTITY,
        help="W&B entity",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="W&B project",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pr_curve.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="last",
        choices=["last", "best"],
        help=(
            "Which epoch to read precision/recall from: 'last' (last recorded) "
            "or 'best' (epoch with the highest --best-metric score)"
        ),
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default="f1",
        choices=["f1", "auroc"],
        help=(
            "In 'best' mode, the metric used to pick the epoch: 'f1' (max F1 "
            "across thresholds) or 'auroc' (val_binary/auroc)"
        ),
    )
    args = parser.parse_args()

    points_per_run = get_pr_points(
        args.entity, args.project, mode=args.mode, best_metric=args.best_metric
    )
    if not points_per_run:
        print("no runs with usable precision/recall history found")
    else:
        plot_pr_curves(points_per_run, args.output)
