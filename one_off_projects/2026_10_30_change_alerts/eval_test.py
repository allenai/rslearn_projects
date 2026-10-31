"""Evaluate change alert runs on the test scenarios and write a CSV.

Models are either round 3 runs (a RUNS_BY_NAME key, e.g. history_pool_s0) or round 1/2
history runs ("{project}/{run}", e.g. 2026_10_12_change_alerts/fixed4_history_pool),
whose checkpoints are tested with the round 3 sampler (see
make_model_configs.make_old_run_config).

There are two test sets, each with scenarios E (days after the change):

- round3: the round 3 test split, prediction windows covering the 4 weeks before E days
  after the change, with the quarterly mosaics ending at the window start.
- round2: the round 2 test_history datasets, where the time series has the latest 4
  weekly mosaics within 63 days before E days after the change, and quarterly mosaics
  anchored 27 days after the change. Testing round 2 runs on it reproduces the round 2
  results, and testing round 3 runs on it compares them on the same inputs.

For each model, test set, and scenario, this runs `rslearn model test` in-process with
the best checkpoint of the run, and records the test metrics:

- test_category/balanced_accuracy: mean per-category recall (the headline metric).
- test_category/accuracy: pixel accuracy of the category.
- test_category/change_auroc: AUROC of change versus no change.
- test_timestep/accuracy and test_timestep/within1: accuracy of the predicted
  timestep at change pixels, exactly or within one timestep.

RSLP_PREFIX must be set (the checkpoints are under ${RSLP_PREFIX}/projects) unless
--management_dir is passed.
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

from common import ROUND2_END_DAYS, SOURCE_DATASETS, TEST_END_DAYS
from lightning.pytorch import LightningDataModule, LightningModule
from make_model_configs import (
    RUNS_BY_NAME,
    make_old_run_config,
    make_round2_test_config,
    make_test_config,
    make_train_config,
    write_config,
)
from rslearn.arg_parser import RslearnArgumentParser
from rslearn.lightning_cli import RslearnLightningCLI
from rslearn.utils.jsonargparse import init_jsonargparse

# Round 1/2 runs to compare against: the fixed4_history_pool baseline of both rounds,
# and the other round 2 history runs.
OLD_MODELS = [
    "2026_10_07_change_alerts/fixed4_history_pool",
    "2026_10_12_change_alerts/fixed4_history_pool",
    "2026_10_12_change_alerts/rand4_history_pool",
    "2026_10_12_change_alerts/rand1_history_pool",
    "2026_10_12_change_alerts/fixed4_history_breakpoint",
]
TEST_SETS = ["round3", "round2"]


def make_model_config(source: str, model: str) -> dict:
    """Make the training config of a round 3 run or a round 1/2 run."""
    if model in RUNS_BY_NAME:
        return make_train_config(source, RUNS_BY_NAME[model])
    project_name, run = model.split("/")
    return make_old_run_config(source, project_name, f"{source}_{run}")


def run_test(config_fname: Path, extra_args: list[str]) -> dict[str, float]:
    """Run `rslearn model test` in-process and return the logged metrics."""
    cli = RslearnLightningCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule,
        args=["test", "--config", str(config_fname)] + extra_args,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        parser_class=RslearnArgumentParser,
    )
    return {key: float(value) for key, value in cli.trainer.callback_metrics.items()}


def main() -> None:
    """Evaluate the models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=",".join(SOURCE_DATASETS))
    parser.add_argument(
        "--models",
        default=",".join(list(RUNS_BY_NAME) + OLD_MODELS),
        help="Comma-separated round 3 run names and round 1/2 {project}/{run} names",
    )
    parser.add_argument("--test_sets", default=",".join(TEST_SETS))
    parser.add_argument(
        "--end_days",
        default=",".join(str(days) for days in TEST_END_DAYS),
        help="Comma-separated scenarios E; the round2 test set only uses those in "
        "ROUND2_END_DAYS",
    )
    parser.add_argument("--out", required=True, help="Output CSV filename")
    parser.add_argument(
        "--management_dir", default=None, help="Override the management directory"
    )
    parser.add_argument(
        "--extra_args",
        default="",
        help="Extra space-separated arguments for rslearn model test",
    )
    args = parser.parse_args()

    init_jsonargparse()
    extra_args = args.extra_args.split()
    if args.management_dir:
        extra_args += ["--management_dir", args.management_dir]

    rows: list[dict[str, str | float]] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for source in args.sources.split(","):
            for model in args.models.split(","):
                model_config = make_model_config(source, model)
                for test_set in args.test_sets.split(","):
                    for end_days in [int(days) for days in args.end_days.split(",")]:
                        if test_set == "round3":
                            config = make_test_config(model_config, end_days)
                        elif test_set == "round2":
                            if end_days not in ROUND2_END_DAYS:
                                continue
                            config = make_round2_test_config(
                                model_config, source, end_days
                            )
                        else:
                            raise ValueError(f"unknown test set {test_set}")
                        config_fname = Path(tmp_dir) / "config.yaml"
                        write_config(config, config_fname)
                        print(
                            f"testing {source} {model} {test_set} E={end_days}",
                            file=sys.stderr,
                        )
                        metrics = run_test(config_fname, extra_args)
                        rows.append(
                            {
                                "source": source,
                                "model": model,
                                "test_set": test_set,
                                "end_days": end_days,
                            }
                            | metrics
                        )

    key_fields = ["source", "model", "test_set", "end_days"]
    fieldnames = key_fields + sorted(
        {key for row in rows for key in row} - set(key_fields)
    )
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
