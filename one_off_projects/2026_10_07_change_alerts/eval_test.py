"""Evaluate trained change alert runs on the test scenarios and write a CSV.

For each source dataset, run, and scenario E (the time series ends E days after the
change), this derives the test config from the training config (see
make_model_configs.make_test_config), runs `rslearn model test` in-process with the
best checkpoint of the run, and records the test metrics:

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

from common import SOURCE_DATASETS, TEST_END_DAYS
from lightning.pytorch import LightningDataModule, LightningModule
from make_model_configs import (
    RUNS_BY_NAME,
    make_test_config,
    make_train_config,
    write_config,
)
from rslearn.arg_parser import RslearnArgumentParser
from rslearn.lightning_cli import RslearnLightningCLI
from rslearn.utils.jsonargparse import init_jsonargparse


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
    """Evaluate the runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=",".join(SOURCE_DATASETS))
    parser.add_argument("--runs", default=",".join(RUNS_BY_NAME))
    parser.add_argument(
        "--end_days", default=",".join(str(days) for days in TEST_END_DAYS)
    )
    parser.add_argument("--out", required=True, help="Output CSV filename")
    parser.add_argument(
        "--ds_path",
        default=None,
        help="Override the test dataset path (only for testing with one source and context)",
    )
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
            for run_name in args.runs.split(","):
                run = RUNS_BY_NAME[run_name]
                train_config = make_train_config(source, run)
                for end_days in [int(days) for days in args.end_days.split(",")]:
                    config = make_test_config(
                        train_config, source, run, end_days, ds_path=args.ds_path
                    )
                    config_fname = (
                        Path(tmp_dir) / f"{source}_{run.name}_{end_days}.yaml"
                    )
                    write_config(config, config_fname)
                    print(f"testing {source} {run.name} E={end_days}", file=sys.stderr)
                    metrics = run_test(config_fname, extra_args)
                    rows.append(
                        {"source": source, "run": run.name, "end_days": end_days}
                        | metrics
                    )

    fieldnames = ["source", "run", "end_days"] + sorted(
        {key for row in rows for key in row} - {"source", "run", "end_days"}
    )
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
