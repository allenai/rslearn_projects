"""Run the full scan -> collapse -> text report pipeline for one dated report.

This is the command executed inside the weekly Beaker job (see
``launch_weekly_job.py``). For a given ``out_dir`` and ``date`` it writes::

    {out_dir}/{date}_usage.jsonl      # disk_usage.py scan (one line per directory)
    {out_dir}/{date}_collapsed.json   # collapse.py bounded tree
    {out_dir}/{date}_report.txt       # text_report.py listing

The scan is the only slow stage (hours on a full weka bucket). The Beaker job is
preemptible with ``auto_resume``, so this wrapper is written to be re-run safely:

- ``disk_usage.py`` checkpoints its frontier to ``{date}_usage.jsonl.ckpt`` and
  resumes from it if the job is restarted mid-scan.
- Once the scan completes, a ``{date}_usage.jsonl.done`` marker is written. A
  restart after that point skips the scan entirely and only re-runs the cheap
  collapse and report stages (which are always re-run since they take seconds).

Each stage is executed as a subprocess of the corresponding script rather than
imported, so the scanner's ``multiprocessing`` workers are unaffected by the
forkserver preload that ``rslp.main`` configures for training workloads.
"""

import os
import subprocess  # nosec
import sys

from rslp.log_utils import get_logger

logger = get_logger(__name__)


def _run_stage(name: str, args: list[str]) -> None:
    """Run one pipeline stage as ``python -m rslp.weka_disk_usage.<name> <args>``."""
    cmd = [sys.executable, "-m", f"rslp.weka_disk_usage.{name}"] + args
    logger.info("running %s: %s", name, " ".join(cmd))
    subprocess.run(cmd, check=True)  # nosec


def weekly_report(
    out_dir: str,
    date: str,
    root: str = "/weka/dfive-default",
    workers: int = 64,
    max_depth: int = 10,
    collapse_gb: float = 10.0,
    max_children: int = 100,
    min_gb: float = 1000.0,
) -> None:
    """Scan ``root`` and write the dated usage JSONL, collapsed JSON, and text report.

    Args:
        out_dir: directory to write outputs to; created if needed.
        date: date prefix for the output files, e.g. "20260925".
        root: root directory to scan.
        workers: number of scanner worker processes.
        max_depth: collapse.py --max_depth.
        collapse_gb: collapse.py --collapse_gb.
        max_children: collapse.py --max_children.
        min_gb: text_report.py --min_gb.
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, date)
    usage_path = f"{prefix}_usage.jsonl"
    done_path = f"{usage_path}.done"
    collapsed_path = f"{prefix}_collapsed.json"
    report_path = f"{prefix}_report.txt"

    if os.path.exists(done_path):
        logger.info("scan already completed (%s exists), skipping scan", done_path)
    else:
        _run_stage(
            "disk_usage",
            [
                "--root",
                root,
                "--output",
                usage_path,
                "--workers",
                str(workers),
            ],
        )
        with open(done_path, "w") as f:
            f.write("")
        logger.info("scan complete, wrote marker %s", done_path)

    _run_stage(
        "collapse",
        [
            "--input",
            usage_path,
            "--output",
            collapsed_path,
            "--max_depth",
            str(max_depth),
            "--collapse_gb",
            str(collapse_gb),
            "--max_children",
            str(max_children),
        ],
    )
    _run_stage(
        "text_report",
        [
            "--input",
            collapsed_path,
            "--output",
            report_path,
            "--min_gb",
            str(min_gb),
        ],
    )
    logger.info("wrote %s, %s, %s", usage_path, collapsed_path, report_path)
