"""Launch Canada NBAC test jobs from a training config and checkpoint.

This script creates temporary test configs (one per split), launches test jobs
through `rslp.main common beaker_train`, then removes the temporary configs.

Usage:
    cd /weka/dfive-default/hadriens/rslearn_projects
    source ./.venv/bin/activate

    # Using --run-name (auto-resolve config from generated_configs/ subdirs):
    python -m rslp.wildfire.Canada_nbac.launch_test_jobs \
        --run-name lf_mixing_d768_lr1e6_gdo01_deep_s2do05_pfnorm_pfdo01_era5warm_ep40 \
        --batch-size 128 \
        --splits val

    # Val split with explicit config path (sweep F1 thresholds, report best):
    python -m rslp.wildfire.Canada_nbac.launch_test_jobs \
        --train-config /path/to/train_config.yaml \
        --batch-size 16 \
        --splits val

    # Test splits with fixed F1 threshold from val:
    python -m rslp.wildfire.Canada_nbac.launch_test_jobs \
        --run-name lf_mixing_d768_lr1e6_gdo01_deep_s2do05_pfnorm_pfdo01_era5warm_ep40 \
        --batch-size 16 \
        --splits test test_hard \
        --f1-threshold 0.42

    # Optional explicit checkpoint override:
    python -m rslp.wildfire.Canada_nbac.launch_test_jobs \
        --train-config /path/to/train_config.yaml \
        --batch-size 16 \
        --ckpt-path /path/to/checkpoints/epoch=14-step=73155.ckpt
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import subprocess  # nosec B404
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IMAGE_NAME = "hadriens/rslpomp_hspec_260401_metricswriting"
DEFAULT_CLUSTERS = ["ai2/neptune"]
DEFAULT_PRIORITY = "high"
DEFAULT_GPUS = 1
DEFAULT_SPLITS = ("val", "test", "test_hard")
DEFAULT_WEKA_MOUNT = {
    "bucket_name": "dfive-default",
    "mount_path": "/weka/dfive-default",
}
PROJECT_DATA_ROOT = Path("/weka/dfive-default/hadriens/project_data/projects")
BEAKER_NAME_MAX = 128
# beaker_train builds: "{project_id}_{experiment_id}_{uuid8}" — 10 extra chars.
_BEAKER_NAME_OVERHEAD = 10

REPO_ROOT = Path(__file__).resolve().parents[3]
GENERATED_CONFIGS_ROOT = (
    REPO_ROOT / "data" / "helios" / "wildfire" / "CanadaNbac" / "generated_configs"
)
TEMP_CONFIG_DIR = GENERATED_CONFIGS_ROOT / "_tmp_test_launch"

# Regex matching the timestamp suffix in config filenames: _YYYYMMDD_HHMMSS.yaml
_TIMESTAMP_RE = re.compile(r"_(\d{8}_\d{6})\.yaml$")


def resolve_existing_path(path_arg: str) -> Path:
    """Resolve an input path and ensure it exists."""
    raw = Path(path_arg).expanduser()
    if raw.is_absolute():
        resolved = raw
        if not resolved.exists():
            raise FileNotFoundError(f"Path does not exist: {resolved}")
        return resolved

    candidates = [(Path.cwd() / raw).resolve(), (REPO_ROOT / raw).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Path does not exist: {path_arg} (checked relative to cwd and {REPO_ROOT})"
    )


def resolve_config_from_run_name(
    run_name: str, skip_confirmation: bool = False
) -> Path:
    """Find the latest config matching *run_name* across all subdirs of GENERATED_CONFIGS_ROOT.

    The naming convention is ``{run_name}_{YYYYMMDD}_{HHMMSS}.yaml``.
    Subdirectories whose name starts with ``_`` (e.g. ``_tmp_test_launch``) are
    skipped.

    When multiple candidates exist the latest one (by embedded timestamp) is
    selected and — unless *skip_confirmation* is True — the user is asked to
    confirm.
    """
    if not GENERATED_CONFIGS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Generated configs root does not exist: {GENERATED_CONFIGS_ROOT}"
        )

    candidates: list[tuple[str, Path]] = []  # (timestamp_str, path)
    for subdir in sorted(GENERATED_CONFIGS_ROOT.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        for cfg_file in subdir.glob("*.yaml"):
            m = _TIMESTAMP_RE.search(cfg_file.name)
            if m is None:
                continue
            # Derive the run name by stripping the timestamp suffix.
            file_run_name = cfg_file.name[: m.start()]
            if file_run_name == run_name:
                candidates.append((m.group(1), cfg_file))

    if not candidates:
        raise FileNotFoundError(
            f"No config found for run name '{run_name}' under {GENERATED_CONFIGS_ROOT}"
        )

    # Sort by timestamp string (lexicographic == chronological for YYYYMMDD_HHMMSS).
    candidates.sort(key=lambda t: t[0])
    latest_ts, latest_path = candidates[-1]

    if len(candidates) > 1:
        print(f"\nFound {len(candidates)} config(s) matching run name '{run_name}':")
        for ts, p in candidates:
            marker = " <-- latest" if p == latest_path else ""
            print(f"  [{ts}] {p.parent.name}/{p.name}{marker}")

        if not skip_confirmation:
            response = input(f"\nUse the latest config ({latest_path.name})? [Y/n] ")
            if response.strip().lower() == "n":
                print("Aborted.")
                sys.exit(0)
    else:
        print(f"\nResolved config for '{run_name}':")
        print(f"  {latest_path}")

    return latest_path


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected a YAML mapping in {path}, got {type(cfg)!r}")
    return cfg


def infer_run_name_from_checkpoint(ckpt_path: Path) -> str:
    """Infer run name from checkpoint path.

    Expected shape:
        .../<run_name>/checkpoints/<checkpoint_file>.ckpt
    """
    parent = ckpt_path.parent
    if parent.name == "checkpoints" and parent.parent.name:
        return parent.parent.name
    return parent.name or ckpt_path.stem


def _checkpoint_rank(ckpt_path: Path) -> tuple[int, int, float]:
    """Rank checkpoint candidates so newer epoch/step wins when available."""
    match = re.search(r"epoch=(\d+)-step=(\d+)\.ckpt$", ckpt_path.name)
    mtime = ckpt_path.stat().st_mtime
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        return (epoch, step, mtime)
    # Fallback for non-standard names: prefer latest mtime.
    return (-1, -1, mtime)


def get_project_and_run_names(train_cfg: dict[str, Any]) -> tuple[str, str]:
    """Return project/run identifiers from either config schema.

    Supports both the newer NBAC schema:
      - ``project_name`` / ``run_name``

    and the older schema still present in some archived/generated configs:
      - ``rslp_project`` / ``rslp_experiment``
    """
    project = train_cfg.get("project_name")
    if not isinstance(project, str) or not project:
        project = train_cfg.get("rslp_project")

    run_name = train_cfg.get("run_name")
    if not isinstance(run_name, str) or not run_name:
        run_name = train_cfg.get("rslp_experiment")

    if not isinstance(project, str) or not project:
        raise ValueError(
            "Cannot resolve project identifier: expected `project_name` or `rslp_project`"
        )
    if not isinstance(run_name, str) or not run_name:
        raise ValueError(
            "Cannot resolve run identifier: expected `run_name` or `rslp_experiment`"
        )

    return project, run_name


def resolve_checkpoint_from_config(train_cfg: dict[str, Any]) -> Path:
    """Resolve checkpoint automatically from either old or new config identifiers.

    The new checkpoint system (ManagedBestLastCheckpoint) saves best.ckpt and
    last.ckpt directly in {management_dir}/{project_name}/{run_name}/.
    Falls back to the old {project}/{experiment}/checkpoints/ layout for
    backward compatibility with existing runs.
    """
    project, experiment = get_project_and_run_names(train_cfg)

    # New layout: checkpoints live directly in the project dir
    project_dir = PROJECT_DATA_ROOT / project / experiment
    # Old layout: checkpoints in a subdirectory
    old_checkpoints_dir = project_dir / "checkpoints"

    # Prefer best.ckpt in the new layout
    best_ckpt = project_dir / "best.ckpt"
    if best_ckpt.is_file():
        return best_ckpt

    # Fall back to scanning for non-last checkpoints (old or new layout)
    for search_dir in [project_dir, old_checkpoints_dir]:
        if not search_dir.is_dir():
            continue
        all_ckpts = [p for p in search_dir.glob("*.ckpt") if p.is_file()]
        non_last_ckpts = [
            p
            for p in all_ckpts
            if p.name != "last.ckpt" and not p.name.startswith("last-")
        ]
        if non_last_ckpts:
            return max(non_last_ckpts, key=_checkpoint_rank)

    raise FileNotFoundError(
        f"Cannot auto-resolve checkpoint: no best/non-last checkpoint found in "
        f"{project_dir} or {old_checkpoints_dir}"
    )


def make_test_config(
    train_cfg: dict[str, Any],
    split: str,
    batch_size: int,
    experiment_id: str,
    f1_threshold: float | None = None,
    center_crop: bool = False,
) -> dict[str, Any]:
    """Create one test config for a split.

    Args:
        train_cfg: The base training config dict.
        split: Dataset split name (e.g. "val", "test", "test_hard").
        batch_size: Batch size to write into the config.
        experiment_id: Experiment identifier for this test run.
        f1_threshold: If provided, use this single F1 threshold instead of
            sweeping.  Typically obtained from the best threshold reported on
            the val split, then passed here for test / test_hard evaluation.
        center_crop: If True, use ``center_crop`` instead of ``load_all_crops``
            for the test data config.  Defaults to False (all crops).
    """
    cfg = copy.deepcopy(train_cfg)
    data = cfg.setdefault("data", {})
    if not isinstance(data, dict):
        raise ValueError("Expected `data` to be a mapping in training config")

    init_args = data.setdefault("init_args", {})
    if not isinstance(init_args, dict):
        raise ValueError("Expected `data.init_args` to be a mapping in training config")

    test_cfg = init_args.get("test_config")
    if not isinstance(test_cfg, dict):
        test_cfg = {}
    test_cfg["groups"] = [split]
    test_cfg.pop("load_all_crops", None)
    test_cfg.pop("center_crop", None)
    if center_crop:
        test_cfg["center_crop"] = True
    else:
        test_cfg["load_all_crops"] = True
    init_args["test_config"] = test_cfg
    init_args["batch_size"] = batch_size

    if f1_threshold is not None:
        # Use the single user-provided threshold (e.g. best threshold from val).
        f1_thresholds = [f1_threshold]
        sweep_mode = False
    else:
        # Sweep mode: dense in [0.25, 0.75], coarser tails in [0.1, 0.25) and (0.75, 0.9].
        f1_thresholds = (
            [round(i * 0.025, 3) for i in range(4)]
            + [round(0.1 + i * 0.005, 3) for i in range(60)]
            + [round(0.4 + i * 0.01, 3) for i in range(53)]
            + [round(0.94 + i * 0.02, 2) for i in range(3)]
        )
        sweep_mode = True

    # Ensure F1 metric with threshold reporting is enabled for segmentation tasks.
    task_cfg = init_args.get("task")
    if isinstance(task_cfg, dict):
        task_init_args = task_cfg.get("init_args")
        if isinstance(task_init_args, dict):
            tasks = task_init_args.get("tasks")
            if isinstance(tasks, dict):
                for cur_task in tasks.values():
                    if not isinstance(cur_task, dict):
                        continue
                    class_path = cur_task.get("class_path", "")
                    if not isinstance(class_path, str):
                        continue
                    if not class_path.endswith("SegmentationTask"):
                        raise ValueError(f"Expected SegmentationTask, got {class_path}")
                    cur_init_args = cur_task.get("init_args")
                    if not isinstance(cur_init_args, dict):
                        cur_init_args = {}
                        cur_task["init_args"] = cur_init_args
                    cur_init_args["enable_f1_metric"] = True
                    cur_init_args["report_metric_per_class"] = True
                    cur_init_args["report_best_threshold"] = sweep_mode
                    cur_init_args["f1_metric_thresholds"] = [f1_thresholds]

    # Late-fusion configs with ERA5: ERA5 is stored at 1×1 spatial via
    # NumpyRasterFormat (returns array as-is, ignoring requested bounds).
    # Without a resolution_factor, AllCropsDataset crops ERA5 at window-
    # resolution coordinates, overshooting the tiny tensor and causing a
    # tensor mismatch across batch items.  The denominator must exceed the
    # largest window dimension (in pixels) so that every crop offset
    # floor-divides to 0, mapping all crops to the same (0,0,1,1) slice.
    # 10000 px at 10 m ≈ 100 km — safely larger than any window.
    inputs = init_args.get("inputs", {})
    if isinstance(inputs, dict) and "era5_daily" in inputs:
        era5_cfg = inputs["era5_daily"]
        if isinstance(era5_cfg, dict) and "resolution_factor" not in era5_cfg:
            era5_cfg["resolution_factor"] = {
                "class_path": "rslearn.utils.geometry.ResolutionFactor",
                "init_args": {"numerator": 1, "denominator": 260},
            }

    # Disable model management auto-checkpoint-loading since we provide
    # the checkpoint explicitly via --ckpt_path.
    cfg.pop("management_dir", None)

    project_name, _ = get_project_and_run_names(train_cfg)
    cfg.pop("rslp_project", None)
    cfg.pop("rslp_experiment", None)
    cfg["project_name"] = project_name
    cfg["run_name"] = experiment_id
    return cfg


def write_temp_config(run_name: str, split: str, cfg: dict[str, Any]) -> Path:
    """Write a temp config to disk, flush to Weka, and verify readback."""
    TEMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{run_name}_TESTMODE_{split}_{timestamp}.yaml"
    path = TEMP_CONFIG_DIR / fname

    yaml_str = yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)
    with open(path, "w") as f:
        f.write(yaml_str)

    # Force flush on Weka to reduce stale-read race conditions.
    os.sync()

    with open(path) as f:
        readback = f.read()
    if readback != yaml_str:
        raise RuntimeError(f"Config verification FAILED for {path}")

    return path


def to_repo_relative(path: Path) -> str:
    """Convert an absolute path to a path relative to rslearn_projects root."""
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def build_launch_cmd(
    config_path: str,
    checkpoint_path: str,
    experiment_id: str,
    image_name: str,
    clusters: list[str],
    priority: str,
    gpus: int,
    weka_mount: dict[str, str],
) -> list[str]:
    """Build the beaker_train launch command for test mode."""
    extra_args = [
        f"--ckpt_path={checkpoint_path}",
        "--model.init_args.write_test_metrics=true",
    ]

    cmd = [
        "python",
        "-m",
        "rslp.main",
        "common",
        "beaker_train",
        "--image_name",
        image_name,
    ]
    cmd.extend([f"--cluster+={cluster}" for cluster in clusters])
    cmd.extend(
        [
            "--experiment_id",
            experiment_id,
            "--gpus",
            str(gpus),
            "--config_path",
            config_path,
            f"--weka_mounts+={json.dumps(weka_mount)}",
            "--priority",
            priority,
            "--mode",
            "test",
            "--extra_args",
            json.dumps(extra_args),
        ]
    )
    return cmd


def launch_one(
    split: str, experiment_id: str, config_path: str, cmd: list[str], dry_run: bool
) -> None:
    """Print launch info and optionally execute the command."""
    print(f"\n{'='*80}")
    print(f"  Split:       {split}")
    print(f"  Experiment:  {experiment_id}")
    print(f"  Config:      {config_path}")
    print(f"  Command:     {' '.join(shlex.quote(x) for x in cmd)}")
    print(f"{'='*80}")
    if dry_run:
        print("  [DRY RUN] Skipping launch.")
        return
    subprocess.check_call(cmd, cwd=str(REPO_ROOT))  # nosec B603
    print(f"  Launched: {experiment_id}")


def remove_file(path: Path) -> None:
    """Best-effort file removal."""
    try:
        path.unlink()
    except OSError:
        pass


def main() -> None:
    """Generate temporary test configs and launch Canada NBAC evaluation jobs."""
    parser = argparse.ArgumentParser(
        description="Launch Canada NBAC test jobs for val/test/test_hard splits"
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--train-config",
        default=None,
        help="Training config YAML to use as source for temporary test configs.",
    )
    config_group.add_argument(
        "--run-name",
        default=None,
        help=(
            "Run/experiment name (without timestamp) to auto-resolve the training "
            "config from generated_configs/ subdirectories (S2, latefusion, era5d, …). "
            "The latest timestamped config matching the name is selected."
        ),
    )
    parser.add_argument(
        "--ckpt-path",
        default=None,
        help=(
            "Optional checkpoint path to evaluate. If omitted, the script auto-resolves "
            "best.ckpt (or the best non-last checkpoint) from "
            "/weka/dfive-default/hadriens/project_data/projects/{project_name}/{run_name}/."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size to write into temporary test configs.",
    )
    parser.add_argument(
        "--image-name",
        default=DEFAULT_IMAGE_NAME,
        help=f"Beaker image name (default: {DEFAULT_IMAGE_NAME}).",
    )
    parser.add_argument(
        "--cluster",
        nargs="+",
        default=DEFAULT_CLUSTERS,
        help=f"One or more Beaker clusters (default: {DEFAULT_CLUSTERS[0]}).",
    )
    parser.add_argument(
        "--priority",
        default=DEFAULT_PRIORITY,
        help=f"Beaker priority (default: {DEFAULT_PRIORITY}).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=DEFAULT_GPUS,
        help=f"GPUs per job (default: {DEFAULT_GPUS}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to launch (default: val test test_hard).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write temp configs and print commands without launching.",
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip the interactive launch confirmation prompt.",
    )
    parser.add_argument(
        "--yes",
        dest="skip_confirmation",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--f1-threshold",
        type=float,
        default=None,
        help=(
            "Fixed F1 threshold to use instead of sweeping. "
            "Pass the best threshold found on the val split when evaluating "
            "test / test_hard splits."
        ),
    )
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help=(
            "Use center_crop=true instead of load_all_crops=true for the test "
            "data config. By default, load_all_crops=true is always set."
        ),
    )
    parser.add_argument(
        "--keep-configs",
        action="store_true",
        help="Do not delete generated temporary configs.",
    )
    args = parser.parse_args()

    invalid_splits = [s for s in args.splits if s not in set(DEFAULT_SPLITS)]
    if invalid_splits:
        print(
            f"Unsupported split(s): {invalid_splits}. Allowed: {list(DEFAULT_SPLITS)}"
        )
        sys.exit(1)

    if args.run_name:
        train_config_path = resolve_config_from_run_name(
            args.run_name, skip_confirmation=args.skip_confirmation
        )
    else:
        train_config_path = resolve_existing_path(args.train_config)
    train_cfg = load_yaml_config(train_config_path)
    cfg_run_name = train_cfg.get("run_name")

    if args.ckpt_path:
        ckpt_path = resolve_existing_path(args.ckpt_path)
        ckpt_source = "provided"
    else:
        ckpt_path = resolve_checkpoint_from_config(train_cfg)
        ckpt_source = "auto-resolved from training config"

    run_name = infer_run_name_from_checkpoint(ckpt_path)

    if isinstance(cfg_run_name, str) and cfg_run_name and cfg_run_name != run_name:
        print(
            "WARNING: training config run_name differs from checkpoint-derived run name:"
        )
        print(f"  config run_name: {cfg_run_name}")
        print(f"  checkpoint run name:    {run_name}")
        print("  using checkpoint-derived run name for eval experiment IDs")

    jobs = [(split, f"eval-{split}_{run_name}") for split in args.splits]

    print(f"Will launch {len(jobs)} test job(s):")
    print(f"  Train config: {train_config_path}")
    print(f"  Checkpoint:   {ckpt_path} ({ckpt_source})")
    print(f"  Batch size:   {args.batch_size}")
    if args.f1_threshold is not None:
        print(f"  F1 threshold: {args.f1_threshold} (fixed)")
    else:
        print("  F1 threshold: sweep (will report best)")
    print(f"  Image:        {args.image_name}")
    print(f"  Cluster(s):   {args.cluster}")
    print(f"  Priority:     {args.priority}")
    print(f"  GPUs:         {args.gpus}")
    print(f"  Crop mode:    {'center_crop' if args.center_crop else 'load_all_crops'}")
    print("  Split jobs:")
    for split, experiment_id in jobs:
        print(f"    - {split} -> {experiment_id}")

    if not args.dry_run and not args.skip_confirmation:
        response = input("\nProceed with launching these test job(s)? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)

    launched = 0
    kept_configs: list[Path] = []
    for split, experiment_id in jobs:
        cfg = make_test_config(
            train_cfg,
            split,
            args.batch_size,
            experiment_id,
            f1_threshold=args.f1_threshold,
            center_crop=args.center_crop,
        )
        temp_config = write_temp_config(run_name, split, cfg)
        rel_config_path = to_repo_relative(temp_config)

        cmd = build_launch_cmd(
            config_path=rel_config_path,
            checkpoint_path=str(ckpt_path),
            experiment_id=experiment_id,
            image_name=args.image_name,
            clusters=args.cluster,
            priority=args.priority,
            gpus=args.gpus,
            weka_mount=DEFAULT_WEKA_MOUNT,
        )

        launch_succeeded = False
        try:
            launch_one(
                split=split,
                experiment_id=experiment_id,
                config_path=rel_config_path,
                cmd=cmd,
                dry_run=args.dry_run,
            )
            launch_succeeded = True
            launched += 1
        finally:
            if args.keep_configs:
                kept_configs.append(temp_config)
            elif args.dry_run or launch_succeeded:
                remove_file(temp_config)

    action = "Planned" if args.dry_run else "Submitted"
    print(f"\n{'='*80}")
    print(f"Done. {action} {launched}/{len(jobs)} test job(s).")
    if args.keep_configs:
        print(f"Kept {len(kept_configs)} temporary config file(s).")
        for p in kept_configs:
            print(f"  - {p}")
    else:
        print("Temporary config files removed after launch.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

# cd /weka/dfive-default/hadriens/rslearn_projects
# source ./.venv/bin/activate

# Preview commands without launching (auto-resolve config + checkpoint via --run-name):
# python -m rslp.wildfire.Canada_nbac.launch_test_jobs \
#   --run-name s2_mts_atp_lr1e5_cs5_bs16_aug_tdrop05 \
#   --batch-size 16 \
#   --dry-run

# Same thing using explicit --train-config:
# python -m rslp.wildfire.Canada_nbac.launch_test_jobs \
#   --train-config /weka/dfive-default/hadriens/rslearn_projects/data/helios/wildfire/CanadaNbac/generated_configs/S2/s2_mts_atp_lr1e5_cs5_bs16_aug_tdrop05_20260303_175549.yaml \
#   --batch-size 16 \
#   --dry-run

# Launch all default splits (val, test, test_hard), auto-resolve checkpoint:
# python -m rslp.wildfire.Canada_nbac.launch_test_jobs \
#   --run-name s2_mts_atp_lr1e5_cs5_bs16_aug_tdrop05 \
#   --batch-size 16 \
#   --skip-confirmation

# Launch with an explicit checkpoint override:
# python -m rslp.wildfire.Canada_nbac.launch_test_jobs \
#   --run-name s2_mts_atp_lr1e5_cs5_bs16_aug_tdrop05 \
#   --batch-size 16 \
#   --ckpt-path /weka/dfive-default/hadriens/project_data/projects/20260321_wf_nbac_newsample/s2_mts_atp_lr1e5_cs5_bs16_aug_tdrop05/checkpoints/epoch=14-step=73155.ckpt \
#   --skip-confirmation

# Launch with custom cluster/image/priority using --run-name:
# Example:
# cd /weka/dfive-default/hadriens/rslearn_projects
# source ./.venv/bin/activate
# python -m rslp.wildfire.Canada_nbac.launch_test_jobs \
#   --run-name base_b3_focal_pos_a09_g30_rc_v3 \
#   --batch-size 128 \
#   --splits val test test_hard \
#   --cluster ai2/neptune \
#   --image-name hadriens/rslpomp_hspec_260408_wbfix \
#   --priority high \
#   --f1-threshold 0.5
