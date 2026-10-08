"""Step 5: publish a retrained classifier into the served Docker image.

1. upload the checkpoint from ``$RSLP_PREFIX/projects/<project>/<run>/best.ckpt`` to GCS;
2. patch the Dockerfile's classifier download and ``config.py``'s CLASSIFY_MODEL_CONFIG;
3. build and push the image.

Only prints the plan by default; ``--upload``, ``--patch``, ``--build`` and ``--push``
perform each step.

Usage:
    python -m rslp.landsat_vessels.feedback.publish \
        --run-name olmoearth_base_layerdecay_<date> \
        --config data/landsat_vessels/config_classifier_<date>.yaml --upload --patch
"""

import argparse
import os
import subprocess  # nosec B404 - runs gsutil/docker from internally-built argv
from pathlib import Path

from rslp.landsat_vessels.feedback import config


def _run(cmd: list[str], do_it: bool) -> None:
    """Print a command, and run it when do_it is set."""
    print(f"  $ {' '.join(cmd)}")
    if do_it:
        subprocess.run(cmd, check=True)  # nosec B603 - cmd built internally, no shell


def _patch_file(path: Path, old: str, new: str, do_it: bool) -> None:
    """Replace ``old`` with ``new`` in a file, reporting the count."""
    text = path.read_text()
    count = text.count(old)
    if count == 0:
        print(f"  WARNING: pattern not found in {path.name}: {old!r}")
        return
    print(f"  {path}: {count}x {old!r} -> {new!r}")
    if do_it:
        path.write_text(text.replace(old, new))


def main() -> None:
    """Plan (or perform) publishing a retrained classifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True, help="new (retrained) run_name")
    parser.add_argument(
        "--project-name",
        default=config.CLASSIFIER_PROJECT_NAME,
        help="training project_name (default: %(default)s)",
    )
    parser.add_argument(
        "--old-run-name",
        default=config.DEPLOYED_RUN_NAME,
        help="run_name currently in the Dockerfile (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="new classifier config to point config.py at",
    )
    parser.add_argument(
        "--rslp-prefix",
        default=os.environ.get("RSLP_PREFIX", "/weka/dfive-default/rslearn-eai"),
        help="where trained checkpoints live (default: $RSLP_PREFIX or weka)",
    )
    parser.add_argument("--upload", action="store_true", help="do the gsutil upload")
    parser.add_argument(
        "--patch", action="store_true", help="edit Dockerfile/config.py"
    )
    parser.add_argument("--build", action="store_true", help="run docker compose build")
    parser.add_argument("--push", action="store_true", help="run docker push")
    parser.add_argument(
        "--image",
        default="landsat_vessels:latest",
        help="image tag to build/push",
    )
    args = parser.parse_args()

    ckpt = (
        Path(args.rslp_prefix)
        / "projects"
        / args.project_name
        / args.run_name
        / "best.ckpt"
    )
    gcs_uri = config.gcs_checkpoint_uri(args.project_name, args.run_name)
    gcs_url = config.gcs_checkpoint_url(args.project_name, args.run_name)

    print("== 1. upload checkpoint ==")
    if ckpt.exists():
        size_mb = ckpt.stat().st_size / 1e6
        print(f"  source: {ckpt} ({size_mb:.0f} MB)")
    else:
        print(f"  WARNING: checkpoint not found: {ckpt} (has the retrain finished?)")
    print(f"  dest:   {gcs_uri}")
    _run(["gsutil", "cp", str(ckpt), gcs_uri], args.upload)

    print("\n== 2. patch Dockerfile + config.py ==")
    old_path = f"{args.project_name}/{args.old_run_name}"
    new_path = f"{args.project_name}/{args.run_name}"
    dockerfile = Path(config.DOCKERFILE_REL)
    _patch_file(dockerfile, old_path, new_path, args.patch)
    if args.config:
        config_py = Path(config.CONFIG_PY_REL)
        old_cfg = Path(config.BASE_CLASSIFIER_CONFIG).name
        new_cfg = Path(args.config).name
        if new_cfg != old_cfg:
            _patch_file(config_py, old_cfg, new_cfg, args.patch)
        else:
            print(f"  config.py already points at {new_cfg}")
    else:
        print(
            "  (no --config given; make sure config.py's CLASSIFY_MODEL_CONFIG run_name "
            f"matches {args.run_name})"
        )
    print(f"  container will wget: {gcs_url}")

    print("\n== 3. build & push image ==")
    compose_dir = Path("rslp/landsat_vessels")
    print(f"  (cwd: {compose_dir})")
    _run(
        ["docker", "compose", "-f", str(compose_dir / "docker-compose.yaml"), "build"],
        args.build,
    )
    _run(["docker", "push", args.image], args.push)

    print("\n== 4. redeploy ==")
    print("  smoke-test the container endpoint, then redeploy the image to Skylight")
    print("  integration (https://app-int.skylight.earth) and bump the model version.")
    if not (args.upload or args.patch or args.build or args.push):
        print("\n(plan only -- re-run with --upload --patch [--build --push] to apply)")


if __name__ == "__main__":
    main()
