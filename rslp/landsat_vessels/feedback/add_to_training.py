"""Step 3: write a dated classifier config that adds a feedback group to training.

Copies a base config, appends the group to ``default_config.groups``, and sets a new
``run_name`` so the retrain gets its own checkpoint.

Usage:
    python -m rslp.landsat_vessels.feedback.add_to_training --group feedback_<date>
        [--base-config <previous cycle's config, to keep its feedback groups>]
"""

import argparse
from pathlib import Path

import yaml

from rslp.landsat_vessels.feedback import config


def main() -> None:
    """Generate a dated classifier config that trains on the new feedback group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", required=True, help="feedback group to add, e.g. feedback_20260928"
    )
    parser.add_argument(
        "--base-config",
        default=config.BASE_CLASSIFIER_CONFIG,
        help="classifier config to derive from (default: %(default)s)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="new run_name (default: <RUN_NAME_STEM>_<group date>)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output path (default: data/landsat_vessels/config_classifier_<date>.yaml)",
    )
    args = parser.parse_args()

    with open(args.base_config) as f:
        data = yaml.safe_load(f)

    default_config = data["data"]["init_args"]["default_config"]
    groups = list(default_config.get("groups", []))
    if args.group not in groups:
        groups.append(args.group)
    default_config["groups"] = groups
    print(f"training groups: {groups}")

    date_tag = args.group.split("_")[-1]
    run_name = args.run_name or f"{config.RUN_NAME_STEM}_{date_tag}"
    data["run_name"] = run_name
    print(f"run_name: {run_name}")

    out_path = Path(
        args.out or f"{config.DATA_DIR_REL}/config_classifier_{date_tag}.yaml"
    )
    with out_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    print(f"\nwrote {out_path}")

    print("\nnext: retrain (1 GPU), then publish:")
    print(f"  rslearn model fit --config {out_path}")
    print(
        f"  python -m rslp.landsat_vessels.feedback.publish "
        f"--run-name {run_name} --config {out_path}"
    )


if __name__ == "__main__":
    main()
