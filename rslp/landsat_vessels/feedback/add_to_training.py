"""Stage 3 - fold a feedback group into the classifier's training config.

Reads the deployed classifier config (Run d,
``data/landsat_vessels/config_classifier_20260908d.yaml``), appends the new feedback
group to ``data.init_args.default_config.groups`` and ``train_config`` is inherited from
it, and writes a new dated config with a fresh ``run_name`` so the retrain is a distinct,
publishable checkpoint. The original config is never modified.

Formatting/comments are preserved when ``ruamel.yaml`` is available; otherwise PyYAML is
used (these configs carry no comments, so nothing is lost in practice).

Usage:
    python -m rslp.landsat_vessels.feedback.add_to_training --group feedback_20260911
    # -> data/landsat_vessels/config_classifier_20260911.yaml,
    #    run_name olmoearth_base_layerdecay_20260911

Then launch the retrain (1 GPU) with the printed command.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rslp.landsat_vessels.feedback import config

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> tuple[Any, Any]:
    """Load a YAML config, preferring ruamel to preserve layout."""
    try:
        from ruamel.yaml import YAML

        yaml = YAML()
        yaml.preserve_quotes = True
        with path.open() as f:
            return yaml.load(f), yaml
    except ImportError:
        import yaml as pyyaml

        with path.open() as f:
            return pyyaml.safe_load(f), None


def _dump_yaml(data: Any, yaml: Any, path: Path) -> None:
    """Write a config back, using whichever loader produced it."""
    if yaml is not None:  # ruamel
        with path.open("w") as f:
            yaml.dump(data, f)
    else:
        import yaml as pyyaml

        with path.open("w") as f:
            pyyaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def main() -> None:
    """Generate a dated classifier config that trains on the new feedback group."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", required=True, help="feedback group to add, e.g. feedback_20260911"
    )
    parser.add_argument(
        "--base-config",
        default=str(REPO_ROOT / config.BASE_CLASSIFIER_CONFIG),
        help="classifier config to derive from (default: deployed Run d)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="new run_name (default: <RUN_NAME_STEM>_<group date>, e.g. "
        "olmoearth_base_layerdecay_20260911)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output config path (default: data/landsat_vessels/config_classifier_<date>.yaml)",
    )
    args = parser.parse_args()

    base_path = Path(args.base_config)
    data, yaml = _load_yaml(base_path)

    default_config = data["data"]["init_args"]["default_config"]
    groups = list(default_config.get("groups", []))
    if args.group in groups:
        print(f"note: {args.group} already in groups {groups}")
    else:
        groups.append(args.group)
        default_config["groups"] = groups
    print(f"training groups: {groups}")

    date_tag = args.group.split("_")[-1] or f"{datetime.now(timezone.utc):%Y%m%d}"
    # Name new runs purely by date (no a/b/c/d experiment letter): the checkpoint then
    # lands at gs://.../<project>/<RUN_NAME_STEM>_<date>/best.ckpt, date in the path.
    run_name = args.run_name or f"{config.RUN_NAME_STEM}_{date_tag}"
    data["run_name"] = run_name
    print(f"run_name: {run_name}")
    print(f"project_name: {data.get('project_name')}")

    out_path = (
        Path(args.out)
        if args.out
        else (REPO_ROOT / config.DATA_DIR_REL / f"config_classifier_{date_tag}.yaml")
    )
    _dump_yaml(data, yaml, out_path)
    rel_out = (
        out_path.relative_to(REPO_ROOT)
        if out_path.is_relative_to(REPO_ROOT)
        else out_path
    )
    print(f"\nwrote {out_path}")

    print("\nnext: launch the retrain (1 GPU), then publish with the new run_name:")
    print(
        "  python -m rslp.main olmoearth_pretrain launch_finetune \\\n"
        "    --olmoearth_checkpoint_path "
        "/weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 \\\n"
        "    --patch_size 4 --encoder_embedding_size 768 --image_name favyen/rslphelios3 \\\n"
        f"    --config_paths+={rel_out} \\\n"
        "    --cluster+=ai2/ceres-cirrascale \\\n"
        f"    --project_name {data.get('project_name')} --run_name {run_name} --gpus 1"
    )
    print(
        f"\n  python -m rslp.landsat_vessels.feedback.publish "
        f"--run-name {run_name} --project-name {data.get('project_name')} "
        f"--config {rel_out}"
    )


if __name__ == "__main__":
    main()
