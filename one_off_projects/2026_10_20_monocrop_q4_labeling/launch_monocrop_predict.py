"""Launch a Beaker GPU job that runs the monocrop classifier on the prediction dataset.

The model is data/forest_loss_driver/monocrop_classifier/model_classify_pool.yaml with
the checkpoint of 20260914_monocrop_classifier/olmoearth_v1_2_base_classify_pool_01. To
avoid touching that run's directory (code archive, logs), we write a copy of the config
that points at the prediction dataset and at a private management directory, where
{management_dir}/{PROJECT}/{RUN}/best.ckpt is a symlink to the original checkpoint.

The config copy is written next to this script so it is included in the code archive
that beaker_train uploads. Run from the rslearn_projects root with RSLP_PREFIX and
RSLP_WEKA_PREFIX set (e.g. both /weka/dfive-default/rslearn-eai):

    python one_off_projects/2026_10_20_monocrop_q4_labeling/launch_monocrop_predict.py \
        --ds_path /weka/.../monocrop_classifier/predict_20261021_q4/ \
        --management_dir /weka/.../2026_10_20_monocrop_q4_labeling/predict_mgmt
"""

import argparse
import os
from datetime import timedelta

import yaml

from rslp.common.beaker_train import beaker_train
from rslp.utils.beaker import WekaMount

BASE_CONFIG = "data/forest_loss_driver/monocrop_classifier/model_classify_pool.yaml"
CHECKPOINT = "/weka/dfive-default/rslearn-eai/projects/20260914_monocrop_classifier/olmoearth_v1_2_base_classify_pool_01/best.ckpt"
CONFIG_OUT = "one_off_projects/2026_10_20_monocrop_q4_labeling/predict_config.yaml"
PROJECT = "monocrop_q4_predict_20261021"
RUN = "olmoearth_v1_2_base_classify_pool_01"


def main(cli_args: argparse.Namespace) -> None:
    """Write the predict config and checkpoint symlink, then launch the job."""
    with open(BASE_CONFIG) as f:
        config = yaml.safe_load(f)
    config["data"]["init_args"]["path"] = cli_args.ds_path
    config["management_dir"] = cli_args.management_dir
    config["project_name"] = PROJECT
    config["run_name"] = RUN
    with open(CONFIG_OUT, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    run_dir = os.path.join(cli_args.management_dir, PROJECT, RUN)
    os.makedirs(run_dir, exist_ok=True)
    link = os.path.join(run_dir, "best.ckpt")
    if not os.path.lexists(link):
        os.symlink(CHECKPOINT, link)

    if cli_args.dry_run:
        print(f"wrote {CONFIG_OUT} and {link}, not launching")
        return
    beaker_train(
        image_name=cli_args.image,
        cluster=["ai2/jupiter", "ai2/ceres"],
        config_path=CONFIG_OUT,
        mode="predict",
        gpus=1,
        weka_mounts=[
            WekaMount(bucket_name="dfive-default", mount_path="/weka/dfive-default")
        ],
        project_id=PROJECT,
        experiment_id=RUN,
        priority="urgent",
        min_runtime=timedelta(hours=8),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--management_dir", required=True)
    parser.add_argument("--image", default="favyen/rslpomp20261007b")
    parser.add_argument("--dry_run", action="store_true")
    main(parser.parse_args())
