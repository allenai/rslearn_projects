"""Launch Sentinel-2 vessel prediction jobs on Beaker."""

import argparse
import json
import multiprocessing
import os
import random
import uuid

import dotenv
import tqdm
from beaker import (
    Beaker,
    Constraints,
    DataMount,
    DataSource,
    EnvVar,
    ExperimentSpec,
    Priority,
    TaskResources,
)
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

WORKSPACE = "ai2/earth-systems"
BUDGET = "ai2/d5"
IMAGE_NAME = "favyen/2024_10_outlawocean_sentinel2"
JSON_OUT_DIR = (
    "gs://rslearn-eai/projects/2024_10_outlawocean_sentinel2/vessel_detections/json/"
)
CROP_OUT_DIR = (
    "gs://rslearn-eai/projects/2024_10_outlawocean_sentinel2/vessel_detections/crops/"
)


def launch_job(scene_id: str):
    """Launch job for the Sentinel-2 scene.

    Args:
        scene_id: the scene name in which to detect vessels.
    """
    beaker = Beaker.from_env(default_workspace=WORKSPACE)

    with beaker.session():
        env_vars = [
            EnvVar(
                name="WANDB_API_KEY",
                secret="RSLEARN_WANDB_API_KEY",
            ),
            EnvVar(
                name="GOOGLE_APPLICATION_CREDENTIALS",
                value="/etc/credentials/gcp_credentials.json",
            ),
            EnvVar(
                name="GCLOUD_PROJECT",
                value="skylight-proto-1",
            ),
            EnvVar(
                name="RSLP_BUCKET",
                value=os.environ["RSLP_BUCKET"],
            ),
            EnvVar(
                name="MKL_THREADING_LAYER",
                value="GNU",
            ),
        ]

        spec = ExperimentSpec.new(
            budget=BUDGET,
            description=f"sentinel2_vessel_{scene_id}",
            beaker_image=IMAGE_NAME,
            priority=Priority.low,
            command=["python", "-m", "rslp.main"],
            arguments=[
                "sentinel2_vessels",
                "predict",
                scene_id,
                "/tmp/x/",
                JSON_OUT_DIR + scene_id + ".json",
                CROP_OUT_DIR + scene_id,
            ],
            constraints=Constraints(
                cluster=[
                    "ai2/prior-cirrascale",
                    "ai2/prior-elanding",
                    "ai2/jupiter-cirrascale-2",
                    "ai2/neptune-cirrascale",
                    "ai2/pluto-cirrascale",
                    "ai2/general-cirrascale",
                ]
            ),
            preemptible=True,
            datasets=[
                DataMount(
                    source=DataSource(secret="RSLEARN_GCP_CREDENTIALS"),
                    mount_path="/etc/credentials/gcp_credentials.json",
                ),
            ],
            env_vars=env_vars,
            resources=TaskResources(gpu_count=1),
        )
        unique_id = str(uuid.uuid4())[0:8]
        beaker.experiment.create(f"sentinel2_vessel_{scene_id}_{unique_id}", spec)


def check_scene_done(out_path: UPath, scene_id: str) -> bool:
    """Checks whether the scene ID is done processing already.

    It is determined based on existence of output JSON file for that scene.

    Args:
        out_path: the directory where output JSON files should appear.
        scene_id: the scene ID to check.

    Returns:
        whether the job was completed
    """
    return scene_id, (out_path / (scene_id + ".json")).exists()


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Launch beaker experiment for rslearn_projects",
    )
    parser.add_argument(
        "--json_fname",
        type=str,
        help="Path to JSON file containing list of scene IDs to run on",
        required=True,
    )
    parser.add_argument(
        "--count",
        type=int,
        help="Number of jobs to start",
        default=None,
    )
    args = parser.parse_args()

    # See which scenes are not done yet.
    with open(args.json_fname) as f:
        scene_ids: list[str] = json.load(f)

    out_path = UPath(JSON_OUT_DIR)
    p = multiprocessing.Pool(32)
    outputs = star_imap_unordered(
        p,
        check_scene_done,
        [dict(out_path=out_path, scene_id=scene_id) for scene_id in scene_ids],
    )

    missing_scene_ids: list[str] = []
    for scene_id, is_done in tqdm.tqdm(
        outputs, desc="Check if scenes are processed already", total=len(scene_ids)
    ):
        if is_done:
            continue
        missing_scene_ids.append(scene_id)

    p.close()

    # Run up to count of them.
    if args.count and len(missing_scene_ids) > args.count:
        run_scene_ids = random.sample(missing_scene_ids, args.count)
    else:
        run_scene_ids = missing_scene_ids

    print(
        f"Got {len(scene_ids)} total scenes, {len(missing_scene_ids)} pending, running {len(run_scene_ids)} of them"
    )
    for scene_id in tqdm.tqdm(run_scene_ids, desc="Starting Beaker jobs"):
        launch_job(scene_id)
