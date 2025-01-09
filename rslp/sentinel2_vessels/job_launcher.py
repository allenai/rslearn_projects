"""Launch Sentinel-2 vessel prediction jobs on Beaker."""

import argparse
import json
import multiprocessing
import random
import uuid

import dotenv
import tqdm
from beaker import (
    Beaker,
    Constraints,
    DataMount,
    DataSource,
    ExperimentSpec,
    Priority,
    TaskResources,
)
from rslearn.utils.mp import star_imap_unordered
from upath import UPath

from rslp import launcher_lib
from rslp.launch_beaker import BUDGET, DEFAULT_WORKSPACE

from .predict_pipeline import PredictionTask


def launch_job(image_name: str, tasks: list[PredictionTask]) -> None:
    """Launch job for the Sentinel-2 scene.

    Args:
        image_name: the name of the beaker image to use.
        tasks: the prediction tasks to run.
    """
    beaker = Beaker.from_env(default_workspace=DEFAULT_WORKSPACE)
    first_scene_id = tasks[0].scene_id
    encoded_tasks = [
        dict(
            scene_id=task.scene_id,
            json_path=task.json_path,
            crop_path=task.crop_path,
        )
        for task in tasks
    ]

    with beaker.session():
        env_vars = launcher_lib.get_base_env_vars(use_weka_prefix=True)

        spec = ExperimentSpec.new(
            budget=BUDGET,
            description=f"sentinel2_vessel_{first_scene_id}",
            beaker_image=image_name,
            priority=Priority.low,
            command=["python", "-m", "rslp.main"],
            arguments=[
                "sentinel2_vessels",
                "predict",
                json.dumps(encoded_tasks),
                "/tmp/x/",
            ],
            constraints=Constraints(
                cluster=[
                    # Don't use Renton cirrascale since it loads very slowly from Weka.
                    "ai2/prior-elanding",
                    "ai2/jupiter-cirrascale-2",
                    "ai2/neptune-cirrascale",
                ]
            ),
            preemptible=True,
            datasets=[
                DataMount(
                    source=DataSource(secret="RSLEARN_GCP_CREDENTIALS"),  # nosec
                    mount_path="/etc/credentials/gcp_credentials.json",  # nosec
                ),
            ],
            env_vars=env_vars,
            resources=TaskResources(gpu_count=1),
        )
        unique_id = str(uuid.uuid4())[0:8]
        beaker.experiment.create(f"sentinel2_vessel_{first_scene_id}_{unique_id}", spec)


def check_scene_done(out_path: UPath, scene_id: str) -> tuple[str, bool]:
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
        "--image_name",
        type=str,
        help="The name of the beaker image to use",
        required=True,
    )
    parser.add_argument(
        "--json_fname",
        type=str,
        help="Path to JSON file containing list of scene IDs to run on",
        required=True,
    )
    parser.add_argument(
        "--json_out_dir",
        type=str,
        help="JSON output path",
        required=True,
    )
    parser.add_argument(
        "--crop_out_dir",
        type=str,
        help="Crop output path",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Prediction tasks per Beaker job",
        default=1,
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

    out_path = UPath(args.json_out_dir)
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
    random.shuffle(run_scene_ids)
    for i in tqdm.tqdm(
        range(0, len(run_scene_ids), args.batch_size), desc="Starting Beaker jobs"
    ):
        batch = run_scene_ids[i : i + args.batch_size]
        tasks: list[PredictionTask] = []
        for scene_id in batch:
            tasks.append(
                PredictionTask(
                    scene_id=scene_id,
                    json_path=f"{args.json_out_dir}{scene_id}.json",
                    crop_path=f"{args.crop_out_dir}{scene_id}",
                )
            )

        launch_job(args.image_name, tasks)
