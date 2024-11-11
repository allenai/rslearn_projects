"""Launch landsat vessel prediction jobs on Beaker."""

import argparse
import json
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
from upath import UPath

from rslp.launch_beaker import (
    BUDGET,
    DEFAULT_WORKSPACE,
    IMAGE_NAME,
    get_base_env_vars,
)


def launch_job(
    scene_zip_path: str,
    json_dir: str,
    crop_dir: str | None = None,
    scratch_dir: str | None = None,
    use_weka_prefix: bool = False,
) -> None:
    """Launch a job for the landsat scene zip file.

    Args:
        scene_zip_path: the path to the landsat scene zip file.
        json_dir: the path to the directory containing the json files.
        crop_dir (optional): the path to the directory containing the crop files.
        scratch_dir (optional): the path to the directory containing the scratch files.
        use_weka_prefix: whether to use the weka prefix.
    """
    scene_id = (scene_zip_path.split("/")[-1]).split(".")[0]
    beaker = Beaker.from_env(default_workspace=DEFAULT_WORKSPACE)

    # this requires directory paths to end with '/'
    config = {
        "scene_zip_path": scene_zip_path,
        "json_path": json_dir + scene_id + ".json",
        "scratch_path": scratch_dir + scene_id if scratch_dir else None,
        "crop_path": crop_dir + scene_id if crop_dir else None,
    }

    with beaker.session():
        env_vars = get_base_env_vars(use_weka_prefix=use_weka_prefix)

        spec = ExperimentSpec.new(
            budget=BUDGET,
            description=f"landsat_vessel_{scene_id}",
            beaker_image=IMAGE_NAME,
            command=["python", "-m", "rslp.main"],
            arguments=[
                "landsat_vessels",
                "predict",
                "--config",
                json.dumps(config),
            ],
            constraints=Constraints(
                cluster=[
                    "ai2/prior-elanding",
                    "ai2/jupiter-cirrascale-2",
                    "ai2/neptune-cirrascale",
                ]
            ),
            priority=Priority.low,
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
        beaker.experiment.create(f"landsat_vessel_{scene_id}_{unique_id}", spec)


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Launch beaker experiment for landsat prediction jobs",
    )
    parser.add_argument(
        "--zip_dir",
        type=str,
        help="Path to directory containing zip files containing landsat scenes (GCS or WEKA)",
        required=True,
    )
    parser.add_argument(
        "--json_dir",
        type=str,
        help="Path to directory containing json files",
        required=True,
    )
    parser.add_argument(
        "--crop_dir",
        type=str,
        help="Path to directory containing crop files",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--scratch_dir",
        type=str,
        help="Path to directory containing scratch files",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run the script",
    )
    args = parser.parse_args()

    use_weka_prefix = "weka://" in args.zip_dir
    try:
        zip_dir_upath = UPath(args.zip_dir)
        zip_paths = list(zip_dir_upath.glob("*.zip"))
    except Exception:
        # using S3 protocol to access WEKA is only supported on ai2 clusters
        # as a workaround for other machines, we load the corresponding gcs bucket first
        # then generate WEKA paths for beaker jobs
        zip_dir_path = UPath(args.zip_dir.replace("weka://dfive-default/", "gs://"))
        zip_paths = list(zip_dir_path.glob("*.zip"))
        zip_paths = [
            str(zip_path).replace("gs://", "weka://dfive-default/")
            for zip_path in zip_paths
        ]

    assert len(zip_paths) > 0, "No zip files found in the directory"

    if args.dry_run:
        print(f"Dry run: launching job for {zip_paths[0]}")
        launch_job(
            str(zip_paths[0]),
            args.json_dir,
            args.crop_dir,
            args.scratch_dir,
            use_weka_prefix,
        )
    else:
        for scene_zip_path in tqdm.tqdm(zip_paths, desc="Launching beaker jobs"):
            launch_job(
                str(scene_zip_path),
                args.json_dir,
                args.crop_dir,
                args.scratch_dir,
                use_weka_prefix,
            )
