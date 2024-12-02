"""Launch landsat vessel prediction jobs on Beaker."""

import argparse
import hashlib
import json
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
from upath import UPath

from rslp.launch_beaker import (
    BUDGET,
    DEFAULT_WORKSPACE,
    IMAGE_NAME,
    get_base_env_vars,
)


def launch_job(
    scene_zip_path: str | None = None,
    window_path: str | None = None,
    json_dir: str = "",
    crop_dir: str | None = None,
    scratch_dir: str | None = None,
    use_weka_prefix: bool = False,
) -> None:
    """Launch a job for the landsat scene zip file.

    Args:
        scene_zip_path: the path to the landsat scene zip file.
        window_path: the path to the directory containing the windows.
        json_dir: the path to the directory containing the json files.
        crop_dir (optional): the path to the directory containing the crop files.
        scratch_dir (optional): the path to the directory containing the scratch files.
        use_weka_prefix: whether to use the weka prefix.
    """
    if scene_zip_path:
        job_id = scene_zip_path.split("/")[-1].split(".")[0]
    elif window_path:
        job_id = window_path.split("/")[-1]
    else:
        raise ValueError("No valid path found!")
    beaker = Beaker.from_env(default_workspace=DEFAULT_WORKSPACE)

    # this requires directory paths to end with '/'
    config = {
        "scene_zip_path": scene_zip_path,
        "window_path": window_path,
        "json_path": json_dir + job_id + ".json",
        "scratch_path": scratch_dir + job_id if scratch_dir else None,
        "crop_path": crop_dir + job_id if crop_dir else None,
    }

    with beaker.session():
        env_vars = get_base_env_vars(use_weka_prefix=use_weka_prefix)
        # Add AWS credentials for downloading data
        env_vars.append(
            EnvVar(
                name="AWS_ACCESS_KEY_ID",
                secret="AWS_ACCESS_KEY_ID",  # nosec
            )
        )
        env_vars.append(
            EnvVar(
                name="AWS_SECRET_ACCESS_KEY",
                secret="AWS_SECRET_ACCESS_KEY",  # nosec
            )
        )
        spec = ExperimentSpec.new(
            budget=BUDGET,
            description=f"landsat_vessel_{job_id}",
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
        beaker.experiment.create(f"landsat_vessel_{job_id}_{unique_id}", spec)


if __name__ == "__main__":
    dotenv.load_dotenv()
    parser = argparse.ArgumentParser(
        description="Launch beaker experiment for landsat prediction jobs",
    )
    parser.add_argument(
        "--zip_dir",
        type=str,
        help="Path to directory containing zip files containing landsat scenes (GCS or WEKA)",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--window_dir",
        type=str,
        help="Path to the directory containing the windows",
        required=False,
        default=None,
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

    use_weka_prefix = "weka://" in args.zip_dir if args.zip_dir else False

    paths = []
    if args.zip_dir:
        try:
            zip_dir_upath = UPath(args.zip_dir)
            paths = list(zip_dir_upath.glob("*.zip"))
        except Exception:
            # using S3 protocol to access WEKA is only supported on ai2 clusters
            # as a workaround for other machines, we load the corresponding gcs bucket first
            # then generate WEKA paths for beaker jobs
            zip_dir_path = UPath(args.zip_dir.replace("weka://dfive-default/", "gs://"))
            zip_paths = list(zip_dir_path.glob("*.zip"))
            paths = [
                str(zip_path).replace("gs://", "weka://dfive-default/")
                for zip_path in zip_paths
            ]
    elif args.window_dir:
        window_dir_upath = UPath(args.window_dir)
        # Only use the validation split of windows
        paths = [
            str(window_path)
            for window_path in window_dir_upath.iterdir()
            if hashlib.sha256(window_path.name.encode()).hexdigest()[0] in ["0", "1"]
        ]
    assert len(paths) > 0, "No valid paths found!"

    if args.dry_run:
        print(f"Dry run: launching job for {paths[0]}")
        launch_job(
            scene_zip_path=str(paths[0]) if args.zip_dir else None,
            window_path=str(paths[0]) if args.window_dir else None,
            json_dir=args.json_dir,
            crop_dir=args.crop_dir,
            scratch_dir=args.scratch_dir,
            use_weka_prefix=use_weka_prefix,
        )
    else:
        for scene_zip_path in tqdm.tqdm(paths, desc="Launching beaker jobs"):
            launch_job(
                scene_zip_path=str(scene_zip_path) if args.zip_dir else None,
                window_path=str(scene_zip_path) if args.window_dir else None,
                json_dir=args.json_dir,
                crop_dir=args.crop_dir,
                scratch_dir=args.scratch_dir,
                use_weka_prefix=use_weka_prefix,
            )
