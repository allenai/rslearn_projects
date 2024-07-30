import argparse
import uuid

from beaker import Beaker, Constraints, DataMount, DataSource, EnvVar, ExperimentSpec, Priority, TaskResources

DEFAULT_WORKSPACE = "ai2/earth-systems"
BUDGET = "ai2/prior"
IMAGE_NAME = "favyen/rslearn"

def launch_job(config_path: str, workspace: str):
    beaker = Beaker.from_env(default_workspace=workspace)

    with beaker.session():
        spec = ExperimentSpec.new(
            budget=BUDGET,
            description=config_path,
            beaker_image=IMAGE_NAME,
            priority=Priority.high,
            command=["python", "-m", "rslp.main"],
            arguments=["model", "fit", "--config", config_path],
            constraints=Constraints(cluster=["ai2/jupiter-cirrascale-2"]),
            preemptible=True,
            datasets=[
                DataMount(
                    source=DataSource(secret="RSLEARN_GCP_CREDENTIALS"),
                    mount_path="/etc/credentials/gcp_credentials.json",
                ),
            ],
            env_vars=[
                EnvVar(
                    name="WANDB_API_KEY",
                    secret="RSLEARN_WANDB_API_KEY",
                ),
                EnvVar(
                    name="GOOGLE_APPLICATION_CREDENTIALS",
                    value="/etc/credentials/gcp_credentials.json",
                ),
            ],
            resources=TaskResources(gpu_count=1),
        )
        beaker.experiment.create(config_path.replace("/", "_") + "-" + str(uuid.uuid4()), spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch beaker experiment for rslearn_projects",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to configuration file relative to rslearn_projects repository root",
        required=True,
    )
    parser.add_argument(
        "--workspace",
        type=str,
        help="Which workspace to run the experiment in",
        default=DEFAULT_WORKSPACE,
    )
    args = parser.parse_args()
    launch_job(args.config_path, args.workspace)
