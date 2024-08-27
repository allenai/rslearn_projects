"""Customized LightningCLI for rslearn_projects."""

import os

import fsspec
import jsonargparse
import wandb
from lightning.pytorch.callbacks import Callback
from rslearn.main import RslearnLightningCLI

from rslp import launcher_lib

CHECKPOINT_DIR = "gs://{rslp_bucket}/projects/{project_id}/{experiment_id}/checkpoints/"


class SaveWandbRunIdCallback(Callback):
    """Callback to save the wandb run ID to GCS in case of resume."""

    def __init__(self, project_id: str, experiment_id: str):
        """Create a new SaveWandbRunIdCallback.

        Args:
            project_id: the project ID.
            experiment_id: the experiment ID.
        """
        self.project_id = project_id
        self.experiment_id = experiment_id

    def on_fit_start(self, trainer, pl_module):
        """Called just before fit starts I think.

        Args:
            trainer: the Trainer object.
            pl_module: the LightningModule object.
        """
        run_id = wandb.run.id
        launcher_lib.upload_wandb_id(self.project_id, self.experiment_id, run_id)


class CustomLightningCLI(RslearnLightningCLI):
    """Extended LightningCLI to manage cloud checkpointing and wandb run naming.

    This provides AI2-specific configuration that should be used across
    rslearn_projects.
    """

    def add_arguments_to_parser(self, parser) -> None:
        """Add experiment ID argument.

        Args:
            parser: the argument parser
        """
        super().add_arguments_to_parser(parser)
        parser.add_argument(
            "--rslp_project",
            type=str,
            help="A unique name for the project for which this is one experiment.",
            required=True,
        )
        parser.add_argument(
            "--rslp_experiment",
            type=str,
            help="A unique name for this experiment.",
            required=True,
        )
        parser.add_argument(
            "--autoresume",
            type=bool,
            help="Auto-resume from existing checkpoint",
            default=False,
        )

    def before_instantiate_classes(self):
        """Called before Lightning class initialization."""
        super().before_instantiate_classes()
        subcommand = self.config.subcommand
        c = self.config[subcommand]

        # Add and configure WandbLogger as needed.
        if not c.trainer.logger:
            c.trainer.logger = jsonargparse.Namespace(
                {
                    "class_path": "lightning.pytorch.loggers.WandbLogger",
                    "init_args": jsonargparse.Namespace(),
                }
            )
        c.trainer.logger.init_args.project = c.rslp_project
        c.trainer.logger.init_args.name = c.rslp_experiment

        # Set the checkpoint directory to canonical GCS location.
        checkpoint_callback = None
        upload_wandb_callback = None
        if "callbacks" in c.trainer:
            for existing_callback in c.trainer.callbacks:
                if (
                    existing_callback.class_path
                    == "lightning.pytorch.callbacks.ModelCheckpoint"
                ):
                    checkpoint_callback = existing_callback
                if existing_callback.class_path == "SaveWandbRunIdCallback":
                    upload_wandb_callback = existing_callback
        else:
            c.trainer.callbacks = []

        if not checkpoint_callback:
            checkpoint_callback = jsonargparse.Namespace(
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": jsonargparse.Namespace(
                        {
                            "save_last": True,
                            "save_top_k": 1,
                            "monitor": "val_loss",
                        }
                    ),
                }
            )
            c.trainer.callbacks.append(checkpoint_callback)
        checkpoint_dir = CHECKPOINT_DIR.format(
            rslp_bucket=os.environ["RSLP_BUCKET"],
            project_id=c.rslp_project,
            experiment_id=c.rslp_experiment,
        )
        checkpoint_callback.init_args.dirpath = checkpoint_dir

        if not upload_wandb_callback:
            upload_wandb_callback = jsonargparse.Namespace(
                {
                    "class_path": "SaveWandbRunIdCallback",
                    "init_args": jsonargparse.Namespace(
                        {
                            "project_id": c.rslp_project,
                            "experiment_id": c.rslp_experiment,
                        }
                    ),
                }
            )
            c.trainer.callbacks.append(upload_wandb_callback)

        # Check if there is an existing checkpoint.
        # If so, and autoresume is disabled, we should throw error.
        # If autoresume is enabled, then we should resume from the checkpoint.
        gcs = fsspec.filesystem("gcs")
        if gcs.exists(checkpoint_dir + "last.ckpt"):
            if not c.autoresume:
                raise ValueError("autoresume is off but checkpoint already exists")
            c.ckpt_path = checkpoint_dir + "last.ckpt"
            print(f"found checkpoint to resume from at {c.ckpt_path}")
            # TODO: additional logic here to continue the wandb run.

            wandb_id = launcher_lib.download_wandb_id(c.rslp_project, c.rslp_experiment)
            if wandb_id:
                print(f"resuming wandb run {wandb_id}")
                c.trainer.logger.init_args.id = wandb_id
