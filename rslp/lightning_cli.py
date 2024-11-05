"""Customized LightningCLI for rslearn_projects."""

import os

import jsonargparse
import wandb
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only
from rslearn.main import RslearnLightningCLI
from upath import UPath

import rslp.utils.fs  # noqa: F401 (imported but unused)
from rslp import launcher_lib
from rslp.log_utils import get_logger

logger = get_logger(__name__)

CHECKPOINT_DIR = (
    "{rslp_prefix}/projects/{project_id}/{experiment_id}/{run_id}checkpoints/"
)


class SaveWandbRunIdCallback(Callback):
    """Callback to save the wandb run ID to GCS in case of resume."""

    def __init__(self, project_id: str, experiment_id: str, run_id: str) -> None:
        """Create a new SaveWandbRunIdCallback.

        Args:
            project_id: the project ID.
            experiment_id: the experiment ID.
            run_id: the run ID (for hyperparameter experiments)
        """
        self.project_id = project_id
        self.experiment_id = experiment_id
        self.run_id = run_id

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called just before fit starts I think.

        Args:
            trainer: the Trainer object.
            pl_module: the LightningModule object.
        """
        wandb_id = wandb.run.id
        launcher_lib.upload_wandb_id(
            self.project_id, self.experiment_id, self.run_id, wandb_id
        )


class CustomLightningCLI(RslearnLightningCLI):
    """Extended LightningCLI to manage cloud checkpointing and wandb run naming.

    This provides AI2-specific configuration that should be used across
    rslearn_projects.
    """

    def add_arguments_to_parser(self, parser: jsonargparse.ArgumentParser) -> None:
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
        parser.add_argument(
            "--load_best",
            type=bool,
            help="Load best checkpoint from GCS for test/predict",
            default=False,
        )
        parser.add_argument(
            "--force_log",
            type=bool,
            help="Log to W&B even for test/predict",
            default=False,
        )
        parser.add_argument(
            "--no_log",
            type=bool,
            help="Disable W&B logging for fit",
            default=False,
        )

    def before_instantiate_classes(self) -> None:
        """Called before Lightning class initialization."""
        super().before_instantiate_classes()
        subcommand = self.config.subcommand
        c = self.config[subcommand]

        run_id = os.environ.get("RSLP_RUN_ID", None)
        run_id_path = f"{run_id}/" if run_id else ""
        checkpoint_dir = UPath(
            CHECKPOINT_DIR.format(
                rslp_prefix=os.environ["RSLP_PREFIX"],
                project_id=c.rslp_project,
                experiment_id=c.rslp_experiment,
                run_id=run_id_path,
            )
        )

        if (subcommand == "fit" and not c.no_log) or c.force_log:
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

            # Configure DDP strategy with find_unused_parameters=True
            c.trainer.strategy = jsonargparse.Namespace(
                {
                    "class_path": "lightning.pytorch.strategies.DDPStrategy",
                    "init_args": jsonargparse.Namespace(
                        {"find_unused_parameters": True}
                    ),
                }
            )

        if subcommand == "fit" and not c.no_log:
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
            checkpoint_callback.init_args.dirpath = str(checkpoint_dir)

            if not upload_wandb_callback:
                upload_wandb_callback = jsonargparse.Namespace(
                    {
                        "class_path": "SaveWandbRunIdCallback",
                        "init_args": jsonargparse.Namespace(
                            {
                                "project_id": c.rslp_project,
                                "experiment_id": c.rslp_experiment,
                                "run_id": run_id,
                            }
                        ),
                    }
                )
                c.trainer.callbacks.append(upload_wandb_callback)

        # Check if there is an existing checkpoint.
        # If so, and autoresume/load_best are disabled, we should throw error.
        # If autoresume is enabled, then we should resume from last.ckpt.
        # If load_best is enabled, then we should try to identify the best checkpoint.
        # We still use last.ckpt to see if checkpoint exists since last.ckpt should
        # always be written.
        if (checkpoint_dir / "last.ckpt").exists():
            if c.load_best:
                # Checkpoints should be either:
                # - last.ckpt
                # - of the form "A=B-C=D-....ckpt" with one key being epoch=X
                # So we want the one with the highest epoch, and only use last.ckpt if
                # it's the only option.
                # User should set save_top_k=1 so there's just one, otherwise we won't
                # actually know which one is the best.
                best_checkpoint = None
                best_epochs = None
                for option in checkpoint_dir.iterdir():
                    if not option.name.endswith(".ckpt"):
                        continue

                    # Try to see what epochs this checkpoint is at.
                    # If it is last.ckpt or some other format, then set it 0 so we only
                    # use it if it's the only option.
                    extracted_epochs = 0
                    parts = option.name.split(".ckpt")[0].split("-")
                    for part in parts:
                        kv_parts = part.split("=")
                        if len(kv_parts) != 2:
                            continue
                        if kv_parts[0] != "epoch":
                            continue
                        extracted_epochs = int(kv_parts[1])

                    if best_checkpoint is None or extracted_epochs > best_epochs:
                        best_checkpoint = option
                        best_epochs = extracted_epochs

                c.ckpt_path = str(best_checkpoint)

            elif c.autoresume:
                c.ckpt_path = str(checkpoint_dir / "last.ckpt")

            else:
                raise ValueError("autoresume is off but checkpoint already exists")

            logger.info(f"found checkpoint to resume from at {c.ckpt_path}")

            wandb_id = launcher_lib.download_wandb_id(
                c.rslp_project, c.rslp_experiment, run_id
            )
            if wandb_id and subcommand == "fit":
                logger.info(f"resuming wandb run {wandb_id}")
                c.trainer.logger.init_args.id = wandb_id
