"""Customized LightningCLI for rslearn_projects."""

import hashlib
import json
import os
import shutil
import sys
import tempfile

import fsspec
import jsonargparse
import wandb
import lightning as L
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.cli import SaveConfigCallback
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

logger = get_logger(__name__)


def get_cached_checkpoint(checkpoint_fname: UPath) -> str:
    """Get a local cached version of the specified checkpoint.

    If checkpoint_fname is already local, then it is returned. Otherwise, it is saved
    in a deterministic local cache directory under the system temporary directory, and
    the cached filename is returned.

    Note that the cache is not deleted when the program exits.

    Args:
        checkpoint_fname: the potentially non-local checkpoint file to load.

    Returns:
        a local filename containing the same checkpoint.
    """
    is_local = isinstance(
        checkpoint_fname.fs, fsspec.implementations.local.LocalFileSystem
    )
    if is_local:
        return checkpoint_fname.path

    cache_id = hashlib.sha256(str(checkpoint_fname).encode()).hexdigest()
    local_fname = os.path.join(
        tempfile.gettempdir(), "rslearn_cache", "checkpoints", f"{cache_id}.ckpt"
    )

    if os.path.exists(local_fname):
        logger.info(
            "using cached checkpoint for %s at %s", str(checkpoint_fname), local_fname
        )
        return local_fname

    logger.info("caching checkpoint %s to %s", str(checkpoint_fname), local_fname)
    os.makedirs(os.path.dirname(local_fname), exist_ok=True)
    with checkpoint_fname.open("rb") as src:
        with open(local_fname + ".tmp", "wb") as dst:
            shutil.copyfileobj(src, dst)
    os.rename(local_fname + ".tmp", local_fname)

    return local_fname


class SaveWandbRunIdCallback(Callback):
    """Callback to save the wandb run ID to GCS in case of resume."""

    def __init__(
        self,
        project_id: str,
        experiment_id: str,
        run_id: str | None,
        config_str: str | None,
    ) -> None:
        """Create a new SaveWandbRunIdCallback.

        Args:
            project_id: the project ID.
            experiment_id: the experiment ID.
            run_id: the run ID (for hyperparameter experiments)
            config_str: the JSON-encoded configuration of this experiment
        """
        self.project_id = project_id
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.config_str = config_str

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

        if self.config_str is not None and "rslp_project" not in wandb.config:
            wandb.config.update(json.loads(self.config_str))


class SaveConfigToProjectDirCallback(SaveConfigCallback):
    """Callback to save the configuration to checkpoint directory."""

    def save_config(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """Save the configuration."""
        # Lightning handles ensuring that this function is only called on rank 0, so we
        # don't need to worry about it ourselves.
        # This is done in the setup function of SaveConfigCallback.
        run_id = os.environ.get("RSLP_RUN_ID", None)
        run_id_path = f"{run_id}/" if run_id else ""
        checkpoint_dir = UPath(
            CHECKPOINT_DIR.format(
                rslp_prefix=os.environ["RSLP_PREFIX"],
                project_id=self.config.rslp_project,
                experiment_id=self.config.rslp_experiment,
                run_id=run_id_path,
            )
        )
        config_fname = checkpoint_dir / "config.yaml"
        if config_fname.exists():
            return

        config = self.parser.dump(self.config, skip_none=False)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with config_fname.open("w") as f:
            f.write(config)


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
            "--rslp_description",
            type=str,
            help="Description of the experiment",
            default="",
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
        parser.add_argument(
            "--profiler",
            type=str,
            help="Profiler to use for training. Can be 'simple' or 'advanced'",
            default=None,
        )

    def _get_checkpoint_path(
        self, checkpoint_dir: UPath, load_best: bool = False, autoresume: bool = False
    ) -> str | None:
        """Get path to checkpoint to load from, or None to not restore checkpoint.

        With --load_best=true, we load the best-performing checkpoint. An error is
        thrown if the checkpoint doesn't exist.

        With --autoresume=true, we load last.ckpt if it exists, but proceed with
        default initialization otherwise.

        Otherwise, we do not restore any existing checkpoint (i.e., we use default
        initialization), and throw an error if there is an existing checkpoint.

        When training, it is suggested to use no option (don't expect to restart
        training) or --autoresume=true (if restart is expected, e.g. due to
        preemption). For inference, it is suggested to use --load_best=true.

        Args:
            checkpoint_dir: the directory where checkpoints are stored.
            load_best: whether to load the best performing checkpoint and require a
                checkpoint to exist.
            autoresume: whether to load the checkpoint if it exists but proceed even
                if it does not.

        Returns:
            the path to the checkpoint for setting c.ckpt_path, or None if no
                checkpoint should be restored.
        """
        if load_best:
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
                # If it is some other format, then set it 0 so we only use it if it's
                # the only option.
                # If it is last.ckpt then we set it -100 to only use it if there is not
                # even another format like "best.ckpt".
                extracted_epochs = 0
                if option.name == "last.ckpt":
                    extracted_epochs = -100

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

            if best_checkpoint is None:
                raise ValueError(
                    f"load_best enabled but no checkpoint is available in {checkpoint_dir}"
                )

            # Cache the checkpoint so we only need to download once in case we
            # reuse it later.
            # We only cache with --load_best since this is the only scenario where it
            return get_cached_checkpoint(best_checkpoint)

        elif autoresume:
            last_checkpoint_path = checkpoint_dir / "last.ckpt"
            if last_checkpoint_path.exists():
                return last_checkpoint_path
            else:
                return None

        else:
            last_checkpoint_path = checkpoint_dir / "last.ckpt"
            if last_checkpoint_path.exists():
                raise ValueError("autoresume is off but checkpoint already exists")
            else:
                return None

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
            if c.rslp_description:
                c.trainer.logger.init_args.notes = c.rslp_description

            # Configure DDP strategy with find_unused_parameters=True
            c.trainer.strategy = jsonargparse.Namespace(
                {
                    "class_path": "lightning.pytorch.strategies.DDPStrategy",
                    "init_args": jsonargparse.Namespace(
                        {"find_unused_parameters": True}
                    ),
                }
            )

            # Configure profiler if specified
            if c.profiler:
                max_steps = 100
                c.trainer.profiler = c.profiler
                c.trainer.max_steps = max_steps
                logger.info(f"Using profiler: {c.profiler}")
                logger.info(f"Setting max_steps to {max_steps}")

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

            # Save W&B run ID callback.
            if not upload_wandb_callback:
                config_str = json.dumps(
                    c.as_dict(), default=lambda _: "<not serializable>"
                )
                upload_wandb_callback = jsonargparse.Namespace(
                    {
                        "class_path": "SaveWandbRunIdCallback",
                        "init_args": jsonargparse.Namespace(
                            {
                                "project_id": c.rslp_project,
                                "experiment_id": c.rslp_experiment,
                                "run_id": run_id,
                                "config_str": config_str,
                            }
                        ),
                    }
                )
                c.trainer.callbacks.append(upload_wandb_callback)

        checkpoint_path = self._get_checkpoint_path(
            checkpoint_dir, load_best=c.load_best, autoresume=c.autoresume
        )
        if checkpoint_path is not None:
            logger.info(f"found checkpoint to resume from at {checkpoint_path}")
            c.ckpt_path = checkpoint_path

            wandb_id = launcher_lib.download_wandb_id(
                c.rslp_project, c.rslp_experiment, run_id
            )
            if wandb_id and subcommand == "fit":
                logger.info(f"resuming wandb run {wandb_id}")
                c.trainer.logger.init_args.id = wandb_id


def custom_model_handler() -> None:
    """Overrides model_handler in rslearn.main to use CustomLightningCLI.

    It also sets the save_config_callback.
    """
    # Decreased strictness of type checking for model and datamodule classes
    # to allow for multiple dataset training tasks
    CustomLightningCLI(
        model_class=L.LightningModule,
        datamodule_class=L.LightningDataModule,
        args=sys.argv[2:],
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=SaveConfigToProjectDirCallback,
        save_config_kwargs={"overwrite": True, "save_to_log_dir": False},
    )
