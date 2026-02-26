import json
import os
import pathlib
import shutil
from typing import Any

import lightning.pytorch as L
import numpy as np
import pytest
import shapely
import torch
import yaml
from rslearn.arg_parser import RslearnArgumentParser
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset, Window
from rslearn.train.data_module import RslearnDataModule
from rslearn.train.lightning_module import RslearnLightningModule
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import STGeometry
from rslearn.utils.jsonargparse import init_jsonargparse
from rslearn.utils.raster_array import RasterArray
from rslearn.utils.raster_format import GeotiffRasterFormat
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

from rslp.lightning_cli import (
    CHECKPOINT_DIR,
    CustomLightningCLI,
    SaveConfigToProjectDirCallback,
)
from rslp.log_utils import get_logger
from rslp.utils.rslearn import run_model_predict

logger = get_logger(__name__)

# We need to use some config for this, so here we use the landsat_vessels one.
MODEL_CONFIG_FNAME = "data/landsat_vessels/config_detector.yaml"
DS_CONFIG_FNAME = "data/landsat_vessels/predict_dataset_config.json"

CLASSES = ["cat", "dog"]
PROPERTY_NAME = "category"


class CrashAtEpochCallback(L.Callback):
    """Callback that raises an exception at a specified epoch.

    This prevents on_train_end from running, ensuring that tests exercise
    checkpoint-saving behavior during training rather than at shutdown.
    """

    def __init__(self, crash_at_epoch: int = 2) -> None:
        """Initialize CrashAtEpochCallback.

        Args:
            crash_at_epoch: the epoch (0-indexed) at which to crash.
        """
        self.crash_at_epoch = crash_at_epoch

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Raise if we reached the crash epoch.

        Args:
            trainer: the trainer.
            pl_module: the lightning module.
        """
        if trainer.current_epoch >= self.crash_at_epoch:
            raise RuntimeError("Intentional crash for testing")


def test_error_if_no_checkpoint(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """Verify that an error is raised if --load_best=true with no checkpoint."""

    # Copy the config.json so that the dataset is valid.
    shutil.copyfile(DS_CONFIG_FNAME, tmp_path / "config.json")
    (tmp_path / "windows" / "default").mkdir(parents=True)

    # Overwrite RSLP_PREFIX to ensure the checkpoint won't exist.
    monkeypatch.setenv("RSLP_PREFIX", str(tmp_path))

    with pytest.raises(FileNotFoundError):
        run_model_predict(MODEL_CONFIG_FNAME, UPath(tmp_path))


def test_prefer_best_over_last(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """Verify that best.ckpt is preferred over last.ckpt."""
    # We create a valid last.ckpt and verify it works.
    # Then we create a corrupted best.ckpt and verify that it fails.

    # Copy the config.json so that the dataset is valid.
    shutil.copyfile(DS_CONFIG_FNAME, tmp_path / "config.json")
    (tmp_path / "windows" / "default").mkdir(parents=True)

    # Copy last.ckpt from actual RSLP_PREFIX to the tmp_path.
    # Then set RSLP_PREFIX=tmp_path.
    with open(MODEL_CONFIG_FNAME) as f:
        model_config = yaml.safe_load(f)
    actual_checkpoint_dir = UPath(
        CHECKPOINT_DIR.format(
            rslp_prefix=os.environ["RSLP_PREFIX"],
            project_id=model_config["rslp_project"],
            experiment_id=model_config["rslp_experiment"],
            run_id="",
        )
    )
    monkeypatch.setenv("RSLP_PREFIX", str(tmp_path))
    fake_checkpoint_dir = UPath(
        CHECKPOINT_DIR.format(
            rslp_prefix=os.environ["RSLP_PREFIX"],
            project_id=model_config["rslp_project"],
            experiment_id=model_config["rslp_experiment"],
            run_id="",
        )
    )
    fake_checkpoint_dir.mkdir(parents=True)
    logger.debug(f"copy from {actual_checkpoint_dir} to {fake_checkpoint_dir}")
    with (actual_checkpoint_dir / "last.ckpt").open("rb") as src:
        with (fake_checkpoint_dir / "last.ckpt").open("wb") as dst:
            shutil.copyfileobj(src, dst)

    # Verify that the run_model_predict succeeds.
    run_model_predict(MODEL_CONFIG_FNAME, UPath(tmp_path))

    # Create corrupted best.ckpt and verify run_model_predict fails (implying that it
    # preferred best.ckpt over last.ckpt).
    with (fake_checkpoint_dir / "best.ckpt").open("w") as f:
        f.write("bad")
    with pytest.raises(Exception):
        run_model_predict(MODEL_CONFIG_FNAME, UPath(tmp_path))


@pytest.fixture
def classification_dataset(tmp_path: pathlib.Path) -> Dataset:
    """Create a minimal classification dataset with one 32x32 window."""
    ds_path = UPath(tmp_path / "dataset")

    dataset_config = {
        "layers": {
            "image": {
                "type": "raster",
                "band_sets": [{"dtype": "uint8", "bands": ["band"]}],
            },
            "targets": {"type": "vector"},
        },
    }
    ds_path.mkdir(parents=True, exist_ok=True)
    with (ds_path / "config.json").open("w") as f:
        json.dump(dataset_config, f)
    dataset = Dataset(ds_path)

    window = Window(
        storage=dataset.storage,
        group="default",
        name="default",
        projection=WGS84_PROJECTION,
        bounds=(0, 0, 32, 32),
        time_range=None,
    )
    window.save()

    image = np.random.randint(0, 255, size=(1, 32, 32), dtype=np.uint8)
    layer_dir = window.get_layer_dir("image")
    GeotiffRasterFormat().encode_raster(
        layer_dir / "band",
        window.projection,
        window.bounds,
        RasterArray(chw_array=image),
    )
    window.mark_layer_completed("image")

    feature = Feature(
        STGeometry(WGS84_PROJECTION, shapely.Point(16, 16), None),
        {PROPERTY_NAME: CLASSES[0]},
    )
    layer_dir = window.get_layer_dir("targets")
    GeojsonVectorFormat().encode_vector(layer_dir, [feature])
    window.mark_layer_completed("targets")

    return dataset


def test_save_last_checkpoint_every_epoch(
    classification_dataset: Dataset,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With CustomLightningCLI, last checkpoint is saved on every epoch.

    We crash at epoch 2 to avoid on_train_end saving last.ckpt at shutdown. With lr=0
    the model never updates so val_loss is constant: save_top_k should only save the
    epoch-0 checkpoint, but the "last" callback should update last.ckpt every epoch.
    """
    init_jsonargparse()

    rslp_project = "test_project"
    rslp_experiment = "test_experiment"

    monkeypatch.setenv("RSLP_PREFIX", str(tmp_path))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("DATASET_PATH", str(classification_dataset.path))

    cfg = {
        "model": {
            "class_path": "rslearn.train.lightning_module.RslearnLightningModule",
            "init_args": {
                "model": {
                    "class_path": "rslearn.models.singletask.SingleTaskModel",
                    "init_args": {
                        "encoder": [
                            {
                                "class_path": "rslearn.models.swin.Swin",
                                "init_args": {
                                    "arch": "swin_t",
                                    "input_channels": 1,
                                    "num_outputs": len(CLASSES),
                                },
                            }
                        ],
                        "decoder": [
                            {
                                "class_path": "rslearn.train.tasks.classification.ClassificationHead",
                            },
                        ],
                    },
                },
                "optimizer": {
                    "class_path": "rslearn.train.optimizer.AdamW",
                    "init_args": {"lr": 0},
                },
            },
        },
        "data": {
            "class_path": "rslearn.train.data_module.RslearnDataModule",
            "init_args": {
                "path": "${DATASET_PATH}",
                "inputs": {
                    "image": {
                        "data_type": "raster",
                        "layers": ["image"],
                        "bands": ["band"],
                        "passthrough": True,
                        "dtype": "FLOAT32",
                    },
                    "targets": {
                        "data_type": "vector",
                        "layers": ["targets"],
                    },
                },
                "task": {
                    "class_path": "rslearn.train.tasks.classification.ClassificationTask",
                    "init_args": {
                        "property_name": PROPERTY_NAME,
                        "classes": CLASSES,
                    },
                },
                "batch_size": 1,
            },
        },
        "trainer": {
            "max_epochs": 3,
            "accelerator": "cpu",
            "callbacks": [
                {
                    "class_path": "tests.integration.test_lightning_cli.CrashAtEpochCallback",
                    "init_args": {"crash_at_epoch": 2},
                },
            ],
        },
        "rslp_project": rslp_project,
        "rslp_experiment": rslp_experiment,
    }

    tmp_fname = tmp_path / "config.yaml"
    with tmp_fname.open("w") as f:
        json.dump(cfg, f)

    with pytest.raises(RuntimeError, match="Intentional crash"):
        CustomLightningCLI(
            model_class=RslearnLightningModule,
            datamodule_class=RslearnDataModule,
            args=["fit", "--config", str(tmp_fname)],
            subclass_mode_model=True,
            subclass_mode_data=True,
            save_config_callback=SaveConfigToProjectDirCallback,
            save_config_kwargs={"overwrite": True, "save_to_log_dir": False},
            parser_class=RslearnArgumentParser,
        )

    checkpoint_dir = pathlib.Path(
        CHECKPOINT_DIR.format(
            rslp_prefix=str(tmp_path),
            project_id=rslp_project,
            experiment_id=rslp_experiment,
            run_id="",
        )
    )

    # The best checkpoint should be from epoch 0 (the first and only time the metric
    # was recorded as "best", since lr=0 means val_loss is constant).
    best_ckpt_path = checkpoint_dir / "best.ckpt"
    assert best_ckpt_path.exists(), "best.ckpt should exist"
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    assert (
        best_ckpt["epoch"] == 0
    ), f"Best checkpoint should be from epoch 0, but got epoch {best_ckpt['epoch']}"

    # last.ckpt should be from epoch 1 (the last fully completed epoch before crash).
    last_ckpt_path = checkpoint_dir / "last.ckpt"
    assert last_ckpt_path.exists(), "last.ckpt should exist"
    last_ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
    assert (
        last_ckpt["epoch"] == 1
    ), f"last.ckpt should be from epoch 1, but got epoch {last_ckpt['epoch']}"
