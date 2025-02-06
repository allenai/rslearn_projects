import os
import pathlib
import shutil
from typing import Any

import pytest
import yaml
from upath import UPath

from rslp.lightning_cli import CHECKPOINT_DIR
from rslp.log_utils import get_logger
from rslp.utils.rslearn import run_model_predict

logger = get_logger(__name__)

# We need to use some config for this, so here we use the landsat_vessels one.
MODEL_CONFIG_FNAME = "data/landsat_vessels/config_detector.yaml"
DS_CONFIG_FNAME = "data/landsat_vessels/predict_dataset_config.json"


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
