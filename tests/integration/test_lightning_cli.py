import pathlib
import shutil
from typing import Any

import pytest
from upath import UPath

from rslp.utils.rslearn import run_model_predict


def test_error_if_no_checkpoint(tmp_path: pathlib.Path, monkeypatch: Any) -> None:
    """Verify that an error is raised if --load_best=true with no checkpoint."""
    # We need to use some config for this, so here we use the landsat_vessels one.
    model_config_fname = "data/landsat_vessels/config_detector.yaml"
    ds_config_fname = "data/landsat_vessels/predict_dataset_config.json"

    # Copy the config.json so that the dataset is valid.
    shutil.copyfile(ds_config_fname, tmp_path / "config.json")
    (tmp_path / "windows" / "default").mkdir(parents=True)

    # Overwrite RSLP_PREFIX to ensure the checkpoint won't exist.
    monkeypatch.setenv("RSLP_PREFIX", str(tmp_path))

    with pytest.raises(FileNotFoundError):
        run_model_predict(model_config_fname, UPath(tmp_path))
