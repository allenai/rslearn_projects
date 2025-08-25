"""Test rslp.forest_loss_driver.webapp.index_windows module."""

import json
import pathlib

from rslearn.dataset import Window

from rslp.forest_loss_driver.webapp.index_windows import (
    OUTPUT_GEOJSON_SUFFIX,
    WINDOWS_FNAME,
    index_windows,
)


def test_index_windows(tmp_path: pathlib.Path) -> None:
    """Test index_windows.

    We create two windows, one with an output and one without one. So index_windows
    should index just one of them.
    """
    ds_path = tmp_path
    group = "default"

    # Good window (has output).
    good_window_name = "good"
    window_dir = Window.get_window_root(ds_path, group, good_window_name)
    window_dir.mkdir(parents=True, exist_ok=True)
    (window_dir / "metadata.json").touch()
    fname = window_dir / OUTPUT_GEOJSON_SUFFIX
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname.touch()

    # Bad window (has no output).
    bad_window_name = "bad"
    window_dir = Window.get_window_root(ds_path, group, bad_window_name)
    window_dir.mkdir(parents=True, exist_ok=True)
    (window_dir / "metadata.json").touch()

    index_windows(str(ds_path))

    # It should be a JSON list of window names that have output.
    # So should just be one window name, the good one where we touched the output
    # filename.
    with (ds_path / WINDOWS_FNAME).open() as f:
        good_window_names = json.load(f)
    assert len(good_window_names) == 1
    assert good_window_names[0] == good_window_name
