"""Test bkt file operations."""

import os
import pathlib

import pytest

from rslp.satlas.bkt import (
    BktDownloadRequest,
    download_from_bkt,
    get_bigtable,
    make_bkt,
)

RUNNING_IN_CI = os.environ.get("CI", "false").lower() == "true"


@pytest.mark.skipif(RUNNING_IN_CI, reason="Skipping in CI environment")
def test_make_and_download_bkt(tmp_path: pathlib.Path) -> None:
    """Test making and downloading a bkt file.

    We create two files and make a bkt from them.

    Then we try to use download_from_bkt to download both of those files and verify
    that it is returned correctly.
    """

    # make_bkt expects a directory structure zoom/col/row.
    zoom = "1"
    data_col0_row0 = b"bkt1"
    data_col0_row1 = b"bkt2"
    bkt_fname = "tests/test_make_and_download_bkt"
    (tmp_path / zoom / "0").mkdir(parents=True)
    with (tmp_path / zoom / "0" / "0").open("wb") as f:
        f.write(data_col0_row0)
    with (tmp_path / zoom / "0" / "1").open("wb") as f:
        f.write(data_col0_row1)
    make_bkt(str(tmp_path), bkt_fname)

    # Now try to download both of those files.
    # We use the metadata to store the expected contents.
    # We also read an extra tile that shouldn't exist.
    download_requests = [
        BktDownloadRequest(bkt_fname, 0, 0, metadata=data_col0_row0),
        BktDownloadRequest(bkt_fname, 0, 1, metadata=data_col0_row1),
        # This one should not exist.
        BktDownloadRequest(bkt_fname, 1, 1),
    ]
    bkt_files_table = get_bigtable()
    # Call download_from_bkt and populate into list so we can check the length.
    results = list(download_from_bkt(bkt_files_table, download_requests))
    assert len(results) == 2
    for expected, actual in results:
        assert expected == actual
