"""Manage bucket files on GCS.

We bucket together small (10-200 KB) files at high zoom levels (e.g. zoom 13) into a
single file at a lower zoom level (e.g. zoom 9) to save on GCS insert fee.

This is similar to https://github.com/mactrem/com-tiles.

The .bkt is just a concatenation of the small files.

We record the byte offsets in a Google Cloud Bigtable database.
"""

import functools
import io
import multiprocessing.pool
import os
import struct
import time
from collections.abc import Generator
from typing import Any

import google.cloud.bigtable.row
import google.cloud.bigtable.row_filters
import google.cloud.bigtable.table
import numpy.typing as npt
import skimage.io
from google.cloud import bigtable, storage
from rslearn.utils.mp import star_imap_unordered

from rslp.log_utils import get_logger

logger = get_logger(__name__)


class BktInserter:
    """A helper class that inserts metadata about bkt files into the database.

    The BktInserter is a separate class from BktWriter so that it can be pickled to
    support use with multiprocessing.
    """

    def __init__(
        self,
        indexes: list[tuple[int, int, int, int]],
        bkt_fname: str,
        bkt_zoom: int,
        zoom: int,
    ):
        """Create a new BktInserter.

        Args:
            indexes: the byte offsets of the files within the bkt. It is a list of
                (col, row, offset, length) tuples.
            bkt_fname: the filename where the bkt will be written.
            bkt_zoom: the zoom level of the bkt.
            zoom: the zoom level of the tiles within the bkt.
        """
        self.indexes = indexes
        self.bkt_fname = bkt_fname
        self.bkt_zoom = bkt_zoom
        self.zoom = zoom

    def run(self, bkt_files_table: google.cloud.bigtable.table.Table) -> None:
        """Insert the metadata into BigTable.

        Args:
            bkt_files_table: the BigTable object
        """
        # Row key in the table is just the bkt fname.
        # Value is [4 byte bkt_zoom][4 byte zoom][indexes].
        # [indexes] is list of indexes encoded as [4 byte col][4 byte row][4 byte offset][4 byte length].
        buf = io.BytesIO()
        buf.write(struct.pack(">II", self.bkt_zoom, self.zoom))
        for col, row, offset, length in self.indexes:
            buf.write(struct.pack(">IIII", col, row, offset, length))
        db_row = bkt_files_table.direct_row(self.bkt_fname)
        db_row.set_cell(b"d", b"d", buf.getvalue())
        db_row.commit()


class BktWriter:
    """Writer for bkt files."""

    def __init__(self) -> None:
        """Create a new BktWriter."""
        self.indexes: list[tuple[int, int, int, int]] = []
        self.buf = io.BytesIO()
        self.offset = 0

    def add(self, col: int, row: int, bytes: bytes) -> None:
        """Add a file to the bkt.

        Args:
            col: the tile column.
            row: the tile row.
            bytes: the data at this tile.
        """
        offset = self.offset
        length = len(bytes)
        self.indexes.append((col, row, offset, length))
        self.buf.write(bytes)
        self.offset += length

    def get_bytes(self) -> bytes:
        """Returns the bytes of the whole bkt file."""
        return self.buf.getvalue()

    def get_inserter(self, bkt_fname: str, bkt_zoom: int, zoom: int) -> "BktInserter":
        """Creates a BktInserter that manages inserting the byte offsets to BigTable.

        Args:
            bkt_fname: the filename where the bkt will be written.
            bkt_zoom: the zoom level of the bkt file.
            zoom: the zoom of the tiles within the bkt file.

        Returns:
            a corresponding BktInserter
        """
        return BktInserter(self.indexes, bkt_fname, bkt_zoom, zoom)

    def insert(
        self,
        bkt_files_table: google.cloud.bigtable.table.Table,
        bkt_fname: str,
        bkt_zoom: int,
        zoom: int,
    ) -> None:
        """Insert the byte offsets for this bkt to BigTable.

        Args:
            bkt_files_table: the BigTable table object.
            bkt_fname: the filename where the bkt will be written.
            bkt_zoom: the zoom level of the bkt file.
            zoom: the zoom of the tiles within the bkt file.
        """
        self.get_inserter(bkt_fname, bkt_zoom, zoom).run(bkt_files_table)


@functools.cache
def get_bucket() -> storage.Bucket:
    """Get the GCS bucket where bkt files should be stored."""
    storage_client = storage.Client(project=os.environ["BKT_PROJECT_ID"])
    bucket = storage_client.bucket(os.environ["BKT_BUCKET_NAME"])
    return bucket


def download_bkt(
    bkt_fname: str,
    idx_map: dict[tuple[int, int], tuple[int, int]],
    wanted: list[tuple[int, int, Any]],
    mode: str,
) -> list[tuple[Any, npt.NDArray | bytes]]:
    """Download tiles in a bkt file.

    Args:
        bkt_fname: the bkt filename in the bucket to download from.
        idx_map: map from tile (col, row) to (offset, length).
        wanted: list of tiles to download. It should be a list of (col, row, metadata)
            where metadata can be arbitrary data used by the caller that will be
            returned with the tile data (which will be emitted in arbitrary order).
            Note that if a tile does not exist within the bkt, it will not be returned
            at all.
        mode: either "image" to decode image and return numpy array, or "raw" to return
            the byte string directly.

    Returns:
        a list of (metadata, contents) where contents is a numpy array if mode is
            "image" or a byte string if mode is "raw".
    """
    bucket = get_bucket()
    output = []

    # Helper to postprocess an output based on the specified return mode.
    def add_output(metadata: Any, contents: npt.NDArray | bytes) -> None:
        if mode == "image":
            buf = io.BytesIO(contents)
            image = skimage.io.imread(buf)
            output.append((metadata, image))

        elif mode == "raw":
            output.append((metadata, contents))

        else:
            raise ValueError(f"invalid mode {mode}")

    wanted = [
        (col, row, metadata) for col, row, metadata in wanted if (col, row) in idx_map
    ]

    if len(wanted) == 1:
        col, row, metadata = wanted[0]
        offset, length = idx_map[(col, row)]
        blob = bucket.blob(bkt_fname)
        contents = blob.download_as_bytes(start=offset, end=offset + length)
        add_output(metadata, contents)

    elif len(wanted) > 1:
        blob = bucket.blob(bkt_fname)
        bkt_bytes = blob.download_as_bytes()
        for col, row, metadata in wanted:
            offset, length = idx_map[(col, row)]
            contents = bkt_bytes[offset : offset + length]
            add_output(metadata, contents)

    return output


# Parallel download from various bkt files.
# Jobs is a list of (bkt_fname, col, row, metadata).
# download_from_bkt is a generator that will yield (metadata, bytes) for each provided job.
def download_from_bkt(
    bkt_files_table: google.cloud.bigtable.table.Table,
    pool: multiprocessing.pool.Pool | None,
    jobs: list[tuple[str, int, int, Any]],
    mode: str = "raw",
) -> Generator[tuple[Any, npt.NDArray | bytes], None, None]:
    """Download tile contents in parallel from several bkt files.

    Args:
        bkt_files_table: the BigTable table containing byte offsets.
        pool: the multiprocessing pool to use for parallelism, or None to read in
            current process.
        jobs: list of (bkt_fname, col, row, metadata) to work through. Jobs referencing
            the same bkt_fname will be grouped together so we don't read the same bkt
            file multiple times.
        mode: the return mode (see download_bkt).

    Yields:
        the (metadata, contents) tuples across all of the jobs.
    """
    # Get indexes associated with each distinct bkt_fname.
    by_bkt_fname: dict[str, list[tuple[int, int, Any]]] = {}
    for bkt_fname, col, row, metadata in jobs:
        if bkt_fname not in by_bkt_fname:
            by_bkt_fname[bkt_fname] = []
        by_bkt_fname[bkt_fname].append((col, row, metadata))

    bkt_jobs: list[dict[str, Any]] = []
    for bkt_fname, wanted in by_bkt_fname.items():
        # Use retry loop since we seem to get error reading from BigTable occasionally.
        def bkt_retry_loop() -> google.cloud.bigtable.row.PartialRowData:
            for _ in range(8):
                try:
                    db_row = bkt_files_table.read_row(
                        bkt_fname,
                        filter_=google.cloud.bigtable.row_filters.CellsColumnLimitFilter(
                            1
                        ),
                    )
                    return db_row
                except Exception as e:
                    print(
                        f"got error reading bkt_files_table for {bkt_fname} (trying again): {e}"
                    )
                    time.sleep(1)
            raise Exception(
                f"repeatedly failed to read bkt_files_table for {bkt_fname}"
            )

        db_row = bkt_retry_loop()

        # Ignore requested files that don't exist.
        if not db_row:
            continue
        # Skip 8-byte header with bkt_zoom/zoom.
        encoded_indexes = db_row.cell_value("d", b"d")[8:]

        indexes = {}
        for i in range(0, len(encoded_indexes), 16):
            col, row, offset, length = struct.unpack(
                ">IIII", encoded_indexes[i : i + 16]
            )
            indexes[(col, row)] = (offset, length)
        bkt_jobs.append(
            dict(
                bkt_fname=bkt_fname,
                idx_map=indexes,
                wanted=wanted,
                mode=mode,
            )
        )

    if pool is None:
        for job in bkt_jobs:
            for metadata, image in download_bkt(**job):
                yield (metadata, image)
    else:
        outputs = star_imap_unordered(pool, download_bkt, bkt_jobs)
        for output in outputs:
            for metadata, image in output:
                yield (metadata, image)


def upload_bkt(bkt_fname: str, contents: bytes) -> None:
    """Upload a bkt file to GCS bucket.

    Args:
        bkt_fname: the bkt filename within the bucket.
        contents: the data to upload.
    """
    bucket = get_bucket()
    blob = bucket.blob(bkt_fname)
    blob.upload_from_string(contents)


# Tuples is list of (bkt_writer, bkt_fname, bkt_zoom, zoom).
def upload_bkts(
    bkt_files_table: google.cloud.bigtable.table.Table,
    pool: multiprocessing.pool.Pool,
    jobs: list[tuple[BktWriter, str, int, int]],
) -> None:
    """Upload several bkt files to GCS in parallel.

    Args:
        bkt_files_table: the BigTable table to store byte offsets.
        pool: a multiprocessing pool for parallelism.
        jobs: list of (bkt_writer, bkt_fname, bkt_zoom, zoom) tuples. bkt_writer is the
            BktWriter where the bkt contents and metadata are stored. bkt_fname is the
            path where the bkt should be written. bkt_zoom in the zoom level of the bkt
            file. zoom is the zoom level of tiles within the bkt.
    """
    # Upload. We upload first since reader will assume that anything existing in
    # BigTable already exists on GCS.
    upload_jobs: list[tuple[str, bytes]] = []
    for bkt_writer, bkt_fname, bkt_zoom, zoom in jobs:
        upload_jobs.append((bkt_fname, bkt_writer.get_bytes()))
    outputs = star_imap_unordered(pool, upload_bkt, upload_jobs)
    for _ in outputs:
        pass
    # Now we insert the metadata.
    for bkt_writer, bkt_fname, bkt_zoom, zoom in jobs:
        bkt_writer.insert(
            bkt_files_table=bkt_files_table,
            bkt_fname=bkt_fname,
            bkt_zoom=bkt_zoom,
            zoom=zoom,
        )


def make_bkt(src_dir: str, dst_path: str) -> None:
    """Make a bkt file from the specified local source directory.

    The source directory must contain files of the form zoom/col/row.ext (the extension
    is ignored).

    A single bkt file is created, so the zoom level of the bkt is always 0.

    Args:
        src_dir: the local directory to turn into a single bkt file.
        dst_path: the bkt filename in the bkt GCS bucket to write to. It must have a
            {zoom} placeholder where the zoom goes.
    """
    bucket = get_bucket()
    bigtable_client = bigtable.Client(project=os.environ["BKT_BIGTABLE_PROJECT_ID"])
    bigtable_instance = bigtable_client.instance(os.environ["BKT_BIGTABLE_INSTANCE_ID"])
    bkt_files_table = bigtable_instance.table("bkt_files")

    for zoom_str in os.listdir(src_dir):
        zoom_dir = os.path.join(src_dir, zoom_str)
        if not os.path.isdir(zoom_dir):
            continue
        zoom = int(zoom_str)
        logger.debug(
            "make_bkt(%s, %s): start collecting files at zoom level %d",
            src_dir,
            dst_path,
            zoom,
        )

        # Read all files at this zoom level from local path into bkt (in memory).
        bkt_writer = BktWriter()
        num_files = 0
        for col_str in os.listdir(zoom_dir):
            col_dir = os.path.join(zoom_dir, col_str)
            col = int(col_str)
            for fname in os.listdir(col_dir):
                row = int(fname.split(".")[0])
                num_files += 1
                with open(os.path.join(col_dir, fname), "rb") as f:
                    contents = f.read()
                    bkt_writer.add(col, row, contents)
        logger.debug(
            "make_bkt(%s, %s): processed %d files at zoom %d",
            src_dir,
            dst_path,
            num_files,
            zoom,
        )

        # Now upload to GCS.
        bkt_fname = dst_path.format(zoom=zoom)
        logger.debug(
            "make_bkt(%s, %s) uploading bkt for zoom level %d to %s",
            src_dir,
            dst_path,
            zoom,
            bkt_fname,
        )
        blob = bucket.blob(bkt_fname)
        blob.upload_from_string(bkt_writer.get_bytes())
        bkt_writer.insert(
            bkt_files_table=bkt_files_table,
            bkt_fname=bkt_fname,
            bkt_zoom=0,
            zoom=zoom,
        )
