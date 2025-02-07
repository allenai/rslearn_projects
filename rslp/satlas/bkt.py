"""Manage bucket files on GCS.

We bucket together small (10-200 KB) files at high zoom levels (e.g. zoom 13) into a
single file at a lower zoom level (e.g. zoom 9) to save on GCS insert fee.

This is similar to https://github.com/mactrem/com-tiles.

The .bkt is just a concatenation of the small files, which here we call items. This is
specialized for storing tiles, so each item falls on a grid at a particular zoom level
and is associated with a column and row. The bkt itself is on a grid at a zoom level
equal to or lower than that of its items, i.e., a coarser grid.

The bkt should contain every item (on the finer grid) that is contained within its tile
(on the coarser grid); if an item is missing, that should mean that there is just no
data there.

Then there will actually be a set of bkt files at the coarser zoom level to cover the
entire region.

We record the columns, rows, and byte offsets of items in a Google Cloud Bigtable
database. Readers can first query Bigtable, then make a range read request to GCS.
"""

import functools
import io
import multiprocessing.pool
import os
import struct
import time
from collections.abc import Generator
from dataclasses import dataclass
from enum import StrEnum
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

# Number of retries for reading from BigTable, since sometimes there are transient
# errors.
BIGTABLE_RETRIES = 8


@dataclass
class BktItemMetadata:
    """Metadata about an item (small file) stored within the bkt."""

    # The column and row of the item on the item (fine-grained) grid.
    col: int
    row: int

    # The byte offset and length of the item within the concatenated bkt file.
    offset: int
    length: int

    def pack(self) -> bytes:
        """Pack the metadata into bytes."""
        return struct.pack(">IIII", self.col, self.row, self.offset, self.length)

    @staticmethod
    def unpack(b: bytes) -> "BktItemMetadata":
        """Unpack a BktItemMetadata from a 16-byte string."""
        col, row, offset, length = struct.unpack(">IIII", b)
        return BktItemMetadata(col, row, offset, length)


class BktInserter:
    """A helper class that stores metadata about bkt files so it can be inserted later.

    The BktInserter is a separate class from BktWriter so that it can be pickled to
    support use with multiprocessing.

    Normal usage is to create BktWriter in parallel, call get_inserter to get the
    BktInserter objects, then collect them in the main thread and finally call
    BktInserter.run. This way the worker threads do not need to make additional
    connections to Bigtable, which really becomes a problem.
    """

    def __init__(
        self,
        item_metadatas: list[BktItemMetadata],
        bkt_fname: str,
        bkt_zoom: int,
        zoom: int,
    ):
        """Create a new BktInserter.

        It stores information that will be needed to insert into Bigtable.

        Args:
            item_metadatas: metadata about items within the bkt.
            bkt_fname: the filename where the bkt will be written.
            bkt_zoom: the zoom level of the bkt.
            zoom: the zoom level of the tiles within the bkt.
        """
        self.item_metadatas = item_metadatas
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
        for item_metadata in self.item_metadatas:
            buf.write(item_metadata.pack())
        db_row = bkt_files_table.direct_row(self.bkt_fname)
        db_row.set_cell(b"d", b"d", buf.getvalue())
        db_row.commit()


class BktWriter:
    """Writer for bkt files.

    Call add to add one item at a time. Then call get_bytes and write the data to GCS.
    Finally call insert (or use get_inserter and then pass to main thread and call run
    on the BktInserter object).

    Callers must write to GCS before Bigtable so that when clients read from Bigtable
    they can expect the files to already be written.

    upload_bkts can help with inserting multiple BktWriters, e.g.:

        bkt_writers = {}
        for col, row in item_tiles:
            # bkt_factor is 2^(item zoom - bkt zoom), i.e. the difference in scale
            # between the item grid and the bkt grid.
            bkt_tile = (col//bkt_factor, row//bkt_factor)
            if bkt_tile not in bkt_writers:
                bkt_writers[bkt_tile] = bkt.BktWriter()
            contents = ...
            bkt_writers[bkt_tile].add(col, row, contents)

        bkt_uploads = []
        for bkt_tile, bkt_writer in bkt_writers.items():
            out_fname = '.../{}/{}/{}.bkt'.format(bkt_zoom, bkt_tile[0], out_tile[1])
            bkt_uploads.append((bkt_writer, out_fname, args.out_zoom, args.zoom))
        # p is a multiprocessing.Pool.
        bkt.upload_bkts(bkt_files_table, p, bkt_uploads)
    """

    def __init__(self) -> None:
        """Create a new BktWriter."""
        self.item_metadatas: list[BktItemMetadata] = []
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
        self.item_metadatas.append(BktItemMetadata(col, row, offset, length))
        self.buf.write(bytes)
        self.offset += length

    def get_bytes(self) -> bytes:
        """Returns the bytes of the whole bkt file.

        This is what should be uploaded to GCS.
        """
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
        return BktInserter(self.item_metadatas, bkt_fname, bkt_zoom, zoom)

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
    """Get the GCS bucket where bkt files should be stored.

    This comes from the environment variables:
    - BKT_PROJECT_ID: GCP project
    - BKT_BUCKET_NAME: the GCS bucket within that project.
    """
    storage_client = storage.Client(project=os.environ["BKT_PROJECT_ID"])
    bucket = storage_client.bucket(os.environ["BKT_BUCKET_NAME"])
    return bucket


@functools.cache
def get_bigtable() -> google.cloud.bigtable.table.Table:
    """Get the BigTable table storing bkt metadata."""
    bigtable_client = bigtable.Client(project=os.environ["BKT_BIGTABLE_PROJECT_ID"])
    bigtable_instance = bigtable_client.instance(os.environ["BKT_BIGTABLE_INSTANCE_ID"])
    bkt_files_table = bigtable_instance.table("bkt_files")
    return bkt_files_table


class DecodeMode(StrEnum):
    """Mode indicating how items should be decoded when downloading in parallel.

    This is used in functions like download_bkts so that the worker processes can
    handle decoding rather than the caller needing to decode in the main thread.
    """

    # Decode it from image bytes to numpy array.
    IMAGE = "image"

    # Yield the bytes directly.
    RAW = "raw"


@dataclass
class BktDownloadRequest:
    """A request to download an item in a bkt file to pass to download_bkts."""

    # Name of bkt file for this job.
    bkt_fname: str

    # Column and row on item (fine-grained) grid to read.
    col: int
    row: int

    # Arbitrary metadata for use by caller.
    # It will be returned with the decoded item data.
    metadata: Any = None


def _download_bkt(
    bkt_fname: str,
    item_metadatas: list[BktItemMetadata],
    wanted: list[BktDownloadRequest],
    decode_mode: DecodeMode,
) -> list[tuple[Any, npt.NDArray | bytes]]:
    """Download tiles in a bkt file.

    Args:
        bkt_fname: the bkt filename in the bucket to download from.
        item_metadatas: the item metadatas for this bkt file.
        wanted: list of BktDownloadRequest that specify the tiles to download. Note
            that if a tile does not exist within the bkt, it will not be returned at
            all.
        decode_mode: how the items should be decoded.

    Returns:
        a list of (metadata, contents) where contents is a numpy array with
            DecodeMode.IMAGE or a byte string with DecodeMode.RAW.
    """
    bucket = get_bucket()
    output = []

    # Helper to postprocess an output based on the specified return mode.
    def add_output(metadata: Any, contents: npt.NDArray | bytes) -> None:
        if decode_mode == DecodeMode.IMAGE:
            buf = io.BytesIO(contents)
            image = skimage.io.imread(buf)
            output.append((metadata, image))

        elif decode_mode == DecodeMode.RAW:
            output.append((metadata, contents))

        else:
            raise ValueError(f"invalid decode mode {decode_mode}")

    # Convert item_metadatas to a map from (col, row) -> (offset, length).
    idx_map = {(m.col, m.row): (m.offset, m.length) for m in item_metadatas}

    # Filter for just the requested tiles that actually exist.
    # The caller should assume that tiles that are not returned simply don't have data.
    wanted = [request for request in wanted if (request.col, request.row) in idx_map]

    if len(wanted) == 1:
        # If there is just one requested item within this bkt, we can do a range read
        # to read only that item.
        request = wanted[0]
        offset, length = idx_map[(request.col, request.row)]
        blob = bucket.blob(bkt_fname)
        contents = blob.download_as_bytes(start=offset, end=offset + length)
        add_output(request.metadata, contents)

    elif len(wanted) > 1:
        # Otherwise, we read the entire bkt file and then extract the segments
        # corresponding to the requested items.
        blob = bucket.blob(bkt_fname)
        bkt_bytes = blob.download_as_bytes()
        for request in wanted:
            offset, length = idx_map[(request.col, request.row)]
            contents = bkt_bytes[offset : offset + length]
            add_output(request.metadata, contents)

    # We return a list of (metadata, contents) from this bkt file.
    # In download_from_bkt, it will combine these tuples across all of the bkt files.
    return output


def _bkt_retry_loop(
    bkt_files_table: google.cloud.bigtable.table.Table, bkt_fname: str
) -> google.cloud.bigtable.row.PartialRowData:
    """Retry loop to read the bkt_fname metadata from BigTable.

    This is used because sometimes there are transient errors reading.
    """

    def attempt_read() -> google.cloud.bigtable.row.PartialRowData:
        return bkt_files_table.read_row(
            bkt_fname,
            filter_=google.cloud.bigtable.row_filters.CellsColumnLimitFilter(1),
        )

    for _ in range(BIGTABLE_RETRIES):
        try:
            return attempt_read()
        except Exception as e:
            logger.warning(
                f"got error reading bkt_files_table for {bkt_fname} (trying again): {e}"
            )
            time.sleep(1)

    # One last read, if it fails then we let the exception go.
    return attempt_read()


def download_from_bkt(
    bkt_files_table: google.cloud.bigtable.table.Table,
    download_requests: list[BktDownloadRequest],
    pool: multiprocessing.pool.Pool | None = None,
    decode_mode: DecodeMode = DecodeMode.RAW,
) -> Generator[tuple[Any, npt.NDArray | bytes], None, None]:
    """Download tile contents in parallel from several bkt files.

    Args:
        bkt_files_table: the BigTable table containing byte offsets.
        download_requests: list of BktDownloadRequest indicating the bkt filenames to
            read from along with the item tiles to read. Download requests from the
            same bkt_fname will be grouped together so we don't read the same bkt
            file multiple times.
        pool: the multiprocessing pool to use for parallelism, or None to read in
            current process.
        decode_mode: how the items should be decoded.

    Yields:
        the (metadata, contents) tuples across all of the jobs. Only items that exist
            in the bkt files will be returned; if non-existing items are requested,
            they would be skipped and caller should assume they have no data.
    """
    # Get tiles to read grouped by each distinct bkt_fname.
    requests_by_bkt_fname: dict[str, list[BktDownloadRequest]] = {}
    for request in download_requests:
        if request.bkt_fname not in requests_by_bkt_fname:
            requests_by_bkt_fname[request.bkt_fname] = []
        requests_by_bkt_fname[request.bkt_fname].append(request)

    # Read from BigTable to identify the offset and length of each requested
    # (col, row) item.
    # We use this to populate a list of jobs (arguments to pass to _download_bkt
    # helper).
    bkt_jobs: list[dict[str, Any]] = []
    for bkt_fname, requests in requests_by_bkt_fname.items():
        db_row = _bkt_retry_loop(bkt_files_table, bkt_fname)

        # Ignore requested files that don't exist.
        if not db_row:
            continue

        # Skip 8-byte header with bkt_zoom/zoom.
        encoded_indexes = db_row.cell_value("d", b"d")[8:]

        item_metadatas = []
        for i in range(0, len(encoded_indexes), 16):
            item_metadatas.append(BktItemMetadata.unpack(encoded_indexes[i : i + 16]))
        bkt_jobs.append(
            dict(
                bkt_fname=bkt_fname,
                item_metadatas=item_metadatas,
                wanted=requests,
                decode_mode=decode_mode,
            )
        )

    if pool is None:
        for job in bkt_jobs:
            for metadata, image in _download_bkt(**job):
                yield (metadata, image)
    else:
        outputs = star_imap_unordered(pool, _download_bkt, bkt_jobs)
        for output in outputs:
            for metadata, image in output:
                yield (metadata, image)


def upload_bkt(bkt_fname: str, contents: bytes) -> None:
    """Upload a bkt file to GCS bucket.

    This is primarily intended to be used as a helper function for multiprocessing.

    Args:
        bkt_fname: the bkt filename within the bucket.
        contents: the data to upload.
    """
    bucket = get_bucket()
    blob = bucket.blob(bkt_fname)
    blob.upload_from_string(contents)


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
    bkt_files_table = get_bigtable()

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
