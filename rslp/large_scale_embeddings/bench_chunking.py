"""Measure what a chunk shape costs to read, over a small real area.

Why this exists
---------------
The store fixes three things and two of them are not evidence-based. The shard is
pinned at 2048 px by the write path: one prediction window writes one object, which is
what makes concurrent writes safe without locking, so it is not a free parameter unless
that changes. The chunk's two dimensions are free, and today's values were chosen thinly:

- ``DEFAULT_CHUNK_SIZE = 256`` came from one comparison against 512 on one access
  pattern. 128 and 1024 were never tried.
- ``DEFAULT_BAND_CHUNK = 64`` is well founded, but only for bytes. It matches the width
  the model is trained to emit, so one chunk is one usable vector. What it costs in
  requests and index size at other depths has been computed, never measured.

This module measures both axes against real access patterns on real data, so the choice
stops being an argument and becomes a table.

What it does not measure, on purpose
------------------------------------
The zoomed-out continental view is not in here. That read is served by the PCA pyramid
in ``pca_v1.zarr``, whose levels exist precisely so a wide extent touches a bounded
number of chunks; asking ``embeddings.zarr`` for a continent is not an access pattern
anyone should have. Benchmarking it here would produce a large number that argues for a
chunk shape nothing needs.

Compression level is also fixed, at the ``DEFAULT_ZSTD_LEVEL`` the store now uses. It was
settled offline by recompressing real chunks (see the table in ``zarr_store.py``), and
mixing it into this sweep would triple the write cost to re-confirm a known answer. One
control variant carries the old level so the in-situ number can be checked against the
offline one.

The design
----------
Prediction is not re-run. A block of finished shards is read out of an existing store
once and rewritten into one variant store per chunk shape, so every variant holds
byte-identical embeddings and the only difference between them is layout. That makes the
comparison clean and makes the whole experiment cost a rewrite rather than a run.

Area: ``BLOCK_SHARDS`` squared shards, 3 x 3 by default, which is 6,144 px or 61.44 km
on a side. Three is the smallest number that is actually meaningful here, for two
reasons. The 20 km AOI pattern is exactly one shard wide, so at 2 x 2 it can only be
placed shard-aligned or corner-straddling, and at 3 x 3 there is also a centre shard
fully surrounded by written neighbours, which is the ordinary case in a global store. And
a single shard's compression ratio varies with its terrain, so a one-shard measurement
would report that terrain rather than the layout.

Reads are placed deliberately off-alignment. A benchmark that happens to align every
read to a chunk boundary flatters large chunks, and no real AOI is drawn that way.

Usage
-----
    python -m rslp.main large_scale_embeddings bench_build_variants \
        --source_store_path gs://BUCKET/.../embeddings.zarr \
        --out_prefix gs://BUCKET/bench/chunking_v1

    python -m rslp.main large_scale_embeddings bench_measure \
        --out_prefix gs://BUCKET/bench/chunking_v1

Build is the expensive half: 4.8 GB of array per variant, 17 variants, so about 55 GB
written and roughly an hour of one core per variant in compression. It parallelises over
variants trivially, one process each. Measure is cheap and can be re-run freely.
"""

import json
import time
from typing import Any

import numpy as np
from upath import UPath

from rslp.log_utils import get_logger

from .zarr_store import (
    DEFAULT_SHARD_SIZE,
    EMBEDDINGS_ARRAY,
    init_store,
    zone_group_name,
)

logger = get_logger(__name__)

# The axes to sweep. Shard stays 2048: see the module docstring.
SPATIAL_CHUNKS = (128, 256, 512, 1024)
BAND_CHUNKS = (16, 32, 64, 128)

# One extra variant at the pre-2026-09 compression level, so the offline recompression
# measurement can be confirmed against a store read through the real path.
CONTROL_ZSTD_LEVEL = 1
CONTROL_SHAPE = (256, 64)

# Edge of the source block, in shards. See the module docstring for why three.
BLOCK_SHARDS = 3

# Where to cut the block out of the source store. These are shard indices, not pixels.
# This 3 x 3 in utm36 is fully written Kenya land: every one of the nine objects is
# ~390 MB, against 537 MB uncompressed, so none of them is mostly nodata. A block with
# empty shards would measure compression of zeros.
SOURCE_ZONE = 36
SOURCE_SHARD_ROW = 449
SOURCE_SHARD_COL = 33

# Dimensions in the source store, and the Matryoshka width a client would actually read.
FULL_DIMS = 128
MATRYOSHKA_DIMS = 64


def variant_name(chunk_size: int, band_chunk: int, zstd_level: int) -> str:
    """Directory name for one variant store.

    Args:
        chunk_size: inner chunk spatial size.
        band_chunk: dimensions per inner chunk along the band axis.
        zstd_level: compression level.

    Returns:
        a name encoding all three, so a listing of the prefix is self-describing.
    """
    return f"sp{chunk_size}_d{band_chunk}_z{zstd_level}.zarr"


def variant_grid(zstd_level: int) -> list[tuple[int, int, int]]:
    """Every (chunk_size, band_chunk, zstd_level) to build.

    Args:
        zstd_level: the level the store writes today, used for the whole grid.

    Returns:
        the sweep, plus one control at the older compression level.
    """
    grid = [(sp, d, zstd_level) for sp in SPATIAL_CHUNKS for d in BAND_CHUNKS]
    control = (*CONTROL_SHAPE, CONTROL_ZSTD_LEVEL)
    if control not in grid:
        grid.append(control)
    return grid


# --------------------------------------------------------------------------- patterns
# Reads are anchored at deliberately awkward offsets: a benchmark that aligns every read
# to a chunk boundary measures the best case for large chunks and nothing a user does.
# The offsets below are prime-ish odd numbers inside the block, so no read starts on a
# 128, 256, 512 or 1024 boundary.
_OFF_Y = 1301
_OFF_X = 907

# Where the 20 km AOI starts. It is exactly one shard wide, so its placement decides
# whether it touches one object or four, and the first attempt here got that backwards:
# anchoring it on the block's centre put it at pixel 2048 in a 3-shard block, which is
# a shard boundary, so it filled one shard exactly and measured the best case. This
# places it half a shard past the first boundary instead, then skews it off every chunk
# size under test.
_AOI_SKEW = 173
_AOI_OFF = DEFAULT_SHARD_SIZE + DEFAULT_SHARD_SIZE // 2 + _AOI_SKEW


def access_patterns(block_px: int) -> list[dict[str, Any]]:
    """The reads to measure, as offsets within the copied block.

    Args:
        block_px: edge of the copied block in pixels.

    Returns:
        one dict per pattern, with a name, a pixel window and a dimension count.
    """
    if _AOI_OFF + DEFAULT_SHARD_SIZE > block_px:
        raise ValueError(
            f"a {block_px} px block is too small for the straddling 20 km AOI read, "
            f"which spans {_AOI_OFF} to {_AOI_OFF + DEFAULT_SHARD_SIZE}; "
            f"BLOCK_SHARDS must be at least 3"
        )
    return [
        {
            # The explorer's click. The pattern compression punishes hardest, because a
            # whole chunk moves to answer 128 bytes.
            "name": "point, 128 dims",
            "dims": FULL_DIMS,
            "y": _OFF_Y,
            "x": _OFF_X,
            "h": 1,
            "w": 1,
        },
        {
            "name": "point, 64 dims",
            "dims": MATRYOSHKA_DIMS,
            "y": _OFF_Y,
            "x": _OFF_X,
            "h": 1,
            "w": 1,
        },
        {
            # A field or a lake margin: 1 km at 10 m.
            "name": "1 km area, 128 dims",
            "dims": FULL_DIMS,
            "y": _OFF_Y,
            "x": _OFF_X,
            "h": 100,
            "w": 100,
        },
        {
            # A Matryoshka AOI read, one shard wide but straddling four of them, which
            # is what an AOI drawn without knowledge of the grid looks like.
            "name": "20 km AOI, 64 dims",
            "dims": MATRYOSHKA_DIMS,
            "y": _AOI_OFF,
            "x": _AOI_OFF,
            "h": DEFAULT_SHARD_SIZE,
            "w": DEFAULT_SHARD_SIZE,
        },
        {
            # A transect: a road, a river, a flight line. Long and thin is the shape
            # that punishes large spatial chunks hardest, and it is missing from the
            # comparison the current 256 was chosen on.
            "name": "40 km transect, 64 dims",
            "dims": MATRYOSHKA_DIMS,
            "y": _OFF_Y,
            "x": _OFF_X,
            "h": 16,
            "w": 4096,
        },
    ]


# ------------------------------------------------------------------------ measurement
# zarr type-checks the store it is handed, so this has to be a real Store subclass and
# not a delegating wrapper. Built at import time so the import of zarr stays inside the
# functions that need it, as elsewhere in this package.
def _counting_store_class() -> Any:
    """Build the counting-store class against the installed zarr.

    Returns:
        a Store subclass that records every range request made through it.
    """
    import zarr.abc.store

    class CountingStore(zarr.abc.store.Store):  # type: ignore[misc]
        """Wraps a zarr store and records every range request made through it.

        Bytes are counted as they come back off the wire, so this measures what a read
        actually costs rather than what the layout implies it should. The two have
        already diverged once: zarr does not coalesce adjacent sub-chunk ranges, which
        made band granularity look free when it costs a round trip per extra chunk.
        """

        def __init__(self, inner: Any) -> None:
            """Wrap a store.

            Args:
                inner: the store to delegate to.
            """
            super().__init__(read_only=True)
            self.inner = inner
            self.log: list[tuple[str, int]] = []

        async def get(self, key: str, prototype: Any, byte_range: Any = None) -> Any:
            """Fetch one range, recording its size.

            Args:
                key: object key.
                prototype: zarr buffer prototype.
                byte_range: the range requested, or None for the whole object.

            Returns:
                whatever the wrapped store returns.
            """
            buf = await self.inner.get(key, prototype, byte_range)
            self.log.append((key, len(buf) if buf is not None else 0))
            return buf

        async def get_partial_values(
            self, prototype: Any, key_ranges: list[tuple[str, Any]]
        ) -> Any:
            """Fetch several ranges, recording each.

            Args:
                prototype: zarr buffer prototype.
                key_ranges: the (key, range) pairs requested.

            Returns:
                whatever the wrapped store returns.
            """
            out = await self.inner.get_partial_values(prototype, key_ranges)
            for (key, _), buf in zip(key_ranges, out):
                self.log.append((key, len(buf) if buf is not None else 0))
            return out

        async def exists(self, key: str) -> bool:
            """Whether the wrapped store holds this key.

            Args:
                key: object key.

            Returns:
                whether it exists.
            """
            return await self.inner.exists(key)

        async def set(self, *args: Any, **kwargs: Any) -> None:
            """Refuse writes: this store exists to measure reads.

            Args:
                args: ignored.
                kwargs: ignored.

            Raises:
                NotImplementedError: always.
            """
            raise NotImplementedError("CountingStore is read-only")

        async def delete(self, key: str) -> None:
            """Refuse deletes: this store exists to measure reads.

            Args:
                key: ignored.

            Raises:
                NotImplementedError: always.
            """
            raise NotImplementedError("CountingStore is read-only")

        @property
        def supports_writes(self) -> bool:
            """Whether writes are supported.

            Returns:
                False.
            """
            return False

        @property
        def supports_deletes(self) -> bool:
            """Whether deletes are supported.

            Returns:
                False.
            """
            return False

        @property
        def supports_partial_writes(self) -> bool:
            """Whether partial writes are supported.

            Returns:
                False.
            """
            return False

        @property
        def supports_listing(self) -> bool:
            """Whether listing is supported.

            Returns:
                whether the wrapped store supports it.
            """
            return bool(self.inner.supports_listing)

        def list(self) -> Any:
            """Delegate listing.

            Returns:
                the wrapped store's iterator.
            """
            return self.inner.list()

        def list_prefix(self, prefix: str) -> Any:
            """Delegate prefix listing.

            Args:
                prefix: key prefix.

            Returns:
                the wrapped store's iterator.
            """
            return self.inner.list_prefix(prefix)

        def list_dir(self, prefix: str) -> Any:
            """Delegate directory listing.

            Args:
                prefix: key prefix.

            Returns:
                the wrapped store's iterator.
            """
            return self.inner.list_dir(prefix)

        def __eq__(self, other: object) -> bool:
            """Identity comparison, which is all zarr needs of a store.

            Args:
                other: object to compare against.

            Returns:
                whether this is the same wrapper.
            """
            return self is other

        def __hash__(self) -> int:
            """Hash by identity, to match __eq__.

            Returns:
                the identity hash.
            """
            return id(self)

        def summary(self) -> dict[str, Any]:
            """Totals for the reads recorded so far.

            Returns:
                bytes moved, requests made, and distinct objects touched.
            """
            return {
                "bytes": sum(n for _, n in self.log),
                "requests": len(self.log),
                "objects": len({key for key, _ in self.log}),
            }

    return CountingStore


def _open_counted(store_path: str, zone_number: int) -> tuple[Any, Any]:
    """Open a variant's embedding array through a counting store.

    Args:
        store_path: the variant store.
        zone_number: UTM zone whose group to read.

    Returns:
        the array and the counter wrapped around its store.
    """
    import zarr
    from zarr.storage import FsspecStore

    counted = _counting_store_class()(FsspecStore.from_url(store_path, read_only=True))
    group = zarr.open_group(counted, mode="r")
    array = group[f"{zone_group_name(zone_number)}/{EMBEDDINGS_ARRAY}"]
    return array, counted


def measure_one(
    store_path: str,
    pattern: dict[str, Any],
    zone_number: int,
    origin_y: int,
    origin_x: int,
    repeats: int = 3,
) -> dict[str, Any]:
    """Run one access pattern against one variant and record what it cost.

    The store is reopened for every repeat. zarr caches a shard index per array handle,
    so reusing one handle would measure the second read of a warm index and hide a cost
    every cold client pays.

    Args:
        store_path: the variant store.
        pattern: one entry from access_patterns().
        zone_number: UTM zone to read.
        origin_y: pixel row the copied block starts at in the variant.
        origin_x: pixel column the copied block starts at in the variant.
        repeats: how many times to run it; the fastest is reported.

    Returns:
        bytes, requests, objects and the best wall clock.
    """
    best: dict[str, Any] | None = None
    for _ in range(repeats):
        array, counted = _open_counted(store_path, zone_number)
        y = origin_y + pattern["y"]
        x = origin_x + pattern["x"]
        counted.log.clear()
        started = time.perf_counter()
        block = np.asarray(
            array[
                0,
                : pattern["dims"],
                y : y + pattern["h"],
                x : x + pattern["w"],
            ]
        )
        elapsed = time.perf_counter() - started
        result = counted.summary()
        result["seconds"] = elapsed
        result["wanted_bytes"] = int(block.size)
        if best is None or elapsed < best["seconds"]:
            best = result
    assert best is not None
    return best


def measure(
    out_prefix: str,
    zone_number: int = SOURCE_ZONE,
    block_shards: int = BLOCK_SHARDS,
    repeats: int = 3,
    results_path: str | None = None,
) -> list[dict[str, Any]]:
    """Replay every access pattern against every variant and report the table.

    Args:
        out_prefix: prefix the variants were built under.
        zone_number: UTM zone the block was copied into.
        block_shards: edge of the copied block in shards.
        repeats: repeats per measurement; the fastest is kept.
        results_path: optional path to write the rows to as JSON.

    Returns:
        one row per (variant, pattern).
    """
    block_px = block_shards * DEFAULT_SHARD_SIZE
    origin_y = SOURCE_SHARD_ROW * DEFAULT_SHARD_SIZE
    origin_x = SOURCE_SHARD_COL * DEFAULT_SHARD_SIZE
    patterns = access_patterns(block_px)

    prefix = UPath(out_prefix)
    variants = sorted(p.name for p in prefix.iterdir() if p.name.endswith(".zarr"))
    if not variants:
        raise ValueError(
            f"no variant stores under {out_prefix}; run build_variants first"
        )
    logger.info("measuring %d variant(s) x %d pattern(s)", len(variants), len(patterns))

    rows: list[dict[str, Any]] = []
    for name in variants:
        for pattern in patterns:
            got = measure_one(
                store_path=f"{out_prefix.rstrip('/')}/{name}",
                pattern=pattern,
                zone_number=zone_number,
                origin_y=origin_y,
                origin_x=origin_x,
                repeats=repeats,
            )
            row = {"variant": name, "pattern": pattern["name"], **got}
            # The number the whole sweep is about: bytes moved over bytes wanted.
            row["amplification"] = got["bytes"] / max(got["wanted_bytes"], 1)
            rows.append(row)
            logger.info(
                "%-22s %-24s %8.2f MB  %3d req  %2d obj  %6.2fs  %8.1fx",
                name,
                pattern["name"],
                got["bytes"] / 1e6,
                got["requests"],
                got["objects"],
                got["seconds"],
                row["amplification"],
            )

    if results_path is not None:
        with UPath(results_path).open("w") as f:
            json.dump(rows, f, indent=2)
        logger.info("wrote %d row(s) to %s", len(rows), results_path)
    return rows


# ------------------------------------------------------------------------------ build
def build_variants(
    source_store_path: str,
    out_prefix: str,
    model_url: str,
    source_data: list[str],
    zone_number: int = SOURCE_ZONE,
    block_shards: int = BLOCK_SHARDS,
    zstd_level: int | None = None,
    only: str | None = None,
) -> None:
    """Copy one block of finished shards into a store per chunk shape.

    Every variant receives byte-identical embeddings, read once from the source and
    written once per variant, so a difference between variants can only be layout.

    Args:
        source_store_path: a finished store to copy from.
        out_prefix: directory to create the variant stores under.
        model_url: passed through to the variants' metadata.
        source_data: passed through to the variants' metadata.
        zone_number: UTM zone to copy.
        block_shards: edge of the block to copy, in shards.
        zstd_level: level for the sweep. Defaults to the store's current default.
        only: build just this one variant name, for running the sweep in parallel.
    """
    import zarr

    from .zarr_store import DEFAULT_ZSTD_LEVEL

    level = DEFAULT_ZSTD_LEVEL if zstd_level is None else zstd_level
    grid = variant_grid(level)
    if only is not None:
        grid = [g for g in grid if variant_name(*g) == only]
        if not grid:
            raise ValueError(f"{only} is not one of {[variant_name(*g) for g in grid]}")

    source = zarr.open_array(
        f"{source_store_path}/{zone_group_name(zone_number)}/{EMBEDDINGS_ARRAY}",
        mode="r",
    )
    time_len = source.shape[0]
    origin_y = SOURCE_SHARD_ROW * DEFAULT_SHARD_SIZE
    origin_x = SOURCE_SHARD_COL * DEFAULT_SHARD_SIZE

    targets = []
    for chunk_size, band_chunk, variant_level in grid:
        name = variant_name(chunk_size, band_chunk, variant_level)
        path = f"{out_prefix.rstrip('/')}/{name}"
        if not UPath(path).exists():
            init_store(
                store_path=path,
                zone_numbers=[zone_number],
                years=list(range(2000, 2000 + time_len)),
                model_url=model_url,
                source_data=source_data,
                resolution=10,
                tile_size=32768,
                dimensions=FULL_DIMS,
                chunk_size=chunk_size,
                shard_size=DEFAULT_SHARD_SIZE,
                band_chunk=band_chunk,
                zstd_level=variant_level,
            )
        targets.append(
            (
                name,
                zarr.open_array(
                    f"{path}/{zone_group_name(zone_number)}/{EMBEDDINGS_ARRAY}",
                    mode="r+",
                ),
            )
        )

    # One shard at a time, read once and written to every variant. A shard is 537 MB
    # uncompressed, so this holds one of those and not the whole block.
    for shard_row in range(block_shards):
        for shard_col in range(block_shards):
            y = origin_y + shard_row * DEFAULT_SHARD_SIZE
            x = origin_x + shard_col * DEFAULT_SHARD_SIZE
            started = time.perf_counter()
            block = np.asarray(
                source[
                    :,
                    :,
                    y : y + DEFAULT_SHARD_SIZE,
                    x : x + DEFAULT_SHARD_SIZE,
                ]
            )
            logger.info(
                "read shard %d/%d at row %d col %d in %.0fs",
                shard_row * block_shards + shard_col + 1,
                block_shards * block_shards,
                y,
                x,
                time.perf_counter() - started,
            )
            for name, target in targets:
                started = time.perf_counter()
                target[:, :, y : y + DEFAULT_SHARD_SIZE, x : x + DEFAULT_SHARD_SIZE] = (
                    block
                )
                logger.info(
                    "  wrote %-22s in %.0fs", name, time.perf_counter() - started
                )
