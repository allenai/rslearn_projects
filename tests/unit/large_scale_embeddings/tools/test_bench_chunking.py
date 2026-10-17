"""Guards on the chunking benchmark's design, not on its measurements."""

from rslp.large_scale_embeddings.tools.bench_chunking import (
    BAND_CHUNKS,
    BLOCK_SHARDS,
    CONTROL_BAND_SHAPE,
    CONTROL_SHAPE,
    CONTROL_ZSTD_LEVEL,
    SPATIAL_CHUNKS,
    access_patterns,
    variant_grid,
    variant_name,
)
from rslp.large_scale_embeddings.zarr_store import (
    DEFAULT_MATRYOSHKA_DIMS,
    DEFAULT_SHARD_SIZE,
)


def test_no_read_starts_on_a_chunk_boundary() -> None:
    """Every pattern must be misaligned to every spatial chunk size under test.

    A read that happens to start on a chunk boundary touches the fewest chunks it
    possibly could, which flatters large chunks and describes no AOI anyone draws. This
    caught the 20 km AOI sitting at pixel 2048 of a 3-shard block, one shard wide and
    perfectly shard-aligned, which is precisely the case the pattern exists to avoid.
    """
    block_px = BLOCK_SHARDS * DEFAULT_SHARD_SIZE
    for pattern in access_patterns(block_px):
        assert (
            pattern["y"] % DEFAULT_SHARD_SIZE != 0
        ), f"{pattern['name']} shard-aligned"
        for chunk in SPATIAL_CHUNKS:
            assert (
                pattern["y"] % chunk != 0
            ), f"{pattern['name']} aligns to {chunk} in y"
            assert (
                pattern["x"] % chunk != 0
            ), f"{pattern['name']} aligns to {chunk} in x"


def test_every_read_fits_inside_the_copied_block() -> None:
    """A pattern running off the block would read nodata and measure compression of it."""
    block_px = BLOCK_SHARDS * DEFAULT_SHARD_SIZE
    for pattern in access_patterns(block_px):
        assert pattern["y"] >= 0 and pattern["x"] >= 0, pattern["name"]
        assert (
            pattern["y"] + pattern["h"] <= block_px
        ), f"{pattern['name']} overruns in y"
        assert (
            pattern["x"] + pattern["w"] <= block_px
        ), f"{pattern['name']} overruns in x"


def test_the_aoi_pattern_straddles_four_shards() -> None:
    """The AOI read is only meaningful if it crosses shards.

    It is exactly one shard wide, so placed on the grid it would touch one object and
    measure the best case. The block has to be at least 2 shards across for the
    straddling placement to have data on all four sides.
    """
    block_px = BLOCK_SHARDS * DEFAULT_SHARD_SIZE
    aoi = next(p for p in access_patterns(block_px) if p["name"].startswith("20 km"))
    first = aoi["y"] // DEFAULT_SHARD_SIZE
    last = (aoi["y"] + aoi["h"] - 1) // DEFAULT_SHARD_SIZE
    assert last > first, "the AOI read sits inside a single shard row"
    assert BLOCK_SHARDS >= 2, "a 1-shard block cannot hold a straddling AOI read"


def test_the_grid_covers_the_swept_axis_and_carries_both_controls() -> None:
    """The sweep is the cross product, plus a band control and a compression control.

    The controls are not candidates: one shows what the wrong band chunk costs and the
    other confirms the compression level. Counting them into the cross product would
    let a control quietly become a recommendation.
    """
    grid = variant_grid(3)
    assert len(grid) == len(SPATIAL_CHUNKS) * len(BAND_CHUNKS) + 2
    assert (*CONTROL_SHAPE, CONTROL_ZSTD_LEVEL) in grid, "no compression control"
    assert (*CONTROL_BAND_SHAPE, 3) in grid, "no band control"
    assert len({variant_name(*g) for g in grid}) == len(grid), "duplicate variant names"


def test_the_band_axis_matches_the_matryoshka_widths() -> None:
    """Every swept band chunk must be a width the encoder actually emits.

    The 2026-09-05 sweep covered 16 and 32 because narrower prefixes were readable
    then. They no longer are, so sweeping them would spend hours of compression
    measuring reads nobody can issue.
    """
    for band in BAND_CHUNKS:
        assert band in DEFAULT_MATRYOSHKA_DIMS, (
            f"band chunk {band} is not one of the emitted widths "
            f"{DEFAULT_MATRYOSHKA_DIMS}"
        )


def test_the_shard_size_is_not_swept() -> None:
    """Shard is pinned by the write path, so it must not appear as an axis.

    One prediction window writes one object, which is what keeps concurrent writes on
    disjoint objects and needs no locking. Sweeping it here would benchmark a store the
    pipeline cannot safely write.
    """
    assert DEFAULT_SHARD_SIZE not in SPATIAL_CHUNKS or DEFAULT_SHARD_SIZE > max(
        SPATIAL_CHUNKS
    ), "a chunk as large as the shard makes sharding pointless"
    for chunk in SPATIAL_CHUNKS:
        assert DEFAULT_SHARD_SIZE % chunk == 0, (
            f"chunk {chunk} does not divide the shard, so a shard's chunk grid would "
            "be ragged"
        )
