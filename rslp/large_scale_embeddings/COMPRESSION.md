Codec Choice
============

Measured evidence for the codec behind `compressors=[ZstdCodec(...)]` in
`zarr_store.py`. The *level* and the chunk shape are settled separately in
CHUNKING.md; this note covers only the choice of zstd over the alternatives.

Like the chunk shape, this is fixed when `init_store` runs. Changing it means reading
and rewriting every object in the archive, though not re-running inference.


How it was measured
-------------------

`tools/measure_compression.py`, run 2026-08-18:

    python rslp/large_scale_embeddings/tools/measure_compression.py \
        --out tools/compression_measurements.json

Three fully populated 2048 x 2048 x 128 int8 blocks (0% nodata) are read from the NYC
store and compressed chunk by chunk, exactly as Zarr would, across eight codecs and
five inner chunk sizes. 84 measurements, checked in at
`tools/compression_measurements.json`.

Only chunk size and codec move the ratio. Shard size cannot, because compression
happens per chunk.

The blocks are real embeddings rather than synthetic data, which matters here: the
codecs separate on how much byte-level redundancy the data has, and int8 embeddings
have very little.


Results
-------

Mean compression ratio across the three blocks, lower is better. Encode throughput is
the mean over all chunk sizes.

| codec | 128 | 256 | 512 | 1024 | 2048 | MB/s |
|---|---|---|---|---|---|---|
| `blosc-zstd-1-shuffle` | 0.7118 | 0.6713 | 0.6776 | 0.6874 | 0.7006 | 617 |
| `zstd-19` | | 0.6516 | | | | 5 |
| `zlib-6` | | 0.6707 | | | | 28 |
| `gzip-1` | 0.6937 | 0.6818 | 0.6818 | 0.6878 | 0.6968 | 67 |
| `zstd-9` | | 0.6886 | | | | 78 |
| **`zstd-3`** | 0.7429 | **0.7068** | 0.6712 | 0.6754 | 0.6835 | 352 |
| `zstd-1` | 0.7622 | 0.7190 | 0.6801 | 0.6864 | 0.6973 | 783 |
| **`lz4`** | 0.9297 | **0.9274** | 0.9281 | 0.9302 | 0.9330 | 2044 |

Blank cells were measured at 256 px only, being too slow to sweep over a full 512 MiB
block at every size.

Encode timings were rounded to 0.1 s at collection, so the throughput column is coarse.
It is reported to separate the codecs by order of magnitude, not to rank neighbours.


What the numbers say
--------------------

**lz4 is not viable.** At the store's 256 px chunk it stores 31.2% more than zstd 3.
Per-block, it is worse than that sounds:

| block | lz4 ratio at 256 px |
|---|---|
| (57344, 499712) | 0.8454 |
| (59392, 495616) | 0.9343 |
| (55296, 497664) | **1.0026** |

On the third block lz4 output was larger than its input, at every chunk size tested.
That is the expected result for a match-only compressor on data with no repeated byte
sequences, which is what a quantized embedding is.

**lz4's speed advantage does not apply to this archive.** Its case is decode
throughput, but reads here are network-bound rather than CPU-bound: a single 2048 px
shard read from GCS took about 292 s during the 2026-11-02 corruption scan. Moving
31% more bytes over that link makes reads slower, not faster. The browser explorer
already loads a zstd wasm decoder without trouble.

**The entropy coders cluster.** gzip-1, zlib-6, zstd-3 and zstd-9 all land between
0.67 and 0.71 at 256 px. zstd wins on throughput at equal ratio, which is the whole
reason to prefer it, not a ratio advantage. zstd-19 buys 5.5% over zstd-3 at 1/70th
the encode speed.

**blosc + zstd-1 measured 6.6% smaller than plain zstd-1 at the same encode speed**,
and about 5% smaller than zstd-3. With int8 data the byte shuffle is a no-op, since
there is one byte per element, so the gain comes from blosc's internal blocking rather
than from shuffling. This has never been tested as a store setting and remains the one
open question here. It is worth considering only if the archive is being regenerated
anyway, since on its own it does not pay for a full rewrite.


What this costs
---------------

One global year, shard 2048 / chunk 256, Nearline us-central1, against 201.6 TB
uncompressed (land footprint measured in COASTAL_COVERAGE.md):

| codec | stored | Nearline |
|---|---|---|
| uncompressed | 201.6 TB | $24,186/yr |
| blosc + zstd-1 | 135.3 TB | $16,237/yr |
| **zstd-3 (current)** | **142.5 TB** | **$17,101/yr** |
| lz4 | 186.9 TB | $22,434/yr |

Choosing lz4 would have cost 44.5 TB and $5,333/yr per global year, and made every
read move 31% more bytes.


Caveats
-------

**The ratios are not the archive's ratios.** These chunks are `(1, 128, c, c)`, the
store's are `(1, 64, 256, 256)`, and the content is NYC rather than global. Compression
varies widely with content: the archive's own median block compresses to 0.786 at
zstd 3 (CHUNKING.md), against 0.7068 here. Use these numbers to compare codecs against
each other, and CHUNKING.md for what a level actually stores.

**Costs beyond compression are not here.** The full storage and write-cost model,
including coastal shard overshoot and Class A operation pricing, lives in the
spreadsheet at `~/src/scratch/embeddings-storage-costs/` and was not migrated. Its
operation prices were never confirmed against a primary source.


Status
------

    compressors = [ZstdCodec(level=DEFAULT_ZSTD_LEVEL)]   # zarr_store.py
    DEFAULT_ZSTD_LEVEL = 3

The live v1.3 archive confirms it, from `embeddings.zarr/utm52/embeddings/zarr.json`
on 2026-11-03:

    sharding_indexed  chunk_shape [1, 64, 256, 256]
      bytes
      zstd  {"level": 3, "checksum": false}

Re-run this before creating a store if the dtype changes, if quantization is replaced,
or if a codec worth testing appears in `numcodecs`. A change of embedding content alone
does not invalidate the ordering, only the absolute ratios.
