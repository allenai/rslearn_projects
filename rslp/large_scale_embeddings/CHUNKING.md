Chunk Shape Read Costs
======================

Measured evidence for `DEFAULT_CHUNK_SIZE`, `DEFAULT_BAND_CHUNK` and
`DEFAULT_ZSTD_LEVEL` in `zarr_store.py`.

**These are fixed when `init_store` runs and cannot be changed afterwards.** Zarr cannot
re-chunk an array in place, so a different choice means rewriting every object in the
archive. That is why this benchmark gates creating a store rather than following it.

The shard is not free to choose: it is pinned to the prediction window so that one
window writes one object, which is what makes concurrent writes safe without locking.


How it was measured
-------------------

`tools/bench_chunking.py`, run 2026-10-06 against the global v1.3 archive:

    python -m rslp.main large_scale_embeddings bench_build_variants \
        --source_store_path gs://BUCKET/geozarr_global_v1.3/.../embeddings.zarr \
        --out_prefix gs://BUCKET/large_scale_embeddings/bench_chunking_20261005 \
        --time_index 8

    python -m rslp.main large_scale_embeddings bench_measure \
        --out_prefix gs://BUCKET/large_scale_embeddings/bench_chunking_20261005 \
        --results_path gs://BUCKET/.../bench_chunking_20261005/results.json

Prediction is not re-run. A 3 x 3 block of finished shards is read out of the source
once and rewritten into each variant, so every variant holds byte-identical embeddings
and the only difference is layout. Reads are anchored off every chunk boundary under
test; a benchmark that aligns its reads measures the best case for large chunks and
describes no AOI anyone draws.

Source block: utm14 shard row 312, col 25, time index 8 (2025). CONUS land in south
Texas, chosen from 2,540 candidate blocks whose nine shards are within 8% of each other
in size, as the one closest to the median. It compresses to 0.786, matching the
archive's own median. The 2026-09-05 sweep used Kenya, which compresses to about 0.73,
so its byte figures were optimistic. Content varies by a wide margin and sets every
number here, so the source is worth choosing rather than inheriting.

Band is no longer swept. The encoder emits two Matryoshka widths, 64 and 128
(`DEFAULT_MATRYOSHKA_DIMS`), so `d64` is correct by construction: a 64-dim read is
exactly one chunk and a 128-dim read exactly two. `d128` is kept as a control only.


Results
-------

Bytes moved and requests issued, at zstd 3 unless noted. Stored is the whole 3 x 3
block, 4,832 MB uncompressed.

| variant | stored | ratio | point 64d | point 128d | 20 km AOI 64d | 40 km transect |
|---|---|---|---|---|---|---|
| `sp64_d64` | 4,134 MB | 0.856 | 0.3 MB / 2 | 0.5 MB / 2 | 481.4 MB / 71 | 36.7 MB / 36 |
| `sp128_d64` | 4,207 MB | 0.871 | 0.9 MB / 2 | 1.8 MB / 2 | 475.5 MB / 59 | 30.8 MB / 36 |
| **`sp256_d64`** | **3,799 MB** | **0.786** | **3.2 MB / 2** | **6.5 MB / 2** | **267.8 MB / 85** | **56.6 MB / 20** |
| `sp512_d64` | 3,507 MB | 0.726 | 12.0 MB / 2 | 23.9 MB / 3 | 307.5 MB / 29 | 110.9 MB / 12 |
| `sp256_d128` | 3,799 MB | 0.786 | 6.5 MB / 2 | 6.5 MB / 2 | 534.7 MB / 49 | 113.0 MB / 20 |
| `sp256_d64_z1` | 3,913 MB | 0.810 | 3.4 MB / 2 | 6.7 MB / 2 | 278.4 MB / 85 | 58.3 MB / 20 |

The AOI column above was measured under zarr 3.4.0 defaults. Read the next section
before drawing anything from it.


The reader decides the AOI cost, not the layout
-----------------------------------------------

**zarr coalesces adjacent ranged reads, and that doubles what small chunks move.**
`array.sharding_coalesce_max_gap_bytes` (1 MiB by default since 3.3) merges nearby
chunk ranges into one GET and reads the gaps between them. The same variants, same
bytes on disk, on the 20 km AOI:

| variant | zarr 3.4.0 default | gap 0 | zarr 3.2.1 | chunks spanned |
|---|---|---|---|---|
| `sp64_d64` | 481.4 MB / 71 req | 245.3 MB / 1093 req | 245.3 MB / 1093 req | 1,089 |
| `sp128_d64` | 475.5 MB / 59 req | 265.4 MB / 293 req | 265.4 MB / 293 req | 289 |
| `sp256_d64` | 267.8 MB / 85 req | 267.8 MB / 85 req | 267.8 MB / 85 req | 81 |

Three things follow:

- **With coalescing off, every row matches theory exactly**, request counts included
  (1,089 chunks plus 4 shard-index reads is 1,093). Setting the gap to 0 reproduces
  3.2.1 byte for byte, so the knob fully controls the behaviour.
- **`sp256` is unaffected.** At 256 px the chunks are already larger than the coalescing
  window, so it reads 267.8 MB over 85 requests under every configuration tested. Its
  cost is a property of the archive.
- **Below 256 px the cost is a property of the client.** A reader that upgrades zarr and
  does not set the knob pays 1.8x more bytes, silently.

`sharding_coalesce_max_bytes: 0` changes nothing further; the gap setting alone does it.


What the numbers say
--------------------

**Band 64 is settled.** `sp256_d128` moves 534.7 MB on the AOI against `sp256_d64`'s
267.8, exactly double, because a 64-dim read has to fetch both band chunks. Nothing
narrower than 64 is worth testing now that 64 is the narrowest width the encoder emits.

**zstd 3 is settled.** Against level 1 at the same shape it stores 2.9% less and moves
4.0% fewer bytes on the AOI, with no time penalty. The saving comes off storage *and*
off every read, since a range request moves compressed bytes. Level 3 is also what the
comparable AlphaEarth mosaic uses, so a like-for-like read comparison stays honest.

**512 loses.** It costs 12.0 MB for a point read and 110.9 MB on the transect, roughly
double 256 on both, and its 8% storage saving does not pay for that.

**Spatial 64 against 256 is a real tradeoff, not a measurement question.** 128 is
dominated by 64 on all three axes and can be set aside.

| | `sp64_d64` | `sp256_d64` |
|---|---|---|
| Point read, 64 dims | **0.3 MB** | 3.2 MB |
| 20 km AOI, coalescing off | **245.3 MB** | 267.8 MB |
| 20 km AOI requests | 1,093 | **85** |
| 20 km AOI if the knob is lost | 481.4 MB | **267.8 MB** |
| Storage | +8.8% | **baseline** |

64 is 11x cheaper on the point read, which is the pattern that dominates fitting a
classifier from scattered labels, and it needs no configuration to get that. It is also
slightly cheaper on the AOI once coalescing is off. It costs 8.8% storage across the
archive, 13x the request count, and a client-side setting someone has to keep set.

256 needs nothing kept set. Its numbers did not move across four client configurations.

**Amplification is inherent and large either way.** A single-point 64-dim read against
`sp256_d64` moves 3.2 MB to deliver 64 bytes, about 50,000x; `sp64_d64` still amplifies
about 4,000x. zstd frames are not seekable, so answering one pixel moves its whole
chunk. A point-query workload wants a different artifact, not a different chunk shape:
band-last dimension order with uncompressed inner chunks would make a point read a
64-byte ranged GET, at about 24% more storage.


Status
------

Unchanged and evidenced: `DEFAULT_BAND_CHUNK = 64`, `DEFAULT_ZSTD_LEVEL = 3`.

`DEFAULT_CHUNK_SIZE` is still 256 and the choice is open pending two answers from the
Studio explorer, the archive's main interactive reader: whether it can pin
`array.sharding_coalesce_max_gap_bytes = 0` and regression-test that it stays pinned,
and whether point reads or AOI reads dominate its cost in practice. If it can hold the
setting, 64 is the better choice and costs 8.8% storage.

Re-run this before creating a store if the embedding dimensionality changes, if the
emitted Matryoshka widths move, or if the dominant access pattern turns out to be
something other than the point and AOI reads assumed here.

What this deliberately does not measure is the zoomed-out continental view. That is
served by the PCA pyramid, whose levels exist so a wide extent touches a bounded number
of chunks; asking `embeddings.zarr` for a continent is not an access pattern anyone
should have, and benchmarking it would argue for a chunk shape nothing needs.


Why the 2026-09-05 sweep is superseded
--------------------------------------

That run concluded "spatial 256 is the right choice" largely on `sp128_d64_z3` moving
410.8 MB on the AOI, 77% worse than 256. Re-measuring those same Kenya variants on
2026-10-05 under zarr 3.2.1 gave 223.9 MB over 293 requests, while `sp256_d64_z3`
reproduced byte for byte at 232,197,373. The 410.8 figure was a coalescing read sitting
in a table whose other rows were not, so that sweep mixed two client behaviours and its
spatial conclusion does not hold. Its band and compression conclusions are unaffected
and are re-confirmed above.

It also swept band chunks 16 and 32, which were readable widths then and are not now.
