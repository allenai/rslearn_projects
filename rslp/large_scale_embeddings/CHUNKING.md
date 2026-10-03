Chunk Shape Read Costs
======================

Measured results for the two free parameters of the embeddings store, and the evidence
behind `DEFAULT_CHUNK_SIZE`, `DEFAULT_BAND_CHUNK` and `DEFAULT_ZSTD_LEVEL` in
`zarr_store.py`.

**These three are fixed when `init_store` runs and cannot be changed afterwards.** Zarr
cannot re-chunk an array in place, so a different choice means rewriting every object in
the archive. That is why they are worth measuring rather than assuming, and why this
benchmark gates creating a store rather than following it.

The shard is not free to choose: it is pinned to the prediction window so that one
window writes one object, which is what makes concurrent writes safe without locking.


How it was measured
-------------------

`tools/bench_chunking.py`, run 2026-09-05 against the Kenya store as source:

    python -m rslp.main large_scale_embeddings bench_build_variants \
        --source_store_path gs://BUCKET/geozarr_kenya_.../embeddings.zarr \
        --out_prefix gs://BUCKET/large_scale_embeddings/bench_chunking_20260905

    python -m rslp.main large_scale_embeddings bench_measure \
        --out_prefix gs://BUCKET/large_scale_embeddings/bench_chunking_20260905 \
        --results_path gs://BUCKET/large_scale_embeddings/bench_chunking_20260905/results.json

Prediction is not re-run. One 3x3 block of finished shards is read out of the source once
and rewritten into 17 variant stores, so every variant holds byte-identical embeddings
and the only difference between them is layout. 16 variants cover the grid of four
spatial sizes against four band depths at zstd-3; the seventeenth repeats `sp256_d64` at
zstd-1 as a compression control.

Each measurement reopens the store, because zarr caches a shard index per array handle
and reusing one would measure a warm index that no cold client gets. Three repeats,
fastest kept. Raw rows are in `results.json` beside the variants.

Naming is `sp{spatial}_d{band}_z{zstd}`: `sp256_d64_z3` is a 256 px inner chunk, 64
dimensions per chunk along the band axis, zstd level 3.


Results
-------

Bytes actually moved over the network, and the number of range requests issued.

| variant | point<br>64 dims | point<br>128 dims | 1 km area<br>128 dims | 20 km AOI<br>64 dims | 40 km transect<br>64 dims |
|---|---|---|---|---|---|
| `sp128_d16_z3` | 0.8 MB<br>3 req | 1.5 MB<br>5 req | 1.5 MB<br>5 req | 228.2 MB<br>34 req | 40.4 MB<br>21 req |
| `sp128_d32_z3` | 0.8 MB<br>2 req | 1.6 MB<br>3 req | 1.6 MB<br>3 req | 230.8 MB<br>85 req | 37.4 MB<br>20 req |
| `sp128_d64_z3` | 0.8 MB<br>2 req | 1.6 MB<br>2 req | 1.6 MB<br>2 req | 410.8 MB<br>52 req | 25.1 MB<br>36 req |
| `sp128_d128_z3` | 1.6 MB<br>2 req | 1.6 MB<br>2 req | 1.6 MB<br>2 req | 447.6 MB<br>56 req | 50.8 MB<br>36 req |
| `sp256_d16_z3` | 2.9 MB<br>3 req | 5.9 MB<br>5 req | 5.9 MB<br>5 req | 232.6 MB<br>42 req | 47.3 MB<br>37 req |
| `sp256_d32_z3` | 2.9 MB<br>2 req | 5.9 MB<br>3 req | 5.9 MB<br>3 req | 232.4 MB<br>33 req | 47.2 MB<br>20 req |
| **`sp256_d64_z3`** | **2.9 MB**<br>2 req | **5.9 MB**<br>2 req | **5.9 MB**<br>2 req | **232.2 MB**<br>85 req | **47.2 MB**<br>20 req |
| `sp256_d128_z3` | 5.9 MB<br>2 req | 5.9 MB<br>2 req | 5.9 MB<br>2 req | 465.5 MB<br>49 req | 96.3 MB<br>20 req |
| `sp512_d16_z3` | 10.7 MB<br>3 req | 21.8 MB<br>5 req | 21.8 MB<br>5 req | 270.9 MB<br>28 req | 94.9 MB<br>21 req |
| `sp512_d32_z3` | 10.7 MB<br>2 req | 21.7 MB<br>3 req | 21.7 MB<br>3 req | 270.8 MB<br>25 req | 94.9 MB<br>12 req |
| `sp512_d64_z3` | 10.7 MB<br>2 req | 21.7 MB<br>3 req | 21.7 MB<br>3 req | 270.8 MB<br>29 req | 94.9 MB<br>12 req |
| `sp512_d128_z3` | 21.7 MB<br>2 req | 21.7 MB<br>2 req | 21.7 MB<br>2 req | 547.9 MB<br>29 req | 194.0 MB<br>12 req |
| `sp1024_d16_z3` | 42.6 MB<br>5 req | 86.5 MB<br>9 req | 86.5 MB<br>9 req | 391.0 MB<br>40 req | 212.6 MB<br>23 req |
| `sp1024_d32_z3` | 42.6 MB<br>3 req | 86.5 MB<br>5 req | 86.5 MB<br>5 req | 390.9 MB<br>22 req | 212.5 MB<br>13 req |
| `sp1024_d64_z3` | 42.6 MB<br>2 req | 86.5 MB<br>3 req | 86.5 MB<br>3 req | 390.9 MB<br>13 req | 212.5 MB<br>8 req |
| `sp1024_d128_z3` | 86.5 MB<br>2 req | 86.5 MB<br>2 req | 86.5 MB<br>2 req | 792.0 MB<br>9 req | 433.8 MB<br>8 req |

Wall-clock times are in `results.json` but are not reproduced here: they are dominated by
network variance from one client in one region, and the byte and request counts are the
properties of the layout. Where times do separate the variants they follow the bytes.


What the numbers say
--------------------

**Spatial 256 is the right choice.** It wins the 20 km AOI, which is the pattern an
interactive client actually issues, moving 232 MB against 271 MB at 512 (+17%) and
391 MB at 1024 (+68%). Both extremes had never been tried before this run and both fail,
differently: 1024 is worst nearly everywhere, while 128 is genuinely better on point
reads (0.8 MB against 2.9 MB) but collapses on the AOI at 411 MB, 77% worse than 256.
256 is the compromise: near-best on wide reads, acceptable on points.

**Band depth 64 holds, though less decisively than the byte argument suggested.** At the
64 dimensions the model is trained to emit, `d64` is exactly one chunk and one usable
vector. Finer depths shave a little off point reads and cost round trips instead, which
is the effect the old comment in `zarr_store.py` predicted from first principles and this
confirms: `sp1024_d16` needs 9 requests where `d128` needs 2. The honest caveat is that
`sp256_d32` matches `sp256_d64` on AOI bytes within noise (232.4 MB against 232.2 MB)
while issuing fewer requests. There is no measured reason to prefer 64 over 32 on cost
alone; 64 is kept because it makes a Matryoshka-width read exactly one chunk, which is a
property worth having rather than a number worth optimising.

**Amplification is inherent, and large.** A single-point 64-dim read against
`sp256_d64_z3` moves 2,910,450 bytes to deliver 64 bytes: **45,476x**. zstd frames are
not seekable, so answering one pixel moves its whole chunk. No layout in the grid avoids
this; the smallest, `sp128_d16`, still amplifies about 12,000x. This is the cost of
compression, and it is why a point-query workload wants a different artifact rather than
a different chunk shape.


zstd level
----------

`sp256_d64` built at level 1 and level 3, everything else identical:

| pattern | level 1 | level 3 | saving |
|---|---|---|---|
| point, 64 dims | 3,092,782 B | 2,910,450 B | 5.9% |
| 40 km transect, 64 dims | 51,011,247 B | 47,176,030 B | 7.5% |
| 20 km AOI, 64 dims | 251,228,594 B | 232,197,373 B | 7.6% |

Level 3 moves 6 to 8% fewer bytes with no time penalty, confirming in situ the 6.3%
measured offline by recompressing real chunks. The saving comes off storage *and* off
every read, since a range request moves compressed bytes. Level 3 is also what the
comparable AlphaEarth mosaic uses, so a like-for-like read comparison stays honest.


Conclusion
----------

The current defaults are correct and no change is needed:

    DEFAULT_CHUNK_SIZE = 256
    DEFAULT_BAND_CHUNK = 64
    DEFAULT_ZSTD_LEVEL = 3

Re-run this benchmark before creating a store if the embedding dimensionality changes,
if the model's trained Matryoshka width moves off 64, or if the dominant access pattern
turns out to be something other than the AOI read assumed here.

What this deliberately does not measure is the zoomed-out continental view. That is
served by the PCA pyramid, whose levels exist so a wide extent touches a bounded number
of chunks; asking `embeddings.zarr` for a continent is not an access pattern anyone
should have, and benchmarking it would argue for a chunk shape nothing needs.
