"""Measure real compression ratios on OlmoEarth GeoZarr embeddings.

Reads whole 2048x2048x128 int8 blocks (one shard's worth) from the NYC store, then
for each (inner chunk size, codec) pair compresses every chunk exactly as Zarr would
and sums the bytes. Emits JSON so the cost spreadsheet cites measured numbers rather
than assumed ones.

Chunks are laid out (1, 128, c, c): a chunk always carries the full embedding vector,
matching the store (zarr_store.py: chunks=(1, dimensions, chunk_size, chunk_size)).
"""

import argparse
import json
import time
from typing import Any

import numcodecs
import numpy as np
import zarr

STORE = (
    "gs://ai2-olmoearth-embeddings-us-central1/large_scale_embeddings/"
    "geozarr_nyc_2020_2025_20260729/"
    "regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1/s2_ps1_ws16_overlap4/s2.zarr"
)
ZONE = "utm18"
YEAR = 2024
SHARD = 2048
NODATA = -128

# (col, row) origins of populated 2048 tiles, from the explorer's config.ts.
BLOCKS = [(57344, 499712), (55296, 497664), (59392, 495616)]

CHUNK_SIZES = [128, 256, 512, 1024, 2048]

# Codecs measured at every chunk size (fast enough to run over a full 512 MiB block).
FAST = {
    "zstd-1": numcodecs.Zstd(level=1),
    "zstd-3": numcodecs.Zstd(level=3),
    "blosc-zstd-1-shuffle": numcodecs.Blosc(
        cname="zstd", clevel=1, shuffle=numcodecs.Blosc.SHUFFLE
    ),
    "lz4": numcodecs.LZ4(),
    "gzip-1": numcodecs.GZip(level=1),
}
# Slower codecs, measured only at the store's current 256 px chunk.
SLOW_AT_256 = {
    "zstd-9": numcodecs.Zstd(level=9),
    "zstd-19": numcodecs.Zstd(level=19),
    "zlib-6": numcodecs.Zlib(level=6),
}


def chunk_bytes(block: np.ndarray, c: int, codec: Any) -> tuple[int, int]:
    """Compress every (1,128,c,c) chunk of a (128,H,W) block; return (raw, compressed)."""
    raw = comp = 0
    h, w = block.shape[1], block.shape[2]
    for y in range(0, h, c):
        for x in range(0, w, c):
            piece = np.ascontiguousarray(block[:, y : y + c, x : x + c])
            raw += piece.nbytes
            comp += len(codec.encode(piece))
    return raw, comp


def main(out: str) -> None:
    """Measure every codec at every chunk size and write the results to out."""
    root = zarr.open_group(store=STORE, mode="r")
    group = root[ZONE]
    years = list(np.asarray(group["time"][:]))
    t = years.index(YEAR)
    print(f"years={years} using index {t} for {YEAR}", flush=True)
    arr = group["embeddings"]
    print(f"array shape={arr.shape} dtype={arr.dtype}", flush=True)

    results = []
    for col, row in BLOCKS:
        t0 = time.time()
        block = np.asarray(arr[t, :, row : row + SHARD, col : col + SHARD])
        nodata_frac = float((block[0] == NODATA).mean())
        print(
            f"\nblock col={col} row={row}: {block.nbytes/2**20:.0f} MiB read in "
            f"{time.time()-t0:.0f}s, nodata={nodata_frac:.3%}",
            flush=True,
        )
        for c in CHUNK_SIZES:
            codecs = dict(FAST)
            if c == 256:
                codecs.update(SLOW_AT_256)
            for name, codec in codecs.items():
                s = time.time()
                raw, comp = chunk_bytes(block, c, codec)
                n_chunks = (SHARD // c) ** 2
                results.append(
                    dict(
                        block=[col, row],
                        nodata_frac=nodata_frac,
                        chunk_px=c,
                        codec=name,
                        chunks_per_shard=n_chunks,
                        raw_bytes=raw,
                        compressed_bytes=comp,
                        ratio=comp / raw,
                        # Zarr's shard index carries 16 bytes per inner chunk.
                        index_bytes=16 * n_chunks,
                        seconds=round(time.time() - s, 1),
                    )
                )
                print(
                    f"  chunk={c:>4} {name:<22} ratio={comp/raw:.4f} "
                    f"({comp/2**20:.1f} MiB) {time.time()-s:.0f}s",
                    flush=True,
                )
        del block

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {len(results)} measurements to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="compression_measurements.json",
        help="where to write the measurements JSON",
    )
    main(parser.parse_args().out)
