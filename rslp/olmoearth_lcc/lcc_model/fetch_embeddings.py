"""Fetch OlmoEarth v1.3 annual embeddings (2017-2025) for every LCC training window.

Reads s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr on data.source.coop (anonymous)
and writes the window's 128x128 crop of all 9 years x 128 dims as the int8 raster layer
``v13_embeddings`` (bands e<year>_<dim>, nodata -128), then marks the layer completed.

The store uses one group per UTM zone in its northern CRS with a continuous northing that goes
negative south of the equator, so a 327NN window maps to northing - 10,000,000 m.

The dataset's config.json must define the layer (see data/olmoearth_lcc/lcc_model/
config.json). Windows whose layer is already completed are skipped unless --overwrite.

Usage:
    python -m rslp.olmoearth_lcc.lcc_model.fetch_embeddings DS_PATH [--workers W] \
        [--shard i --num_shards n] [--limit N] [--overwrite]
"""
import argparse
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import zarr
from rslearn.dataset import Dataset
from rslearn.utils.raster_array import RasterArray
from upath import UPath

from .transforms import EMBEDDING_DIMS, EMBEDDING_KEY, EMBEDDING_NODATA, EMBEDDING_YEARS

STORE = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
STORAGE_OPTIONS = {"anon": True, "endpoint_url": "https://data.source.coop"}
YEARS = EMBEDDING_YEARS
DIMS = EMBEDDING_DIMS
NODATA = EMBEDDING_NODATA
LAYER = EMBEDDING_KEY

_zones: dict = {}


def zone_arrays(root, zone: int):
    if zone not in _zones:
        g = root[f"utm{zone:02d}"]
        years = [int(y) for y in g["time"][:]]
        assert years == YEARS, years
        t = g.attrs["spatial:transform"]  # [a, b, x0, d, e, y0]
        assert t[0] == 10 and t[4] == -10, t
        _zones[zone] = (g["embeddings"], t[2], t[5])
    return _zones[zone]


def fetch(root, window):
    crs = window.projection.crs.to_epsg()
    assert window.projection.x_resolution == 10 and window.projection.y_resolution == -10
    if 32601 <= crs <= 32660:
        zone, south = crs - 32600, False
    elif 32701 <= crs <= 32760:
        zone, south = crs - 32700, True
    else:
        raise ValueError(f"not a UTM CRS: {crs}")
    arr, x0, y0 = zone_arrays(root, zone)
    px0, py0, px1, py1 = window.bounds
    # Pixel (px, py) has CRS coords (10 px, -10 py); store row r has northing y0 - 10 r.
    north_top = -10 * py0 - (10_000_000 if south else 0)
    r0 = int(round((y0 - north_top) / 10))
    c0 = int(round((10 * px0 - x0) / 10))
    h, w = py1 - py0, px1 - px0
    out = np.full((len(YEARS), DIMS, h, w), NODATA, dtype=np.int8)
    H, W = arr.shape[2], arr.shape[3]
    rr0, rr1, cc0, cc1 = max(r0, 0), min(r0 + h, H), max(c0, 0), min(c0 + w, W)
    if rr0 < rr1 and cc0 < cc1:
        out[:, :, rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0] = arr[:, :, rr0:rr1, cc0:cc1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds_path")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    args = ap.parse_args()

    root = zarr.open_group(
        zarr.storage.FsspecStore.from_url(STORE, storage_options=STORAGE_OPTIONS, read_only=True),
        mode="r",
    )
    ds = Dataset(UPath(args.ds_path))
    layer_cfg = ds.layers[LAYER]
    band_set = layer_cfg.band_sets[0]
    fmt = band_set.instantiate_raster_format()
    windows = ds.load_windows(workers=32)
    windows = sorted(windows, key=lambda w: (w.group, w.name))[args.shard :: args.num_shards]
    todo = [w for w in windows if args.overwrite or not w.is_layer_completed(LAYER)]
    print(f"{len(windows)} windows, {len(todo)} to fetch", flush=True)
    if args.limit:
        todo = todo[: args.limit]

    def one(window):
        t = time.time()
        data = fetch(root, window)
        with window.data.open_layer_writer(LAYER) as writer:
            writer.write_raster(
                band_set.bands, fmt, window.projection, window.bounds,
                RasterArray(chw_array=data.reshape(len(YEARS) * DIMS, *data.shape[2:])),
            )
        window.mark_layer_completed(LAYER)
        valid = (data[:, 0] != NODATA).mean(axis=(1, 2))
        return window.group, window.name, time.time() - t, [round(float(v), 3) for v in valid]

    done = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(one, w): w for w in todo}
        for f in as_completed(futs):
            w = futs[f]
            try:
                g, n, dt, valid = f.result()
                done += 1
                print(json.dumps({"ok": f"{g}/{n}", "s": round(dt, 1), "valid": valid}), flush=True)
            except Exception:
                fail += 1
                print(json.dumps({"fail": f"{w.group}/{w.name}", "err": traceback.format_exc()[-800:]}), flush=True)
            if (done + fail) % 100 == 0:
                print(f"PROGRESS {done} ok {fail} fail {time.time()-t0:.0f}s", flush=True)
    print(f"DONE {done} ok {fail} fail {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
