Large-Scale Embeddings
======================

This project computes OlmoEarth embeddings over large areas (up to global scale). The
embeddings are:

- 10 m/pixel, in the appropriate UTM projection for each location.
- 128-dimensional, L2-normalized, and quantized to int8 (see Quantization below).
- Computed from one year of input imagery starting at a user-provided reference
  timestamp.

There are two input variants (`EmbeddingInputs`), which produce different embeddings
and so must be written to different output paths:

- `S2`: twelve monthly Sentinel-2 L2A mosaics.
- `S2_S1`: the above, plus twelve monthly Sentinel-1 RTC mosaics (converted from
  linear intensities to dB). Sentinel-1 is best-effort: where it is unavailable, the
  embeddings are computed from Sentinel-2 alone. Sentinel-2 coverage is required.

The configuration files are in `data/large_scale_embeddings/`: `s2.json`/`s2_s1.json`
are the rslearn dataset configs (imagery comes from the OlmoEarth Datasets sources)
and `s2.yaml`/`s2_s1.yaml` are the model configs (which reference the OlmoEarth
checkpoint on WEKA).


How It Works
------------

The world is divided into 32768x32768-pixel tiles in each UTM zone, and each tile is
one unit of work (one queue job). The prediction pipeline for a tile creates
2048x2048-pixel windows in a scratch rslearn dataset, materializes the input mosaics,
runs the model, and uploads one GeoTIFF per window to `out_path`, named
`{crs}_{x}_{y}.tif` (x/y are the pixel offsets of the window in the UTM projection at
10 m/pixel). GeoTIFFs are uncompressed (the int8 embeddings are high-entropy, so
compression only slows down writes) and tiled with 512x512 blocks, with nodata
value -128.

To limit duplicated work where UTM zones overlap, tiles and windows are skipped unless
they intersect their zone's canonical 6-degree wedge (see `tiling.py`). Windows that
are entirely ocean (per `global_land_mask`) or too close to 0/180 longitude (where
mosaics are unreliable) are also skipped.

When a tile finishes, a marker file `{crs}_{x}_{y}.json` is written to
`completed_path` recording the tile's projection, bounds, and time range, plus which
windows were written and which were skipped (`written`, `skipped_no_data` for windows
without Sentinel-2 coverage, `skipped_longitude`, and `num_filtered_crops` for
wedge/ocean-filtered windows). Tiles with existing markers are excluded when writing
jobs and skipped by the prediction pipeline, so the pipeline is idempotent and jobs
can safely be re-enqueued to retry failures.


Running One Tile Locally
------------------------

This requires a GPU and access to the OlmoEarth checkpoint (e.g. run on a machine with
WEKA mounted). From the rslearn_projects root:

    python -m rslp.main large_scale_embeddings predict \
        --inputs S2 \
        --projection_json '{"crs": "EPSG:32610", "x_resolution": 10, "y_resolution": -10}' \
        --bounds '[32768, -557056, 65536, -524288]' \
        --time_range '["2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"]' \
        --out_path gs://BUCKET/embeddings/s2/2024/ \
        --completed_path gs://BUCKET/embeddings/s2/2024_completed/

`bounds` can be any box whose extents are multiples of 2048 (it does not have to be a
32768x32768 tile). `time_range` is `(T, T)` where T is the reference timestamp; the
dataset config derives the twelve monthly mosaics over the year following T. By
default the scratch rslearn dataset is placed in a temporary directory and deleted;
pass `--scratch_path /path/to/scratch/` to keep it for debugging.


Running at Scale
----------------

Jobs are distributed via a Beaker queue and processed by `rslp.common` workers.

1. Build and push a Beaker image containing rslearn_projects (with the
   `global-land-mask` dependency included).

2. Write jobs to a Beaker queue, one per uncompleted tile:

        python -m rslp.main large_scale_embeddings write_jobs \
            --inputs S2 \
            --timestamp '2025-01-01T00:00:00+00:00' \
            --out_path gs://ai2-olmoearth-embeddings-us-central1/large_scale_embeddings/20270721/regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1/ps1_ws16_s2/2025/ \
            --completed_path gs://ai2-olmoearth-embeddings-us-central1/large_scale_embeddings/20270721/regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1/ps1_ws16_s2/2025_completed/ \
            --queue_name favyen/rslp-large-scale-embeddings-queue

   Without additional arguments this enumerates all land tiles globally (~8,700).
   Options to limit the extent:

   - `--epsg_code 32610`: only one UTM zone.
   - `--wgs84_bounds '[-125.0, 45.0, -116.0, 49.0]'`: only tiles intersecting these
     WGS84 bounds.
   - `--geojson_fname data/large_scale_embeddings/initial_regions.geojson`: only
     tiles intersecting a feature in the given WGS84 GeoJSON file (the included
     `initial_regions.geojson` covers Washington, Montana, Ukraine, Thailand, and
     points in Greenland and coastal Antarctica; 88 tiles).
   - `--count 10`: randomly sample this many tiles.

3. Launch workers on Beaker (WEKA must be mounted for the checkpoint). The
   OlmoEarth Datasets data source needs `OEDATASETS_API_URL` (plain env var) and
   `DATASETS_API_TOKEN` (bearer token, read from the `LCC_DATASETS_API_TOKEN`
   Beaker secret which must exist in the `ai2/earth-systems` workspace):

        python -m rslp.main common launch \
            --image_name favyen/rslpomp20260721b \
            --queue_name favyen/rslp-large-scale-embeddings-queue \
            --num_workers 4 \
            --gpus 1 \
            --priority urgent \
            --cluster '["ai2/jupiter","ai2/ceres"]' \
            --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}' \
            --extra_env_vars '{"OEDATASETS_API_URL": "https://datasets.olmoearth.allenai.org"}' \
            --extra_env_secrets '{"DATASETS_API_TOKEN": "LCC_DATASETS_API_TOKEN"}'

Progress can be monitored by counting marker files in `completed_path`. To retry
failed tiles, simply run `write_jobs` again: completed tiles are excluded.

Remember to use different `out_path`/`completed_path` per input variant and per
reference timestamp.


Quantization
------------

Embeddings are L2-normalized and then quantized following the AlphaEarth scheme (see
`model.py`): `quantized = round(sign(x) * |x|^0.5 * 127.5)` clipped to [-127, 127],
with -128 reserved for nodata. To recover approximate float embeddings:

```python
import numpy as np

def dequantize(v: np.ndarray) -> np.ndarray:
    x = v.astype(np.float32) / 127.5
    return np.sign(x) * np.abs(x) ** 2.0
```

Pixels where all Sentinel-2 mosaics are empty are set to -128 in all bands.
