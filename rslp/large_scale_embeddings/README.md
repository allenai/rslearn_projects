Large-Scale Embeddings
======================

This project computes OlmoEarth embeddings over large areas (up to global scale) and
writes them to a GeoZarr store following the geoemb embeddings-zarr-convention
(https://github.com/geo-embeddings/embeddings-zarr-convention). The embeddings are:

- 10 m/pixel (at the default `--patch_size 1`; in general patch_size x 10 m/pixel),
  in the appropriate UTM projection for each location.
- 128-dimensional, L2-normalized, and quantized to int8 (see Quantization below).
- Computed from one year of input imagery starting at a user-provided reference
  timestamp; multiple reference years form the store's time axis.

There are two input variants (`EmbeddingInputs`), which produce different embeddings
and so must be written to different stores:

- `S2`: twelve monthly Sentinel-2 L2A mosaics.
- `S2_S1`: the above, plus twelve monthly Sentinel-1 RTC mosaics (converted from
  linear intensities to dB). Sentinel-1 is best-effort: where it is unavailable, the
  embeddings are computed from Sentinel-2 alone. Sentinel-2 coverage is required.

The configuration files are in `data/large_scale_embeddings/`: `s2.json`/`s2_s1.json`
are the rslearn dataset configs (imagery comes from the OlmoEarth Datasets sources)
and `s2.yaml`/`s2_s1.yaml` are the model configs.

The model settings are provided as arguments to `write_jobs`/`predict` and override
the defaults in the model configs (they are recorded in each queue job, so workers
can process jobs with differing settings):

- `--checkpoint_path` (required): the OlmoEarth checkpoint to compute embeddings
  with, e.g.
  `/weka/dfive-default/helios/checkpoints/gabrielt/regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1/step560000`.
- `--patch_size` (default `1`): the encoder patch size, yielding one embedding per
  patch_size x patch_size pixels; the output rasters are at 1/patch_size of the
  10 m/pixel window resolution.
- `--window_size` (default `16`): the size of the crops the model operates on (much
  bigger fails with the 12 monthly inputs at patch_size=1 due to GPU memory
  constraints).
- `--overlap_size` (default `4`): overlap in pixels between adjacent crops, to
  mitigate embedding seams at crop boundaries. Must be a multiple of patch_size.
- `--compile_model` (default `true`): whether to compile the encoder transformer
  blocks.

Note that the checkpoint and the patch/window/overlap sizes all affect the resulting
embeddings, so each combination must use its own `store`/`completed_path` (like the
input variants).


Output Store
------------

The store is a Zarr v3 group using the geoemb `utm_zones` spatial layout: one group
per UTM zone number named `utm{NN}` (01-60). Each zone is stored in its northern CRS
(EPSG:326NN) with a continuous northing axis that goes negative south of the equator,
so a single group covers both hemispheres (matching the reference GeoTessera
implementation of the convention). Each zone group holds:

- an `embeddings` array with dimensions `(time, band, y, x)`: `band` is the 128-dim
  embedding vector, `time` is the annual reference years. It is int8, sharded so that
  one shard equals one 2048x2048 prediction window (with 256x256 inner chunks),
  zstd-compressed, with fill/nodata value -128.
- `time`, `x`, and `y` coordinate arrays.
- `proj:` and `spatial:` attributes (CRS and affine transform) and the geoemb
  provenance attributes (model, source data, quantization, etc.).

Because the array is sharded and sparse, only shards that intersect land are written;
ocean and unprocessed regions read back as the -128 nodata value.


How It Works
------------

Each UTM zone number (1-60) is processed once in its northern CRS. The zone is divided
into 32768x32768-pixel tiles, and each tile is one unit of work (one queue job). The
prediction pipeline for a tile creates 2048x2048-pixel windows in a scratch rslearn
dataset, materializes the input mosaics, runs the model, and writes each window's int8
embeddings (at 1/patch_size of the 10 m/pixel input resolution) into the store's zone
array at the window's `(time, y, x)` region. Windows are aligned to the store's shard
grid, so each window writes exactly one shard and concurrent workers never touch the
same shard.

To limit duplicated work where UTM zones overlap, tiles and windows are skipped unless
they intersect their zone's canonical 6-degree longitude wedge, which spans the full
UTM latitude range (see `tiling.py`). Windows that are entirely ocean (per
`global_land_mask`) or too close to 0/180 longitude (where mosaics are unreliable) are
also skipped.

When a tile finishes, a marker file `{crs}_{x}_{y}.json` is written to
`completed_path` recording the tile's projection, bounds, time range, time index, and
which windows were written and which were skipped (`written`, `skipped_no_data` for
windows without Sentinel-2 coverage, `skipped_longitude`, and `num_filtered_crops`
for wedge/ocean-filtered windows). Tiles with existing markers are excluded when
writing jobs and skipped by the prediction pipeline, so the pipeline is idempotent and
jobs can safely be re-enqueued to retry failures.

The store must be created once with `init_store` before any prediction jobs run.
`init_store` writes all group metadata (root, zone groups, arrays, coordinates), so
prediction workers only ever write data regions and never mutate metadata, which
keeps concurrent writes safe.


Running One Tile Locally
------------------------

This requires a GPU and access to the OlmoEarth checkpoint (e.g. run on a machine with
WEKA mounted). From the rslearn_projects root, first create the store, then run a
tile:

    python -m rslp.main large_scale_embeddings init_store \
        --store_path gs://BUCKET/embeddings/s2.zarr \
        --years '[2024]' \
        --model_url https://huggingface.co/allenai/OlmoEarth-v1_2-Small \
        --source_data '["https://sentinel.esa.int/web/sentinel/missions/sentinel-2"]' \
        --zone_numbers '[10]'

    python -m rslp.main large_scale_embeddings predict \
        --inputs S2 \
        --projection_json '{"crs": "EPSG:32610", "x_resolution": 10, "y_resolution": -10}' \
        --bounds '[32768, -557056, 65536, -524288]' \
        --time_range '["2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"]' \
        --store_path gs://BUCKET/embeddings/s2.zarr \
        --completed_path gs://BUCKET/embeddings/s2_completed/ \
        --checkpoint_path /weka/dfive-default/helios/checkpoints/gabrielt/regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1/step560000 \
        --time_index 0

The `projection_json` must be the zone's northern CRS (EPSG:326NN). `bounds` can be
any box whose extents are multiples of 2048 (it does not have to be a 32768x32768
tile). `time_range` is `(T, T)` where T is the reference timestamp; the dataset config
derives the twelve monthly mosaics over the year following T. `time_index` is the
index of this year in the store's time axis (0 for the first year in `--years`). By
default the scratch rslearn dataset is placed in a temporary directory and deleted;
pass `--scratch_path /path/to/scratch/` to keep it for debugging, and
`--debug_geotiff_path /path/` to also write per-window GeoTIFFs for inspection.


Running at Scale
----------------

Jobs are distributed via a Beaker queue and processed by `rslp.common` workers.

1. Build and push a Beaker image containing rslearn_projects (with the
   `global-land-mask`, `zarr`, and `gcsfs` dependencies included).

   Pin images by **Beaker image ID**, not by name. Images are immutable once
   committed, but a name/tag can be reused or deleted, so a name does not identify
   what actually ran. Record the ID and the commit it was built from together.

   Two roles need different things from the image:

   - **Workers** only execute `predict`. An older image keeps working for them as
     long as the job arguments and the store layout have not changed, so there is no
     need to rebuild workers for a supervisor-only change.
   - **The supervisor** needs an image that contains `supervise`, including the
     child-process cycle isolation (without it a hung Beaker RPC can stall the run
     for hours). Verify a supervisor image on a short run before relying on it.

   Validate a new image end-to-end before a long run: S2 -> forward pass -> int8
   GeoZarr write, then check that the dequantized per-pixel L2 norm is ~= 1.0. That
   catches a config-incompatible checkpoint and a broken write path in one pass.

2. Create the store once, covering all reference years and zones:

        python -m rslp.main large_scale_embeddings init_store \
            --store_path gs://BUCKET/PREFIX/s2.zarr \
            --years '[2021, 2022, 2023, 2024, 2025]' \
            --model_url https://huggingface.co/allenai/OlmoEarth-v1_2-Small \
            --source_data '["https://sentinel.esa.int/web/sentinel/missions/sentinel-2"]'

3. Write jobs to a Beaker queue for one reference year, one job per uncompleted tile
   (the year's time index is derived from the store's time axis):

        python -m rslp.main large_scale_embeddings write_jobs \
            --inputs S2 \
            --timestamp '2025-01-01T00:00:00+00:00' \
            --store_path gs://BUCKET/PREFIX/s2.zarr \
            --completed_path gs://BUCKET/PREFIX/s2_2025_completed/ \
            --checkpoint_path /weka/dfive-default/helios/checkpoints/gabrielt/regbtl_v1_2_gdyn_d128_wideread_regsup_latlon_w0p1/step560000 \
            --queue_name USER/QUEUE

   The model settings (`--checkpoint_path`, `--patch_size`, `--window_size`,
   `--overlap_size`, `--compile_model`; see above) are recorded in each job.
   Without additional arguments this enumerates all land tiles globally (~8,700).
   Options to limit the extent:

   - `--epsg_code 32610`: only the zone of this UTM EPSG code (326NN or 327NN both
     map to zone NN).
   - `--wgs84_bounds '[-125.0, 45.0, -116.0, 49.0]'`: only tiles intersecting these
     WGS84 bounds.
   - `--geojson_fname data/large_scale_embeddings/initial_regions.geojson`: only
     tiles intersecting a feature in the given WGS84 GeoJSON file (the included
     `initial_regions.geojson` covers Washington, Montana, Ukraine, Thailand, and
     points in Greenland and coastal Antarctica; 88 tiles).
   - `--count 10`: randomly sample this many tiles.

4. Launch workers on Beaker (WEKA must be mounted for the checkpoint). The
   OlmoEarth Datasets data source needs `OEDATASETS_API_URL` (plain env var) and
   `DATASETS_API_TOKEN` (bearer token, read from the `LCC_DATASETS_API_TOKEN`
   Beaker secret which must exist in the `ai2/earth-systems` workspace):

        python -m rslp.main common launch \
            --image_name USER/IMAGE \
            --queue_name USER/QUEUE \
            --num_workers 4 \
            --gpus 1 \
            --priority urgent \
            --cluster '["ai2/jupiter","ai2/ceres"]' \
            --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}' \
            --extra_env_vars '{"OEDATASETS_API_URL": "https://datasets.olmoearth.allenai.org"}' \
            --extra_env_secrets '{"DATASETS_API_TOKEN": "LCC_DATASETS_API_TOKEN"}' \
            --shared_memory 256GiB

Progress can be monitored by counting marker files in `completed_path`. To retry
failed tiles, simply run `write_jobs` again: completed tiles are excluded.

Run `write_jobs` once per reference year (each with its own `completed_path`), all
targeting the same store. Use a different store per input variant and per set of model
settings (checkpoint and patch/window/overlap sizes), since those change the
embeddings.
