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

There are three input variants (`EmbeddingInputs`), which produce different embeddings
and so must be written to different stores:

- `S2`: twelve monthly Sentinel-2 L2A mosaics.
- `S2_S1`: the above, plus twelve monthly Sentinel-1 RTC mosaics (converted from
  linear intensities to dB).
- `S2_LANDSAT`: the Sentinel-2 mosaics plus twelve monthly Landsat 8/9 Collection 2
  Level-1 mosaics, using the 11 bands the encoder's `landsat` modality defines
  (`B8` at 15 m, then `B1`-`B7`, `B9`-`B11` at 30 m, all resampled onto the window
  grid). Landsat is sourced from a **requester-pays** GCS bucket, so the reading
  project is billed; see the operational envelope below.

The secondary modality is best-effort in both mixed variants: where it is unavailable
the embeddings are computed from Sentinel-2 alone. Sentinel-2 coverage is required.

Each variant has an rslearn dataset config and a model config in
`data/large_scale_embeddings/`, named `{variant}.json` and `{variant}.yaml`. Imagery
comes from the OlmoEarth Datasets sources.


Three-Step Flow
---------------

A full run is three ordered steps. Each depends on the previous one's output, and all
three are idempotent and driven by completion markers, so any of them can be
interrupted and resumed.

1. **`predict`** writes the int8 embeddings. Needs GPUs. Enqueue with `write_jobs`, or
   let `supervise --stage predict` keep the queue and worker pool topped up.
2. **`fit_pca`** samples the archive just written and fits the global false-color
   basis, so the basis reflects exactly the data it will be applied to. Single process,
   reads about one inner chunk per sampled window rather than a pass over the archive.
   A basis fitted on one region does not transfer: measured on real blocks, a
   Washington-fitted basis captured 69.5% of Washington's variance but only 3.0% of
   Ukraine's, and per-region normalization bounds were nearly disjoint. Sampling is
   therefore stratified across every UTM zone with data.
3. **`render_pca`** reads the embeddings back and writes the multiscale `pca_rgb`
   pyramid into the sibling pca store, created once with `init_pca_store`. CPU only, no
   model, so it schedules without competing for GPU capacity. Enqueue with
   `write_render_jobs`, or use `supervise --stage render_pca --gpus 0`. Follow it with
   `annotate_pca_store` to record the basis provenance onto every level.

Three components capture roughly 21-40% of local variance, so `pca_rgb` is a
visualization of the embeddings, not a reduced-dimension version of them.


Store Layout on Disk
--------------------

Two sibling stores per run, under a prefix that records the checkpoint and the model
settings:

    gs://BUCKET/geozarr_{aoi}_{years}_{date}/
      {checkpoint}/
        {variant}_ps1_ws16_overlap4/
          embeddings.zarr        int8 embeddings, one array per UTM zone
          pca_v1.zarr            uint8 false-color pyramid, same zone layout
          completed_{year}/      step 1 markers
          pca_completed_{year}/  step 3 markers

`embeddings.zarr` is named for its contents rather than its inputs, since the input
variant already appears in the path above it and does not need restating.

The PCA output is a **separate store**, not another array inside the embeddings store,
for two reasons. Refitting the basis invalidates every rendered pixel while leaving the
embeddings valid, so the derived layer needs its own lifecycle; putting the basis version
in the store name (`pca_v1`) lets a re-render land beside the old one and cut over
atomically instead of leaving the layer half-rendered and unservable. And the two want
different storage classes: the embeddings are cold, while the RGB layer is read often and
Nearline charges per read.

The pca store holds a multiscale pyramid, `pca_rgb` at level 0 plus `pca_rgb_2`,
`pca_rgb_4` and so on, listed in the `geoemb:multiscales` attribute. That is what makes
the store directly servable from a public bucket with no tile server: a client picks a
level by zoom and reads roughly a constant number of chunks at any extent. Reading a
1600x900 view from the level-0 array alone would need 24 chunks at z14 but 1,296 at z11
and over a million at z6. The pyramid costs 33% more bytes.

Every level keeps one shard per source window footprint (2048 px at level 0, 1024 at
level 1, and so on), so a window stays a whole object owned by a single writer and
concurrent renders need no locking, exactly as for the embeddings.

The model settings are provided as arguments to `write_jobs`/`predict` and override
the defaults in the model configs (they are recorded in each queue job, so workers
can process jobs with differing settings):

- `--checkpoint_path` (required): the OlmoEarth checkpoint to compute embeddings
  with, e.g.
  `/weka/dfive-default/helios/checkpoints/gabrielt/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform/step667200`.
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
        --store_path gs://BUCKET/PREFIX/embeddings.zarr \
        --years '[2024]' \
        --model_url https://huggingface.co/allenai/OlmoEarth-v1_2-Small \
        --source_data '["https://sentinel.esa.int/web/sentinel/missions/sentinel-2"]' \
        --zone_numbers '[10]'

    python -m rslp.main large_scale_embeddings predict \
        --inputs S2 \
        --projection_json '{"crs": "EPSG:32610", "x_resolution": 10, "y_resolution": -10}' \
        --bounds '[32768, -557056, 65536, -524288]' \
        --time_range '["2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"]' \
        --store_path gs://BUCKET/PREFIX/embeddings.zarr \
        --completed_path gs://BUCKET/PREFIX/completed_2024/ \
        --checkpoint_path /weka/dfive-default/helios/checkpoints/gabrielt/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform/step667200 \
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

   **Checkpoint and olmoearth_pretrain must be paired.** A checkpoint's config.json
   serializes every encoder field that existed when it was trained, including defaults,
   and `Config.from_dict` rejects fields the current code has removed. Loading a
   checkpoint against too-new code fails with `Failed to construct 'encoder_config' in
   config`, which names neither the field nor the checkpoint. rslearn's
   `_patch_legacy_encoder_config` only adds a missing key and cannot bridge this.

   Known pairing: the distilled release candidate
   `regbtl_v1_2_gdyn_d768_proj128lin_sup768_w1_newsamp_psuniform/step667200` needs
   olmoearth_pretrain at or before `72ba0a8e` (2026-08-24). The next commit,
   `5c573d7a`, drops `register_read_layers` and `register_shared_read_kv`; later ones
   drop `register_output_dim`, `register_unit_norm` and `register_latent_every_n`, all
   of which that checkpoint's config still carries.

   Validate a new image end-to-end before a long run: S2 -> forward pass -> int8
   GeoZarr write, then check that the dequantized per-pixel L2 norm is ~= 1.0. That
   catches a config-incompatible checkpoint and a broken write path in one pass.

2. Create the store once, covering all reference years and zones:

        python -m rslp.main large_scale_embeddings init_store \
            --store_path gs://BUCKET/PREFIX/embeddings.zarr \
            --years '[2021, 2022, 2023, 2024, 2025]' \
            --model_url https://huggingface.co/allenai/OlmoEarth-v1_2-Small \
            --source_data '["https://sentinel.esa.int/web/sentinel/missions/sentinel-2"]'

3. Write jobs to a Beaker queue for one reference year, one job per uncompleted tile
   (the year's time index is derived from the store's time axis):

        python -m rslp.main large_scale_embeddings write_jobs \
            --inputs S2 \
            --timestamp '2025-01-01T00:00:00+00:00' \
            --store_path gs://BUCKET/PREFIX/embeddings.zarr \
            --completed_path gs://BUCKET/PREFIX/s2_2025_completed/ \
            --checkpoint_path /weka/dfive-default/helios/checkpoints/gabrielt/regbtl_v1_2_gdyn_d128_wideread_regsup_ndvi_w0p1_tanchor_newsamp_psuniform/step667200 \
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
   `DATASETS_API_TOKEN` (bearer token, read from the `OEDATASETS_API_TOKEN`
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
            --extra_env_secrets '{"DATASETS_API_TOKEN": "OEDATASETS_API_TOKEN"}' \
            --shared_memory 256GiB

Progress can be monitored by counting marker files in `completed_path`. To retry
failed tiles, simply run `write_jobs` again: completed tiles are excluded.

Run `write_jobs` once per reference year (each with its own `completed_path`), all
targeting the same store. Use a different store per input variant and per set of model
settings (checkpoint and patch/window/overlap sizes), since those change the
embeddings.


Supervised Runs (recommended)
-----------------------------

For anything longer than a few hours, use `supervise` instead of driving `write_jobs`
and `launch` by hand. It loops: recompute the remaining work from the completion
markers, top the queue up only if it is running shallow, and refill the worker pool.
It exits when every tile has a marker.

        python -m rslp.main large_scale_embeddings supervise \
            --inputs S2 \
            --years '[2024, 2025]' \
            --store_path gs://BUCKET/PREFIX/embeddings.zarr \
            --completed_path_template 'gs://BUCKET/PREFIX/s2_{year}_completed/' \
            --queue_name USER/QUEUE \
            --checkpoint_path /weka/dfive-default/helios/checkpoints/... \
            --image_name USER/IMAGE \
            --cluster '["ai2/jupiter","ai2/ceres"]' \
            --geojson_fname data/large_scale_embeddings/initial_regions.geojson \
            --job_size 8192 --num_workers 8

Run it as a cheap CPU Beaker job, not from a workstation: it must outlive any single
login session, and a laptop-side loop dies with the session (or silently hangs -- the
Beaker client has no RPC timeout, so an in-process watchdog cannot bound it).


Operational envelope
--------------------

Hard-won numbers from the 2024/2025 `initial_regions` run. Re-measure if the model,
image, or cluster changes, but start here.

**Size jobs to finish inside the preemption window.** Workers are preemptible and the
GPU clusters are routinely at zero free slots, so jobs are interrupted constantly. A
job that runs longer than the typical gap between preemptions never completes at all.
Measured throughput was ~2.4 min per window end-to-end (~0.6 min materialize, ~1.8 min
predict) at `patch_size=1`, `window_size=16` on an H100:

    job_size   windows   ~duration   outcome
       32768       256       ~9 h     never completed (always preempted first)
        8192        16      ~38 min   completes reliably
        4096         4      ~12 min   completes, but ~55% of the time is fixed overhead

`job_size=8192` was the sweet spot. Smaller jobs survive better but pay model load and
compile per job, so total GPU time rises.

**Preemption is normal, not an error.** Exit 143 with `canceled_for` naming another job
means preempted; retry is the correct response. Note also that `ai2/jupiter` uses
"strict priority with unallocated-only backfill", so without an allocation your jobs
are backfill and can be evicted at any priority.

**Keep the queue shallow.** A queue entry claimed by a worker that then dies is not
released back to the queue: entries were still CLAIMED 5 hours after being claimed, with
no worker alive for the last 1.4 of those, and the queue API has no call to release one.
They do eventually age out, since `status.expiry` is set from `expires_in_sec` (7 days
by default), but a week is far longer than any job, so within a run that work is lost.
`max_claimed_entries=1` makes it worse: a dead worker's claim permanently occupies that
entry's only claim slot. `wait_timeout` on the queue is unrelated to this; it bounds how
long a worker waits for work to appear. Untested: whether expiry deletes the entry or
returns it to PENDING. Enqueuing a
whole run up front therefore bleeds work steadily -- one run accumulated 327 orphaned
entries. `supervise` enqueues only a small buffer and refills from the markers, which
bounds the loss to about one entry per worker death.

**`MATERIALIZE_PIPELINE_ARGS` pool sizes are the working default; changing them is
untested.** Scaling them to the job's window count was tried and reverted, but the
revert was based on a mismeasured elapsed time, so it is neither proven harmful nor
proven safe. If you revisit it, note that materialize parallelizes over window x
item-group units (each window pulls 12 monthly mosaics, so a 12-window job is ~144
units, not 12), so sizing by window count alone under-parallelizes.

**Measure elapsed time carefully.** These logs are emitted in the machine's local
time, not UTC. Comparing a log timestamp against `date -u` silently adds the UTC
offset -- doing so produced a "7 hours with zero completions" reading of what was
actually 7 minutes, and a wrong conclusion about the pool sizes above. Prefer deltas
between two timestamps from the same log, and remember a single job takes ~38 minutes
at `job_size=8192`, so any window shorter than that tells you nothing.

**Worker deaths are common and not yet explained.** Roughly 68% of attempts on an
8-worker pool ended in SIGKILL (137) or SIGSEGV (139) rather than preemption (143).
Memory pressure from co-location is the leading theory -- a single-GPU worker can share
an 8-GPU node with seven siblings -- but it is unproven, and the materialize pool is
*not* the cause. The cheapest experiment is to request more GPUs per worker so fewer
land per node, which needs no code change. Vary one thing at a time and measure the
completion rate; the failure is frequent enough that a few hours gives a clear signal.

**Storage.** ~385 MB per written window on GCS (2048x2048x128 int8 at zstd level 1,
measured compression ratio 0.717 on real embeddings, range 0.534-0.794). Roughly
5.4 GB per `job_size=8192` block. Only shards intersecting land are written.

**A smaller `job_size` also tightens AOI clipping.** Blocks outside the GeoJSON
features are dropped, whereas a large tile merely intersecting a feature had all of its
land crops processed. `initial_regions` covers ~7,700 windows/year at `job_size=8192`
versus ~11,300 at 32768. That is usually desirable, but it means output extent is not
comparable across `job_size` values.


Quantization
------------

Embeddings are L2-normalized and then quantized following the AlphaEarth signed-power
scheme (see `model.py`): `quantized = round(sign(x) * |x|^0.5 * 127.5)` clipped to
[-127, 127], with -128 reserved for nodata. This is recorded in the store's
`geoemb:quantization` metadata with `method: "signed_power"`. To recover approximate
float embeddings:

```python
import numpy as np

def dequantize(v: np.ndarray) -> np.ndarray:
    x = v.astype(np.float32) / 127.5
    return np.sign(x) * np.abs(x) ** 2.0
```

Pixels where all Sentinel-2 mosaics are empty are set to -128 in all bands.
