Large-Scale Embeddings
======================

This project computes OlmoEarth embeddings over large areas (up to global scale) and
writes them to a GeoZarr store following the geoemb embeddings-zarr-convention
(https://github.com/geo-embeddings/embeddings-zarr-convention). The embeddings are:

- 10 m/pixel (at the default `--patch_size 1`; in general patch_size x 10 m/pixel),
  in the appropriate UTM projection for each location.
- 128-dimensional and quantized to int8 (see Quantization below).
- Computed from one year of input imagery starting at a user-provided reference
  timestamp; multiple reference years form the store's time axis.

There is one input variant (`EmbeddingInputs`), `S2_S1_LANDSAT_DISTILLED`: twelve
monthly Sentinel-2 L2A mosaics, twelve monthly Sentinel-1 RTC mosaics (converted from
linear intensities to dB), and twelve monthly Landsat 8/9 Collection 2 Level-1 mosaics,
using the 11 bands the encoder's `landsat` modality defines (`B8` at 15 m, then
`B1`-`B7`, `B9`-`B11` at 30 m, all resampled onto the window grid), through the model's
128-dim student head. S1 and Landsat are best-effort.

Landsat is sourced from a **requester-pays** GCS bucket, so the reading project is
billed; see the operational envelope below. A different variant would produce different
embeddings and would belong in a different store.

Where a best-effort modality is unavailable the embeddings are computed from what is
present. Sentinel-2 coverage is required.

The variant has an rslearn dataset config and a model config in
`data/large_scale_embeddings/`, named `{variant}.json` and `{variant}.yaml`. Imagery
comes from the OlmoEarth Datasets sources.


Flow
----

All steps are idempotent and driven by completion markers, so any of them can be
interrupted and resumed. `run_all` drives them in order.

1. **The basis**, before the run: olmoearth_run's `embedding_pca.pkl` for this
   foundation model, fitted with its `fit-embedding-pca` (see olmoearth_run's
   `internal-docs/runbooks/embedding_pca_artifact.md`). Point `pca.artifact_path` at
   it. It is read directly and applied to int8 values, as olmoearth_run applies it, so
   both products render identical colors. It must come from the same model: the
   supervisor refuses one fitted on a different embedding width. The basis defines what
   every rendered pixel means, so it cannot change once any are written.
2. **`predict`** writes the int8 embeddings. Needs GPUs. Given the pca paths, it also
   renders each block's multiscale `pca_rgb` pyramid into the sibling pca store
   (created once with `init_pca_store`) from the embeddings it already holds in
   memory, and writes the render marker. Enqueue with `write_jobs`, or let
   `supervise --stage predict` keep the queue and worker pool topped up.
3. **`render_pca`** is now a sweep: it reads embeddings back and renders only blocks
   that have a predict marker but no render marker, such as ones predicted without the
   pca paths. CPU only. Follow it with `annotate_pca_store` to record the basis
   provenance onto every level.
4. **`render_web_pca`** builds the web-mercator display pyramid, one zoom at a time,
   deepest first, after everything else.

**Prefetch.** A predict worker materializes its next block in a subprocess while the
GPU runs the current one. It needs the queue created with `max_claimed_entries=2`, so
Beaker hands over the next entry early; with 1 the worker runs serially as before.
Predict entries carry a `prefetch` field naming the args for that step (see
`rslp.common.worker._Prefetcher`). A claim is then held for about two inference
phases, which is why `claim_stale_seconds` defaults to 180 minutes.

Three components capture roughly 21-40% of local variance, so `pca_rgb` is a
visualization of the embeddings, not a reduced-dimension version of them.


Store Layout on Disk
--------------------

Two sibling stores per run, under a prefix that records the checkpoint and the model
settings:

    gs://BUCKET/{prefix}/
      {checkpoint}/
        {variant}_ps1_ws16_overlap4/
          embeddings.zarr        int8 embeddings, one array per UTM zone
          pca_v1.zarr            uint8 false-color pyramid, same zone layout
          completed_{year}/      predict markers
          pca_completed_{year}/  render_pca markers

`embeddings.zarr` is named for its contents rather than its inputs, since the input
variant already appears in the path above it and does not need restating.

There are two rules for `{prefix}`, because there are two kinds of store.

A **scoped run** -- one region, built once, kept for comparison -- uses
`geozarr_{aoi}_{years}_{date}/{checkpoint}/{variant}/`, e.g.
`geozarr_kenya_2022_2024_20260826/`. Every part of the name is settled the moment it is
created, so nothing in it can drift.

A **long-lived archive** replaces all of that with two independent versions:

    geozarr_global_v{model}/          the encoder release, e.g. v1.3
      v{store}/                       the archive format, e.g. v1
        README.md                     provenance the store cannot carry itself
        {variant}_ps1_ws16_overlap4/
          embeddings.zarr
          completed_{year}/

Neither a date nor a year range works for an archive that is built over months: the
store accumulates regions, and the time axis can be extended (resize the array, extend
the `time` coordinate, then re-consolidate), which would leave a year range in the path
contradicting the data. The years stay discoverable where they are authoritative, in the
`time` coordinate and the `geoemb:` metadata.

The two versions move for different reasons and neither implies the other:

- `v{model}` moves when the encoder does. A different checkpoint produces different
  embeddings, so it gets its own prefix rather than mixing into this one. There is no
  checkpoint path segment here, which means the checkpoint is recorded *only* in
  `geoemb:model`, `geoemb:build_version` and that `README.md` -- write them all.
- `v{store}` moves when a reader that can open the store would stop being able to:
  chunk geometry, dimensionality, quantization. Adding regions or years does not move it.

Nothing parses these paths; the convention is for people reading the bucket.

The PCA output is a **separate store**, for two reasons. Refitting the basis
invalidates every rendered pixel while leaving the embeddings valid, so putting the
basis version in the store name (`pca_v1`) lets a re-render land beside the old one and
cut over atomically. And the two want different storage classes: the embeddings are
cold, while the RGB layer is read often.

The pca store holds a multiscale pyramid, `pca_rgb` at level 0 plus `pca_rgb_2`,
`pca_rgb_4` and so on, listed in the `geoemb:multiscales` attribute. That is what makes
the store directly servable from a public bucket with no tile server: a client picks a
level by zoom and reads roughly a constant number of chunks at any extent, where the
level-0 array alone would need orders of magnitude more as the extent widens. The
pyramid costs 33% more bytes.

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
UTM latitude range (see `tiling.py`). Windows outside the coverage mask (see
`coverage.py`) or touching the antimeridian (where mosaics are unreliable) are also
skipped.

When a tile finishes, a marker file `{crs}_{x}_{y}.json` is written to
`completed_path` recording the tile's projection, bounds, time range, time index, and
which windows were written and which were skipped (`written`, `skipped_no_data` for
windows without Sentinel-2 coverage, `skipped_longitude`, and `num_filtered_crops`
for wedge/ocean-filtered windows), plus `gpu_seconds`, the `window_size` the tile was
computed with and the `worker` that wrote it. Markers without `window_size` predate it
and were computed at 480. Tiles with existing markers are excluded when
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
        --model_url /weka/path/to/checkpoint \
        --matryoshka_dims '[128, 64]' \
        --inputs S2_S1_LANDSAT_DISTILLED \
        --zone_numbers '[10]'

    python -m rslp.main large_scale_embeddings predict \
        --inputs S2_S1_LANDSAT_DISTILLED \
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
pass `--scratch_path /path/to/scratch/` to keep it for debugging.


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
   GeoZarr write, then check the head's `quantize diagnostics` log line reports a
   clipped fraction near zero. That catches a config-incompatible checkpoint, a broken
   write path, and a checkpoint whose head geometry no longer suits the quantizer.

   This run's image is built from local checkouts with `Dockerfile.vendored` rather
   than from the default `Dockerfile`. Since olmoearth_pretrain #623 and #624 merged
   on 2026-10-07 none of the three needs a patch (their default branches are master,
   main and develop respectively), but the default image does not
   install olmoearth_run and tracks branches rather than pinning commits, which a
   month-long run wants. See that file's header for the directories to populate.
   Nothing in the image records which commits went in, so confirm each checkout is
   where you want it before building; `geoemb:build_version` on a written store
   records them after the fact.

   Pin dependencies that read the store. gcsfs 2026.8.0 returns wrong bytes for
   ranged reads, which surfaces as a Zarr shard-index checksum mismatch and looks
   exactly like a corrupt store; the data and its checksums are fine. `requirements.txt`
   pins below it. When a read fails a checksum, verify the stored value independently
   (the shard index is the last `16 * inner_chunks + 4` bytes of the object, crc32c
   little-endian) before suspecting the writer.

2. Create the store once, covering all reference years and zones:

        python -m rslp.main large_scale_embeddings init_store \
            --store_path gs://BUCKET/PREFIX/embeddings.zarr \
            --years '[2021, 2022, 2023, 2024, 2025]' \
            --model_url /weka/path/to/checkpoint \
            --matryoshka_dims '[128, 64]' \
            --inputs S2_S1_LANDSAT_DISTILLED

   `--model_url`, `--source_data`, `--matryoshka_dims` and `--build_version` describe
   the encoder in the store's `geoemb:` metadata. They default to the current release
   and to the source datasets `--inputs` implies, so pass them only to record something
   other than that.

3. Write jobs to a Beaker queue for one reference year, one job per uncompleted tile
   (the year's time index is derived from the store's time axis):

        python -m rslp.main large_scale_embeddings write_jobs \
            --inputs S2_S1_LANDSAT_DISTILLED \
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
   - `--geojson_fname path/to/footprint.geojson`: only tiles intersecting a feature in
     the given WGS84 GeoJSON file. Note this intersects with the coverage mask rather
     than replacing it, so it can only narrow a run, never extend it.
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
            --inputs S2_S1_LANDSAT_DISTILLED \
            --years '[2024, 2025]' \
            --store_path gs://BUCKET/PREFIX/embeddings.zarr \
            --completed_path_template 'gs://BUCKET/PREFIX/s2_{year}_completed/' \
            --queue_name USER/QUEUE \
            --model.checkpoint_path /weka/dfive-default/helios/checkpoints/... \
            --worker.image_name USER/IMAGE \
            --worker.cluster '["ai2/jupiter","ai2/ceres"]' \
            --worker.num_workers 8 \
            --aoi.wgs84_bounds '[-125.0, 45.0, -116.0, 49.0]' \
            --aoi.job_size 4096

Its options are grouped into config objects, so they are namespaced on the command
line: `--model.*` (checkpoint and patch/window/overlap/compile/batch settings),
`--worker.*` (image, cluster, count, priority, credentials), `--cycle.*` (loop pacing),
`--aoi.*` (the ground to cover) and `--pca.*` (paths for the render stages). The five
required values are `--inputs`, `--years`, `--store_path`,
`--completed_path_template` and `--queue_name`, plus `--model.checkpoint_path`,
`--worker.image_name` and `--worker.cluster`.

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

Smaller jobs survive better but pay model load and compile per job, so total GPU time
rises. `job_size` is the unit of work a preemption destroys: the completion marker is
written once, after every window in the block, so a job killed near the end redoes all
of it. Note also that `urgent` is preempted by other `urgent` work, so priority reduces
the rate rather than removing it.

The default is 8192, because materialize cost is mostly per block rather than per
window. Measured on one AOI with four workers each, 2026-10-05:

    job_size   windows   materialize   whole block   per 4096 of area
        4096         4      13.4 min      32.3 min        32.3 min
        8192        16      16.4 min     124.3 min        31.1 min

Four times the ground for a fifth more materialize, so 3.3x the materialize throughput
per unit area. Whole-block time barely moved, because with prefetching a block costs
`max(materialize, inference)` and inference was the longer half. The size only pays
once a faster model inverts that: at a 5x inference speedup the same numbers give 16.1
min per 4096 of area against 6.7, a 2.4x difference. Pick the smaller size for a run
that is inference-bound, or one on a pool preempted often enough that losing four times
the work per preemption outweighs it.

`job_size` does not affect the store's layout: shard and chunk sizes are fixed at
`init_store`. It is purely a scheduling knob, so it can differ between runs against one
store. It cannot change *within* a run, though, because a marker is keyed on its
block's bounds and a different size will not match the markers already written.

**Preemption is normal, not an error.** Exit 143 with `canceled_for` naming another job
means preempted; retry is the correct response.

**Ask for a min_runtime, or the job is unallocated.** On the clusters using the newer
scheduler (jupiter, ceres, titan), a job counts as *allocated* only if its `min_runtime`
exceeds five minutes; anything at or below that is unallocated and runs only when no
allocated job wants the slot. `canceled_for` says so directly: "allocated workloads are
scheduled ahead of unallocated ones". Priority (urgent/high/normal/low) orders jobs
*within* a workspace and does not decide between budgets, so it cannot rescue an
unallocated job. Set `min_runtime` to roughly the time one job needs to make real
progress: a shorter request is placed sooner, and the maximum is eight hours. Pair it
with `auto_resume` so a preempted job is replaced. The older `preemptible` flag is
deprecated and maps to `min_runtime=0`, i.e. unallocated.

Scheduling between budgets is driven by usage against allocation over a 5-7 day
lookback, not by priority, so bursting above the allocation lowers priority later.

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

Embeddings are quantized following the AlphaEarth signed-power scheme (see
`model.py`): `quantized = round(sign(x) * |x|^0.5 * 127.5)` clipped to [-127, 127],
with -128 reserved for nodata. This is recorded in the store's `geoemb:quantization`
metadata with `method: "signed_power"`.

The model output is quantized as emitted, with no L2 normalization: normalizing
discards vector magnitude, which the evals score on, and it is not needed to satisfy
the scheme's [-1, 1] assumption. The head ends in a LayerNorm whose learned gain leaves
coordinates at std ~0.23 against a clip threshold of 0.992 (`QUANTIZE_CLIP_THRESHOLD`),
measured over Australian windows: clipping under 1e-4, round-trip cosine 0.99996. The
head logs those figures every 200 batches, so a checkpoint whose geometry differs shows
up rather than silently clipping.

Normalizing would also cost precision, not just magnitude. Unit-norm coordinates
quantize to a typical code of +/-38 out of 127, against +/-62 as emitted, so the
unnormalized path uses about 0.7 more of the 8 bits.

To recover approximate float embeddings:

```python
import numpy as np


def dequantize(v: np.ndarray) -> np.ndarray:
    x = v.astype(np.float32) / 127.5
    return np.sign(x) * np.abs(x) ** 2.0
```

Pixels where all Sentinel-2 mosaics are empty are set to -128 in all bands.


Coverage area
-------------

`data/large_scale_embeddings/coverage_mask.tif` defines the ground a run covers: a
global 1/120 degree (about 926 m) raster, one bit per cell, baked into the image so
workers share one read-only lookup. `coverage.py` loads and caches it.

It is the union of three things, and it is a strict superset of the land mask it
replaced, so blocks already computed stay valid:

- the previous `global_land_mask` land, kept wholesale as the floor;
- GSHHG full-resolution shorelines (L1 land, L5 Antarctic ice front), rasterised with
  ALL_TOUCHED, which is what recovers barrier islands, keys and atolls that the old
  mask reported as ocean;
- water with all three sensors present in at least 11 months of at least 7 of the years
  2017 to 2025, measured against the datasets service rather than assumed.

To limit a run to part of it, pass `--aoi.wgs84_bounds` or `--aoi.epsg_code`;
`--aoi.geojson_fname` still works if you have a footprint of your own.

To do part of it first rather than only, pass `--aoi.priority` a list of tiers in
priority order. Each tier takes a `geojson_fname`, a list of `years`, or both. A job is
enqueued in the first tier matching it on both counts, then everything else, shuffled
within each tier:

    --aoi.priority '[
      {"geojson_fname": "conus.geojson"},
      {"geojson_fname": "kenya.geojson", "years": [2021, 2022, 2023, 2024, 2025]},
      {"years": [2025]}
    ]'

That runs every year of CONUS, then Kenya's five most recent, then 2025 everywhere
else, then the rest. Omitting `geojson_fname` matches anywhere and omitting `years`
matches every year, so a tier can select on either axis alone; a tier setting neither
is rejected, since it would match everything and strand the tiers after it.

No second queue is involved: the supervisor keeps the queue shallow, a few hundred
entries against tens of thousands outstanding, so enqueue order is already what decides
what gets worked on next. Matching is on the block's centre, which is right for
ordering but is not AOI clipping: a block straddling a footprint's edge is in or out by
its centre alone.

`coverage_world.png` renders it in Equal Earth: magenta is covered, and the
land showing through uncovered is water the mask correctly excludes, or land dropped
on purpose. Antarctica is the second kind: it reaches to about 60S, well inside the UTM
grid, and was removed from the mask rather than being unreachable. The hatched bands
above 84N and below 80S are the unreachable part, where UTM defines no zone.

That PNG is a committed artifact and does not rebuild itself, so regenerate it whenever
the mask changes: `python -m rslp.large_scale_embeddings.tools.render_coverage_world`.
It went six days showing a mask that still included Antarctica.

Sizing a new area before committing to it is one call, and worth making:

    python -c "
    from datetime import UTC, datetime
    from rslp.large_scale_embeddings.predict_pipeline import EmbeddingInputs
    from rslp.large_scale_embeddings.write_jobs import get_jobs
    print(len(get_jobs(inputs=EmbeddingInputs.S2_S1_LANDSAT_DISTILLED,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC), store_path='/tmp/x.zarr',
        completed_path='/tmp/c/', checkpoint_path='/weka/x', time_index=0,
        patch_size=1, window_size=16, overlap_size=4, compile_model=True,
        batch_size=None, epsg_code=None,
        wgs84_bounds=(-5.0, 42.0, 8.0, 51.0), job_size=8192)))"

Point `store_path` and `completed_path` at local paths, not a bucket: enumeration only
needs them to check for markers, and an unauthenticated bucket read fails on the
completion check before it reports a count.

Chunk shape
-----------

The store fixes three geometry parameters, and all three are measured. See
`CHUNKING.md` for the full table and the reasoning; the short version is
`DEFAULT_CHUNK_SIZE = 256`, `DEFAULT_BAND_CHUNK = 64` and `DEFAULT_ZSTD_LEVEL = 3`. 256
matches the earlier embeddings already computed at that size, and its read cost does not
depend on the reader's coalescing setting. 64 was tried on 2026-10-06 for 11x cheaper
point reads and reverted on 2026-10-07.

`DEFAULT_SHARD_SIZE = 2048` is not a tuning parameter at all. One prediction window
writes exactly one object, which is what keeps concurrent writers on disjoint objects
and needs no locking. It moves only if the write path does.

The other two are chosen once and for good: zarr cannot re-chunk an array in place, so
changing them means rewriting every object. Re-run the benchmark before creating a store
if the embedding dimensionality changes, if the emitted Matryoshka widths move, or if
the dominant access pattern stops being the point and AOI reads measured there.

`tools/bench_chunking.py` is what produced them. Two commands:

    python -m rslp.main large_scale_embeddings bench_build_variants \
        --source_store_path gs://BUCKET/.../embeddings.zarr \
        --out_prefix gs://BUCKET/bench/chunking_v1 \
        --model_url https://huggingface.co/allenai/OlmoEarth-v1_3-Base \
        --source_data '["https://sentinel.esa.int/web/sentinel/missions/sentinel-2"]'

    python -m rslp.main large_scale_embeddings bench_measure \
        --out_prefix gs://BUCKET/bench/chunking_v1 \
        --results_path gs://BUCKET/bench/chunking_v1/results.json

Design, and why each choice is what it is:

- **No prediction re-run.** A 3x3 block of finished shards is read out of an existing
  store and rewritten into one variant store per chunk shape. Every variant then holds
  byte-identical embeddings, so any difference between them is layout and nothing else.
  The experiment costs a rewrite, not a run.
- **Area: 3x3 shards, 6,144 px, 61.44 km.** Three is the smallest meaningful number.
  The 20 km AOI pattern is exactly one shard wide, so a 2x2 block can only place it
  shard-aligned or corner-straddling; at 3x3 there is also a centre shard with written
  neighbours on all sides, which is the ordinary case globally. A single shard would
  report its own terrain rather than the layout.
- **Reads placed off-alignment on purpose.** Every pattern starts at an offset divisible
  by none of 128, 256, 512 or 1024. A benchmark that aligns its reads to chunk
  boundaries measures the best case for large chunks and describes no AOI anyone draws.
  A unit test asserts this, and it has already caught the AOI read sitting exactly on a
  shard boundary.
- **Five patterns**: a point at 128 dims and at 64, a 1 km area, a shard-straddling
  20 km Matryoshka AOI, and a 40 km transect.
- **The continental view is deliberately absent.** That read belongs to the PCA pyramid
  in `pca_v1.zarr`, whose levels exist so a wide extent touches a bounded number of
  chunks. Benchmarking it against `embeddings.zarr` would argue for a chunk shape
  nothing needs.
- **Compression held at `DEFAULT_ZSTD_LEVEL`**, which was settled offline by
  recompressing real chunks (the table in `zarr_store.py`). One control variant carries
  the old level 1 so the in-situ result can be checked against the offline one.

Each measurement reports bytes moved, requests made, distinct objects touched, wall
clock, and read amplification (bytes moved over bytes wanted). The store is reopened for
every repeat, because zarr caches a shard index per array handle and reusing one would
hide a cost every cold client pays.

A reference row, measured against the live Kenya store, which is `sp256/d32/z1`:

| pattern | moved | requests | objects | amplification |
| --- | --- | --- | --- | --- |
| point, 128 dims | 6.15 MB | 5 | 1 | 48,053x |
| point, 64 dims | 3.09 MB | 3 | 1 | 48,357x |
| 1 km area, 128 dims | 6.15 MB | 5 | 1 | 5x |
| 40 km transect, 64 dims | 51.02 MB | 37 | 3 | 12x |

Cost of the sweep: 17 variants at 4.8 GB of array each, so about 55 GB written and
roughly an hour of one core per variant in compression. It parallelises one process per
variant. Set `--only sp128_d64_z3.zarr` to build a single one.

Only one reference year is copied (`--time_index`, default 0). T is chunked at 1, so
every year is an independent shard and no pattern here crosses the time axis; copying
all three years of the Kenya store would raise that 55 GB to 246 GB and triple the
compression time for no extra signal.

Workers on GCE
--------------

`deploy/gce_worker_startup.sh` boots a GCP GPU VM as a worker on the same Beaker
queue. The queue is reachable from anywhere with a Beaker token, so the supervisor
needs no change and does not know these workers exist; it keeps managing the Beaker
pool while they consume the same entries.

Use it when the Beaker clusters cannot place what the allocation allows. Standard
provisioning returned STOCKOUT in three of four us-central1 zones for a single A100,
while a Dynamic Workload Scheduler Flex Start request for 100 filled in four minutes.

    gcloud compute instances create embed-worker-1 \
      --machine-type=a2-highgpu-1g --zone=us-central1-f \
      --image-family=common-cu129-ubuntu-2204-nvidia-580 \
      --image-project=deeplearning-platform-release \
      --boot-disk-size=300GB --maintenance-policy=TERMINATE \
      --scopes=https://www.googleapis.com/auth/cloud-platform \
      --labels=role=embedding-worker \
      --metadata-from-file=startup-script=rslp/large_scale_embeddings/deploy/gce_worker_startup.sh \
      --metadata=install-nvidia-driver=True,embed-image-tag=rc-20260930d

Every path, project, secret name and image tag is an instance metadata attribute with
a default; run `grep 'attr '` on the script for the list. The project defaults to the
VM's own, so the script is not tied to one. The `role=embedding-worker` label is what
the coverage slide counts to report GCP workers separately from Beaker ones.

Batch size follows the worker, not the job. `--worker.batch_size` is required and the
supervisor passes it to each worker in `RSLP_WORKER_EXTRA_ARGS`, which the worker
appends to every entry it runs; queue entries carry no `--batch_size` at all. That is
what lets one queue feed an H100 pool and an A100 pool at once, since batching only
groups independent crops and changes footprint and speed, never the embeddings. On GCE
the same override is `embed-worker-extra-args`, and anything passed through
`--worker.env_vars` wins over the supervisor's value because the last one parsed is
the one kept.

Four things Beaker's executor provides implicitly, which the script has to do itself.
Each was found by a run failing without it:

- A container runtime. The deep learning VM images carry the driver and
  `nvidia-container-runtime` but no engine, so docker is installed and the runtime
  registered with `nvidia-ctk`.
- The env set, above all `OEDATASETS_API_URL`. Without it the data source builds
  `/api/v1/items/search` with no scheme and retries forever.
- `--shm-size`. The 64 MB docker default kills DataLoader workers.
- `--ulimit nofile`. The 1024 default exhausts descriptors in the file-descriptor
  sharing strategy, and the loader dies with `EOFError` in `recvfds`.

Secrets are read at boot with the VM's own service account, which needs
`roles/secretmanager.secretAccessor` on each. Nothing lands on the image or in
metadata.

The checkpoint is copied to the path the queue entries name. The supervisor bakes
`--checkpoint_path` into every entry when it enqueues, so reproducing that path
locally is what lets a GCE worker consume an unmodified entry.

An A100 40GB holds the same batch size of 128 as an H100, peaking near 38 GB of 40.
One 4096 block measured 81.6 minutes end to end, 14 of them materialize, against a
51.4 minute whole-run average per block on H100s. That average includes startup and
contention the single measurement does not, so treat the ratio as an upper bound.
