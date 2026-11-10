Large-Scale Embeddings
======================

This project computes OlmoEarth embeddings over large areas (up to global scale) and
writes them to a GeoZarr store. The embeddings are:

- 10 m/pixel, in the UTM zone for each location.
- 128-dimensional and quantized to int8 (see Quantization below). The first 64
  dimensions are a trained Matryoshka prefix and can be used on their own.
- Computed from one year of Sentinel-2 L2A imagery starting at a reference timestamp,
  plus Sentinel-1 RTC and Landsat 8/9 where available.

The configuration files are in `data/large_scale_embeddings/`:
`s2_s1_landsat_distilled.json` is the rslearn dataset config (imagery comes from the
OlmoEarth Datasets sources) and `s2_s1_landsat_distilled.yaml` is the model config.
The model config always runs inference with Searchlight and `torch.compile`.

The model settings are arguments to `write_jobs`/`predict` and are recorded in each
queue job. All of them change the embeddings, so a store and its `completed_path` must
be computed with one fixed set:

- `--checkpoint_path` (required): an OlmoEarth training checkpoint, or a released
  model bundle (a directory with `config.json` and `weights.pth`).
- `--patch_size` (default `2`): the encoder's token patch size.
- `--location_query_patch_size` (default `1`): input pixels per output embedding.
- `--window_size` (default `480`): the size of the crops the model operates on. Lower
  it on GPUs with less memory (384 fits a 40 GB A100).
- `--overlap_size` (default `32`): overlap in pixels between adjacent crops.
- `--context_pixels` (default `48`): imagery margin around each 2048 px window that
  the model sees but that is not written, so pixels at window edges get context.


How It Works
------------

Each UTM zone is processed in its northern CRS (EPSG:326NN), with northing going
negative south of the equator, so one grid covers both hemispheres. The zone is divided
into blocks of `--job_size` pixels (default 8192), and each block is one queue job.

The prediction pipeline for a block creates 2048x2048-pixel windows in a scratch
rslearn dataset, materializes the input mosaics, runs the model, and writes each
window into the store as exactly one shard. Windows are skipped unless they intersect
their zone's canonical 6-degree wedge (see `tiling.py`) and the coverage mask in
`coverage_mask.tif` (land plus coastal water). Windows touching the antimeridian are
also skipped, since mosaics there are unreliable.

The store follows the
[geoemb embeddings-zarr-convention](https://github.com/geo-embeddings/embeddings-zarr-convention)
with one `utm{NN}` group per zone, each holding an `embeddings` array with dimensions
`(time, band, y, x)`. Shards are 2048x2048 pixels by 128 bands, with 256x256x64 inner
chunks.

When a block finishes, a marker `{crs}_{x}_{y}.json` is written to `completed_path`,
recording which windows were written and which were skipped (`skipped_no_data` for
windows without Sentinel-2, `skipped_longitude`, and `num_filtered_crops` for
wedge or coverage filtered windows). A provenance file with the same name, listing
the source scenes behind each window's mosaics, is written to a sibling directory
(`completed_2025/` becomes `provenance_2025/`). Blocks with markers are skipped, so
jobs can safely be re-enqueued to retry failures.

A block whose model output contains a non-finite value fails rather than being
written, so that retrying it recovers the pixels.


Running One Block Locally
-------------------------

This requires a GPU and access to the checkpoint. First create the store (once):

    python -m rslp.main large_scale_embeddings init_store \
        --store_path gs://BUCKET/embeddings.zarr \
        --years '[2024, 2025]' \
        --model_url https://huggingface.co/allenai/MODEL \
        --license_url https://example.com/LICENSE.txt

Then, from the rslearn_projects root:

    python -m rslp.main large_scale_embeddings predict \
        --inputs S2_S1_LANDSAT_DISTILLED \
        --projection_json '{"crs": "EPSG:32610", "x_resolution": 10, "y_resolution": -10}' \
        --bounds '[32768, -557056, 40960, -548864]' \
        --time_range '["2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"]' \
        --store_path gs://BUCKET/embeddings.zarr \
        --completed_path gs://BUCKET/completed_2024/ \
        --time_index 0 \
        --checkpoint_path /path/to/checkpoint

`bounds` can be any box whose extents are multiples of 2048. `time_range` is `(T, T)`
where T is the reference timestamp; the dataset config derives the monthly mosaics
over the year following T. `time_index` is T's year's position in the store's time
axis. Pass `--scratch_path` to keep the scratch dataset for debugging.


Running at Scale
----------------

Jobs are distributed via a Beaker queue and processed by `rslp.common` workers.

1. Build and push an image with `olmoearth_pretrain.Dockerfile`. On H100s, pass
   `--build-arg NATTEN_VERSION=...` matching the installed torch, and pass
   `--build-arg CODE_VERSION=...` to record the commits in every marker.

2. Create the store with `init_store` (see above), listing every year it will hold.

3. Write jobs to a Beaker queue, one per uncompleted block:

        python -m rslp.main large_scale_embeddings write_jobs \
            --inputs S2_S1_LANDSAT_DISTILLED \
            --timestamp '2025-01-01T00:00:00+00:00' \
            --store_path gs://BUCKET/embeddings.zarr \
            --completed_path gs://BUCKET/completed_2025/ \
            --checkpoint_path /path/to/checkpoint \
            --queue_name USER/embeddings-queue

   Without further arguments this enumerates every covered block globally. To limit
   the extent:

   - `--epsg_code 32610`: only one UTM zone.
   - `--wgs84_bounds '[-125.0, 45.0, -116.0, 49.0]'`: only blocks intersecting these
     WGS84 bounds.
   - `--geojson_fname aoi.geojson`: only blocks intersecting a feature in the given
     WGS84 GeoJSON file.
   - `--blocks_fname blocks.json`: exactly the blocks named in a JSON list of
     `{crs}_{x}_{y}` ids.
   - `--count 10`: randomly sample this many blocks.

4. Launch GPU workers. The OlmoEarth Datasets data source needs `OEDATASETS_API_URL`
   and a `DATASETS_API_TOKEN` secret:

        python -m rslp.main common launch \
            --image_name USER/IMAGE \
            --queue_name USER/embeddings-queue \
            --num_workers 4 \
            --gpus 1 \
            --cluster '["ai2/jupiter"]' \
            --weka_mounts+='{"bucket_name": "dfive-default", "mount_path": "/weka/dfive-default"}' \
            --extra_env_vars '{"OEDATASETS_API_URL": "https://datasets.olmoearth.allenai.org"}' \
            --extra_env_secrets '{"DATASETS_API_TOKEN": "LCC_DATASETS_API_TOKEN"}' \
            --shared_memory 256GiB

Each worker materializes its next block in a subprocess while the GPU runs the current
one. This needs the queue created with `max_claimed_entries=2`; with 1, blocks run
serially.

Progress can be monitored by counting markers in `completed_path`. To retry failed
blocks, run `write_jobs` again: completed blocks are excluded.


Quantization
------------

Embeddings are quantized following the AlphaEarth scheme (see `model.py`):
`quantized = round(sign(x) * |x|^0.5 * 127.5)` clipped to [-127, 127], with -128
reserved for nodata. They are not L2-normalized. To recover approximate float
embeddings:

```python
import numpy as np

def dequantize(v: np.ndarray) -> np.ndarray:
    x = v.astype(np.float32) / 127.5
    return np.sign(x) * np.abs(x) ** 2.0
```

Pixels where all Sentinel-2 mosaics are empty are set to -128 in all bands.
