## Change Finder V2: Land Cover Change

Point-based annotation and detection of land cover changes using Sentinel-2
time series. Unlike v1 (which used polygon annotations from a WorldCover
segmentation model), v2 uses manually placed point annotations and a
dual-forward-pass model architecture.

### Annotation Format

The core data structure is a JSON file (list of dicts). Each entry represents
one 128x128 spatial window:

```json
{
  "projection": {"crs": "EPSG:32651", "x_resolution": 10, "y_resolution": 10},
  "bounds": [0, 0, 128, 128],
  "window_name": "example_window",
  "group": "default",
  "time_range": ["2017-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"],
  "positive_points": [
    {
      "lon": 121.5, "lat": 14.6,
      "pre_change": "2020-01-15",
      "first_date_change_noticeable": "2020-04-27",
      "post_change": "2020-07-15",
      "pre_category": "tree",
      "post_category": "urban/built-up"
    }
  ],
  "negative_points": [
    {"lon": 121.51, "lat": 14.61}
  ]
}
```

- `positive_points`: locations where land cover change occurred.
- `negative_points`: locations confirmed as no-change.
- `time_range`: metadata indicating when negative points are valid (not used
  for imagery fetching).
- Dates may be blank/missing if not yet annotated.

Categories: nodata, bare, burnt, crops, fallow/shifting cultivation, grassland,
Lichen and moss, shrub, snow and ice, tree, urban/built-up, water,
wetland (herbaceous).

### Running the Annotation App

The annotation app is a Flask web UI for browsing entries and labeling points.

```bash
# 1. Create the rslearn dataset windows (for imagery)
python -m rslp.change_finder_v2.annotation_app.create_windows \
    annotations.json \
    /path/to/dataset/

# 2. Materialize Sentinel-2 imagery
rslearn dataset prepare     --root /path/to/dataset/ --workers 32
rslearn dataset materialize --root /path/to/dataset/ --workers 32

# 3. Launch the annotation app
python -m rslp.change_finder_v2.annotation_app.server \
    --json annotations.json \
    --ds-path /path/to/dataset/ \
    --port 8080
```

The UI:
- Displays up to 12 monthly Sentinel-2 images per year in a 2-row grid.
- Toggle overlay to see positive (green) and negative (red) points.
- Left-click on overlay to add a positive point; right-click to add negative.
- Click existing points to remove them.
- Iterate through positive points with the point navigator at the top.
- Edit annotation fields (pre_change, first_date_change_noticeable, post_change,
  pre_category, post_category) for the selected positive point.
- Click timestamps below images to copy them to clipboard.
- Navigation: Prev/Next buttons to move between entries. URL hash tracks position.

### Converting V1 Annotations

If you have v1 annotations (GeoJSON + sidecar), convert them:

```bash
python -m rslp.change_finder.land_cover_change_viewer.to_v2_json \
    --geojson /path/to/land_cover_change_src_dst_sel100.geojson \
    --annotations /path/to/land_cover_change_src_dst_sel100.annotations.json \
    --src-ds-path /path/to/ten_year_dataset/ \
    --output annotations.json
```

---

### Model

#### Architecture

The LCC (land cover change) model uses a dual-forward-pass architecture:

- **Encoder**: OlmoEarth-v1-Base (shared weights, patch_size=4). Processes two
  sets of 10 images separately.
- **Pass 1**: 10 quarterly images (historical baseline, ~2.5 years).
- **Pass 2**: 6 quarterly images + 4 frequent images (recent conditions).
- **Decoder**: Features from both passes concatenated (1536ch @ 1/4 res) →
  shared 1x1 conv (768ch) → per-task linear heads reshaping to full resolution
  via 4x4 patch prediction.

Tasks (all per-pixel, loss masked to annotated points only):
- `binary`: change classification (3 cls: nodata / no_change / change)
- `src`: source land cover category (13 cls)
- `dst`: destination land cover category (13 cls)
- `timestamps`: per-image change-period membership (20 sigmoid outputs, one per
  input image; target is 1 if image center time falls within [pre_change, post_change])

#### Prerequisites

The prepare and materialize steps use OlmoEarth Datasets for Sentinel-2 scene
discovery and download. Install `olmoearth_run` and set the required env vars:

```bash
pip install -e /path/to/olmoearth_run
export OEDATASETS_API_URL=https://datasets.olmoearth.allenai.org
export DATASETS_API_TOKEN=<your-token>
```

#### Training Dataset Preparation

The prepare script creates the training dataset from the v2 annotation JSON.
It queries the OlmoEarth Datasets API for both quarterly and frequent imagery,
then rslearn materialize downloads the pixels.

```bash
OUT=/path/to/lcc_model_dataset/
mkdir -p "$OUT"
cp data/change_finder_v2/lcc_model/config.json "$OUT/config.json"
python -m rslp.change_finder_v2.lcc_model.prepare annotations.json "$OUT"
rslearn dataset materialize --root "$OUT" --workers 128
```

The prepare script is **idempotent**: windows that already exist are skipped, so
you can re-run it after adding new annotations and only the new windows will be
created. Then run `rslearn dataset materialize` again to download imagery for
the new windows.

The prepare script:
- Only processes entries with fully annotated positive points (all of pre_change,
  post_change, first_date_change_noticeable, pre_category, post_category present).
- Skips windows that already exist in the dataset.
- Creates 4 frequent-image options per window with varying temporal endpoints
  (up to post_change + 2 years), so some training samples have the change
  further in the past relative to the most recent imagery.
- Window time_range is derived from the annotations (no fixed 10-year range).
- Rasterizes point labels into label_binary/label_src/label_dst layers.
- Writes `lcc_annotations.json` sidecar for training-time annotation injection
  (merges with existing sidecar on re-runs).

#### Training

```bash
rslearn model fit --config data/change_finder_v2/lcc_model/config.yaml
```

Training details:
- Encoder frozen for 10 epochs, then unfrozen with 10x lower LR (effective
  encoder LR = 1e-5 vs decoder 1e-4).
- Optimizer: AdamW lr=1e-4 with ReduceLROnPlateau (factor=0.2, patience=2).
- At each training step, one of 4 frequent-image options is randomly sampled.
  The 16 quarterly images preceding that option's endpoint are selected, then
  split 10/6 across the two encoder passes.
- Val/test uses option 0 deterministically.
- Augmentation: random horizontal/vertical flips.
- crop_size: 64, batch_size: 8, 100 epochs max.

---

### Prediction

For prediction, a separate dataset config handles imagery via rslearn's
standard prepare/materialize pipeline (using OlmoEarth Datasets). The user
creates windows with a single point-in-time timestamp — quarterly images are
fetched for the 4 years preceding it, and the 4 least-cloudy scenes from the
30 days following it are used as frequent images.

#### 1. Create windows

```bash
PREDICT_DS=/path/to/predict_dataset/
mkdir -p "$PREDICT_DS"
cp data/change_finder_v2/lcc_model/config_predict.json "$PREDICT_DS/config.json"

# Create windows over an AOI. The time range is a single point in time:
# the reference timestamp from which quarterly looks back and frequent looks forward.
# Example: detect changes visible as of 2025-06-01
rslearn dataset add_windows \
    --root "$PREDICT_DS" \
    --group predict \
    --fname aoi.geojson \
    --src_crs EPSG:4326 \
    --utm --resolution 10 --grid_size 2048 \
    --start 2025-06-01T00:00:00+00:00 \
    --end 2025-06-01T00:00:00+00:00
```

The `config_predict.json` layers use `time_offset` and `duration` so both layers
derive their search ranges from this single timestamp:
- `sentinel2_quarterly`: time_offset=-1440d, duration=1440d → searches
  [T-1440d, T] for 16 quarterly mosaics (4 years back).
- `sentinel2_frequent_0`: duration=30d → searches [T, T+30d] for the 4
  least-cloudy scenes in the month following T.

Alternatively create windows from bounding boxes. These examples produce exactly
32768x32768 tiles (256 windows of 2048x2048) by providing grid-aligned UTM
coordinates in meters directly via `--src_crs`:

```bash
# Seattle area (UTM 10N) - 327.68 km x 327.68 km tile
rslearn dataset add_windows --root "$PREDICT_DS" --group predict \
    --box=368640,5099520,696320,5427200 \
    --src_crs EPSG:32610 --crs EPSG:32610 --resolution 10 \
    --grid_size 2048 \
    --start 2025-01-01T00:00:00+00:00 --end 2025-01-01T00:00:00+00:00

# Doha area (UTM 39N) - 327.68 km x 327.68 km tile
rslearn dataset add_windows --root "$PREDICT_DS" --group predict \
    --box=389120,2621440,716800,2949120 \
    --src_crs EPSG:32639 --crs EPSG:32639 --resolution 10 \
    --grid_size 2048 \
    --start 2025-01-01T00:00:00+00:00 --end 2025-01-01T00:00:00+00:00
```

#### 2. Prepare and materialize imagery

```bash
rslearn dataset prepare     --root "$PREDICT_DS" --workers 32
rslearn dataset materialize --root "$PREDICT_DS" --workers 128
```

#### 3. Run prediction

```bash
rslearn model predict \
    --config data/change_finder_v2/lcc_model/config_predict.yaml \
    --data.init_args.path="$PREDICT_DS"
```

The `PredictPassBuilder` transform takes the quarterly + frequent inputs and
builds pass1/pass2 without needing annotations or multiple frequent options.

#### Output

Per-window output at `<window>/layers/output_change/<bandset>/geotiff.tif`:
- `binary`: 3-channel probability map (nodata / no_change / change)
- `src`: 13-channel source category probabilities
- `dst`: 13-channel destination category probabilities
- `timestamps`: 20-channel sigmoid outputs (per-image change-period membership)

#### 4. Postprocess: raster to GeoJSON

Convert prediction rasters to a GeoJSON of change polygons:

```bash
python -m rslp.change_finder_v2.lcc_model.postprocess \
    --dataset_path "$PREDICT_DS" \
    --output changes.geojson \
    --threshold 128 \
    --min_pixels 10 \
    --workers 32
```

This script:
- Thresholds the binary change band and computes per-pixel argmax src/dst classes.
- Finds connected components separately for each unique (src, dst) class pair,
  so each polygon represents a single type of land cover transition.
- Estimates a change timestamp per polygon via majority vote of per-pixel argmax
  over the 20 timestamp probability bands, mapped to actual dates from the
  dataset's layer metadata.

Each GeoJSON feature includes: `src_class`, `dst_class`, `num_pixels`,
`avg_change_score`, `timestamp_idx`, `timestamp_start`, `timestamp_end`.

---

### Scaled Prediction (global tiled inference)

To apply the LCC model at scale (e.g. globally), the world is divided into
32768x32768 UTM tiles (10 m/pixel). Each tile, for a single reference timestamp,
becomes one task that is written to a Beaker queue and processed by the
`rslp.common` worker system. This is analogous to `rslp.satlas` but
self-contained: it uses the OlmoEarth Datasets imagery source (no rtree index) and
the LCC-specific pre/post-processing.

Per-tile pipeline: create 2048x2048 windows -> materialize Sentinel-2 imagery ->
run the model -> polygonize the `output_change` raster into a per-tile GeoJSON
(`{EPSG}_{col}_{row}.geojson`). Optionally the merged 49-band raster is also written
(`{EPSG}_{col}_{row}.tif`). Merging the per-tile GeoJSONs into a single layer and
converting to vector tiles is handled separately (not yet implemented).

The prediction pipeline accepts any tile size that is a multiple of 2048; only the
job-writer fixes the tile size to 32768.

#### 1. Enqueue tiles for a reference timestamp

```bash
python -m rslp.main change_finder_v2 write_jobs \
    --timestamp 2025-06-01T00:00:00+00:00 \
    --out_path /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_outputs_20260529/ \
    --queue_name favyen/lcc-prediction-queue
```

Optional flags:
- `--epsg_code 32651`: restrict to one UTM zone.
- `--wgs84_bounds '[-122.5, 47.0, -121.5, 48.0]'`: restrict to tiles intersecting a
  WGS84 bounding box.
- `--batch_size 4`: tiles per worker job.
- `--count 100`: randomly sample at most this many tiles (for testing).
- `--write_raster true`: also write the merged 49-band raster per tile.

Tiles whose outputs already exist in `out_path` are skipped, so re-running resumes.

#### 2. Launch workers

The model's imagery source needs the OlmoEarth Datasets credentials, passed via
`--extra_env_vars` (the `DATASETS_API_TOKEN` secret must exist in the
`ai2/earth-systems` Beaker workspace). Start one worker first to warm caches, then
scale up.

```bash
python -m rslp.main common launch \
    --image_name favyen/rslp_image \
    --queue_name favyen/lcc-prediction-queue \
    --num_workers 50 --gpus 1 --shared_memory 256GiB \
    --cluster '["ai2/jupiter", "ai2/neptune", "ai2/saturn"]' \
    --weka_mounts '[{"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}]' \
    --extra_env_vars '[{"name":"OEDATASETS_API_URL","value":"https://datasets.olmoearth.allenai.org"},{"name":"DATASETS_API_TOKEN","secret":"LCC_DATASETS_API_TOKEN"},{"name":"GDAL_HTTP_TIMEOUT","value":"120"},{"name":"RSLP_PREFIX","value":"/weka/dfive-default/rslearn-eai"}]'
```

Each worker pulls tasks from the queue and runs the `predict_multi` workflow.

#### Random 2048x2048 land tiles (diagnostic sampling)

To get diverse predictions from random locations worldwide (useful for spotting
model mispredictions and iterating on annotations), use `write_jobs_random_2048`.
It samples random lat/lon points, snaps them to 2048-pixel-aligned UTM tiles, and
keeps tiles where at least one corner is on land.

```bash
python -m rslp.main change_finder_v2 write_jobs_random_2048 \
    --start_time 2025-01-01T00:00:00+00:00 \
    --end_time 2025-06-01T00:00:00+00:00 \
    --out_path /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_outputs_20260529/ \
    --queue_name favyen/lcc-prediction-queue \
    --count 500
```

Each tile gets a randomly chosen reference timestamp between `start_time` and
`end_time`.

Arguments:
- `--start_time` / `--end_time`: range from which to uniformly sample each tile's
  reference timestamp.
- `--count 500`: number of land tiles to enqueue.
- `--batch_size 4`: tiles per worker job.
- `--write_raster true`: also write the merged 49-band raster per tile.

Launch workers the same way as for scaled prediction above.
