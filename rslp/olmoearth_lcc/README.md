## OlmoEarth LCC: Land Cover Change

OlmoEarth LCC (land cover change) detects recent land cover change from Sentinel-2
time series. It inputs 16 quarterly mosaics (to establish a historical baseline) and
6 recent weekly mosaics from the last 90 days (to detect changes soon after they
occur) and predicts, per pixel, whether a land cover change occurred, the source and destination land cover
categories, the start and end dates of the change, and a fine-grained change category.

The model is trained on manually verified point annotations. See
[dataset_readme.md](./dataset_readme.md) for how the different phases of the dataset
were collected (the dataset is published at
https://huggingface.co/datasets/allenai/olmoearth_lcc).

Package layout:

- `ten_year_dataset/`: create windows for the ten-year Sentinel-2 dataset used to
  mine candidate changes.
- `land_cover/`: train a per-pixel land cover model and apply it to the ten-year
  dataset to find candidate change pixels (see its [README](./land_cover/README.md)).
- `studio/`: upload/download v2 JSONs to/from OlmoEarth Studio for the LCC
  Annotator lab.
- `annotation_scripts/`: per-phase scripts that produce candidate annotation JSONs.
- `codex_tiles/`: Codex-generated 0.1x0.1 degree tiles with likely changes, and a
  script to sample points from them.
- `lcc_model/`: the LCC model, training dataset preparation, and (scaled) prediction.
- `scripts/`: misc utilities.

### Annotation Format

The core data structure is a JSON file (list of dicts). Each entry represents
one 128x128 spatial window:

```json
{
  "projection": {"crs": "EPSG:32651", "x_resolution": 10, "y_resolution": -10},
  "bounds": [0, 0, 128, 128],
  "window_name": "example_window",
  "group": "default",
  "time_range": ["2017-01-01T00:00:00+00:00", "2024-01-01T00:00:00+00:00"],
  "description": "New subdivision replacing forest",
  "positive_points": [
    {
      "lon": 121.5, "lat": 14.6,
      "pre_change": "2020-01-15",
      "first_date_change_noticeable": "2020-04-27",
      "post_change": "2020-07-15",
      "pre_category": "tree",
      "post_category": "urban/built-up",
      "post_change_category": "new_building"
    }
  ],
  "negative_points": [
    {"lon": 121.51, "lat": 14.61}
  ]
}
```

- `positive_points`: locations where land cover change occurred.
- `negative_points`: locations confirmed as no-change.
- `time_range`: metadata indicating when negative points are valid. For samples
  without positive points and without `anchor_date`, its midpoint is used as the
  anchor date (see below).
- `description` (optional): freeform notes on the sample.
- `anchor_date` (optional, only used for samples without positive points): the
  earliest date the input frequent block should end at; the prepare script's
  earliest frequent option ends right after it and the others later. Falls back
  to the `time_range` midpoint when unset.
- Dates may be blank/missing if not yet annotated.
- `pre_change`: the last date at which the location still appears in its pre-change
  state; `first_date_change_noticeable`: the first date at which the change starts
  to become noticeable; `post_change`: the first date at which the change appears
  complete.
- `stop_date` (optional, positive points only): imagery on/after this date must not
  be used for training, e.g. because another unannotated change happens later. It
  must be after `post_change`. When set, the training dataset only samples frequent
  blocks that end on/before the earliest `stop_date` among the entry's points.
- `transition_date`, `transition_category`, `transition_change_category` (optional,
  positive points only): an intermediate stage of a multi-stage change (e.g.
  mangrove -> bare -> crops): the date it appears, its land cover category, and its
  change category (one of the post-change categories, from the same Studio
  labelset as `post_change_category`). `transition_date` must satisfy
  `first_date_change_noticeable <= transition_date < post_change`.

Land cover categories (`pre_category` / `post_category`): nodata, bare, burnt, crops,
fallow/shifting cultivation, grassland, Lichen and moss, shrub, snow and ice, tree,
urban/built-up, water, wetland (herbaceous), mangrove.

Positive points may also carry change-category fields: one `pre_change_category`
(what was removed or lost) and/or one `post_change_category` (what appeared or
what happened). These were introduced partway through annotation, so some points
have neither; during training the change-category heads are only supervised at
points that have at least one of these fields. The annotation lab's labels are
documented in `olmoearth_studio/ui/labs/lcc-annotator/FIELDS.md`. The model is
trained on these classes:

- Pre-change: deforestation, urban_erosion, wetland_loss, water_contract,
  removed_crop_structure, mangrove_loss.
- Post-change: vegetation_growth, new_building, new_road, new_infrastructure,
  new_crop_field, new_aquafarm, site_clearing, water_expand, mining,
  new_crop_structure, selective_logging, landslide, settlement,
  agricultural_activity, wildfire, ice_motion, flooding, storm_damage,
  vegetation_dieback.

The finer lab labels are mapped onto those classes by `prepare.py`:

| Annotation value | Field | Trained as |
| --- | --- | --- |
| river_channel_migration, river_contract, lake_level_drop, reservoir_level_drop, water_contract_other | pre | water_contract |
| river_channel_migration, river_flooding, lake_level_rise, reservoir_level_rise, coastal_erosion | post | water_expand |
| borrow_pit | post | new_road |
| land_reclamation | post | pre-change water_contract (post-change none), unless the point has a pre-change category |

Older annotations had a separate `same_change_category` field (agricultural_activity,
wildfire, ice_motion, flooding); those categories are now post-change categories
and prepare ignores the old field.
### Ten-Year Dataset

The ten-year dataset has 128x128 windows (10 m/pixel, UTM) with ten `sentinel2_yN`
layers, each containing Sentinel-2 mosaics for one year (six per year). It is used to
mine candidate changes by applying a land cover model to each year and comparing the
predictions (see [land_cover/README.md](./land_cover/README.md)).

- Dataset: `/weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/`
- Dataset config: `data/olmoearth_lcc/ten_year_dataset/config.json`

Windows can be added with three workflows:

```bash
# Random land locations worldwide. Samples random (lat, lon) points, filters to land
# with Sentinel-2 coverage (at least one image per 30-day period in the base window),
# snaps to a 128px UTM grid, and saves the windows. Roughly 30% of samples survive
# the ocean filter; coverage filtering removes additional points.
python -m rslp.main olmoearth_lcc create_windows \
    --ds_path /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/ \
    --num_samples 50000 --workers 32

# Random grid cells around ~1000 major world cities.
python -m rslp.main olmoearth_lcc create_windows_urban \
    --ds_path /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/ \
    --group urban --samples_per_city 100 --workers 32

# Random locations within a fixed set of sub-Saharan African countries.
python -m rslp.main olmoearth_lcc create_windows_africa \
    --ds_path /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/ \
    --group africa --num_samples 50000 --workers 32
```

Then materialize the imagery with the standard rslearn pipeline:

```bash
rslearn dataset prepare --root /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/ --workers 32
rslearn dataset materialize --root /weka/dfive-default/rslearn-eai/datasets/change_finder/ten_year_dataset_20260408/ --workers 128
```

### Annotating in OlmoEarth Studio

Annotation happens in the **LCC Annotator** lab in OlmoEarth Studio
(`olmoearth_studio/ui/labs/lcc-annotator/`; see its README for the workflow and
keys). It shows Sentinel-2 rendered on the fly by Studio, so no rslearn dataset
has to be materialized. v2 JSONs move in and out of a Studio project with the
scripts in `studio/`:

```bash
# 1. Upload a v2 JSON into a Studio project. Creates the project's fields and
#    labels if missing (including unseen legacy category values). Tasks that
#    already exist (by name) are skipped, so it is safe to re-run.
STUDIO_API_KEY=... python -m rslp.olmoearth_lcc.studio.upload_to_studio \
    --json annotations.json --project-id <PROJECT_ID> --dry-run
STUDIO_API_KEY=... python -m rslp.olmoearth_lcc.studio.upload_to_studio \
    --json annotations.json --project-id <PROJECT_ID>

# 2. Annotate: open the project in Studio, then Labs -> LCC Annotator.

# 3. Download the project back to a v2 JSON for lcc_model/prepare.py.
STUDIO_API_KEY=... python -m rslp.olmoearth_lcc.studio.download_from_studio \
    --project-id <PROJECT_ID> --out annotations.json
```

Both scripts default to production Studio; set `STUDIO_API_URL` (or pass
`--base-url`) to use another deployment, e.g. `http://localhost:8000/api/v1`.

Each entry becomes a task named `<5-digit JSON index> <window_name>`, so the
lab and the downloaded JSON keep the original order. The window identity,
the verbatim `time_range`, `description`, `anchor_date` and any extra entry
keys are kept in the task's attributes; each point is an annotation whose
`point_type` field says positive or negative. A round trip reproduces the
entries exactly, except that empty-string point fields are dropped (prepare.py
treats them the same as missing ones).

---

### Model

#### Architecture

The production model (`config_bp_abs_qdrop_bpcat_1pass.yaml`) is
`rslp.olmoearth_lcc.lcc_model.model.ChangeModel` (its module docstring describes the
options the other configs use):

- **Input**: 22 images: 16 quarterly mosaics followed by 6 frequent images (the
  least-cloudy mosaic of each of the 6 most recent 7-day periods with imagery in
  a 90-day block).
- **Encoder**: OlmoEarth-v1.2-Base (patch_size=4, no token pooling) over all 22
  images in a single pass (`num_pass1: null`), giving one token per timestep at
  1/4 resolution.
- **Binary head**: a learned breakpoint scan. For every split of the time series it
  compares the before and after means of the tokens, and change evidence is
  max-pooled over the splits.
- **Other heads**: src decodes the split-attention-weighted before aggregate and dst
  the after aggregate; the pre/post change-category heads decode both concatenated.
  The decoders are small conv stacks that upsample to full resolution.
  `ts_start` / `ts_end` are a per-token logit over the 22 timesteps.

Tasks (all per-pixel, loss masked to annotated points only):
- `binary`: change classification (3 cls: nodata / no_change / change; loss weight
  2)
- `src`: source land cover category (14 cls, including nodata)
- `dst`: destination land cover category (14 cls)
- `pre_change` / `post_change`: change-category classification (8 / 21 cls,
  including nodata and none)
- `ts_start` / `ts_end`: index of the input timestep at which the change starts /
  ends (22 cls, one per input image; supervised only at change points)

#### Prerequisites

The prepare and materialize steps use OlmoEarth Datasets for Sentinel-2 scene
discovery and download. Install `olmoearth_run` and set the required env vars:

```bash
pip install -e /path/to/olmoearth_run
export OEDATASETS_API_URL=https://datasets.olmoearth.allenai.org
export DATASETS_API_TOKEN=<your-token>
```

#### Training Dataset Preparation

The prepare script creates the training dataset from one or more v2 annotation
JSONs.
It queries the OlmoEarth Datasets API for both quarterly and frequent imagery,
then rslearn materialize downloads the pixels.

```bash
OUT=/path/to/lcc_model_dataset/
mkdir -p "$OUT"
cp data/olmoearth_lcc/lcc_model/config.json "$OUT/config.json"
python -m rslp.olmoearth_lcc.lcc_model.prepare \
    --v2-json-paths annotations_a.json annotations_b.json \
    --ds-path "$OUT"
rslearn dataset materialize --root "$OUT" --workers 128 --retry-backoff-seconds 5 --retry-max-attempts 5 --no-use-initial-job
```

The prepare script can be re-run on an existing dataset, e.g. after adding new
annotations. Each window is compared with the existing one: unchanged windows are
skipped, windows whose labels changed get their label layers rewritten in place, and
windows whose geometry or dates changed are recreated (and need materializing again).
Then run `rslearn dataset materialize` again to download imagery for the new and
recreated windows.

The prepare script:
- Only processes entries with fully annotated positive points (all of pre_change,
  post_change, first_date_change_noticeable, pre_category, post_category present).
- Skips windows that are unchanged, rewrites the label layers of windows whose
  labels changed, and recreates windows whose geometry or dates changed (see
  above).
- Skips duplicate `group`/`window_name` entries across the provided JSON inputs.
- Creates up to 8 frequent-image options per window. Each option is a 90-day
  block split into 7-day periods counted back from the block end (only full
  periods, so the most recent 84 days); it holds the least-cloudy mosaic (no
  cloud threshold) of each of the 6 most recent periods that have imagery, and
  options with fewer than 6 such periods are skipped. This is the same matching
  as the prediction config's `sentinel2_frequent_0`. Options are defined by where
  the block ends: right after first_noticeable (+6 days), 15-60 days after
  first_noticeable, right after post_change (+6 days), right after the reference
  point's transition_date (+6 days, if it has one), and random later
  post-change contexts up to post_change + 2 years. The transition option comes
  after the post_change option, so validation (option 2) is unchanged; it takes
  the place of one random option. Every block ends on/before
  min(2026-01-01, stop_date), where 2026-01-01 is the annotation cutoff and
  stop_date is the earliest optional per-point stop date.
- Window time_range is derived from the annotations (no fixed 10-year range): it
  starts 1800 days (20 quarterly periods) before the earliest frequent block and
  ends on the 90-day grid through that block's start (just past the latest block
  end). rslearn builds the quarterly mosaics backwards from the end, so one
  quarterly period ends exactly where the earliest option's frequent block begins,
  the same as the prediction config's `sentinel2_quarterly` (`[T - 1890d, T - 90d]`).
  A negative-only entry with `anchor_date = T - 6d` therefore gets an option 0 that
  sees the same 22 images as prediction at reference timestamp T.
  The API orders scenes with equal cloud cover arbitrarily, so an entry may also
  list `prediction_items` (the scene names from the prediction tile's
  `_meta.json`); prepare prefers those scenes on cloud-cover ties, which makes the
  mosaics identical to what prediction used.
- Rasterizes point labels into label_binary/label_src/label_dst layers, the
  label_pre_change/label_post_change category layers (after mapping annotation
  values onto the training classes, see above), and the
  label_ts_start/label_ts_end layers. The last two hold each positive point's own
  pre_change and post_change (plus `--gap_days`) as uint16 days since 1970-01-01
  (65535 = nodata). Re-running prepare on an existing dataset adds them in place
  without re-materializing imagery.
- Rasterizes each positive point's transition into the label_transition_date
  (uint16 days like label_ts_end, plus `--gap_days`), label_transition_dst (land
  cover class like label_dst) and label_transition_post_change (post-change class
  like label_post_change) layers. A missing transition category is nodata.
  `transition_change_category` goes through the post-change aliases, but
  land_reclamation is not moved to the pre-change head: it is nodata for now.
  These layers are new in `config.json`, so copy their entries into the
  `config.json` of an existing dataset before re-running prepare. Windows whose
  reference point has a transition_date are recreated (for the new frequent
  option) and need re-materialization; other windows get the new label layers in
  place.
- Writes `lcc_annotations.json` sidecar for training-time annotation injection
  (merges with existing sidecar on re-runs).

`lcc_model/filter_v2_jsons.py` applies the same window-level rules as the prepare
script to produce filtered copies of the annotation JSONs (e.g. for publishing).
The copies omit the `description` and `fld` entry fields.

#### Training

```bash
rslearn model fit --config data/olmoearth_lcc/lcc_model/config_bp_abs_qdrop_bpcat_1pass.yaml
```

Training details:
- The encoder is trained from the start, with layer-wise learning rate decay:
  LayerDecayAdamW (lr=1e-4, layer_decay_rate=0.65 over the 12 encoder layers) and
  PlateauScheduler (factor=0.2, patience=5, cooldown=10). The best checkpoint is
  chosen by `val_post_change/accuracy`.
- At each training step, one of the (up to 8) frequent-image options is randomly
  sampled, and the 16 quarterly images preceding that option's earliest frequent
  image are selected. At train time each quarterly image is also dropped with
  probability 0.25 (keeping at least 8), so the model sees different mosaic
  compositions of the same window.
- Val/test uses option 2 deterministically.
- The dst and post_change heads are only trained on what the sampled images can
  show. Per positive point, with `end` the end of the latest frequent image: if
  `end >= post_change` they use post_category / post_change_category; else if the
  point has a transition and `end >= transition_date` they use
  transition_category / transition_change_category (each masked if unset);
  otherwise (the change is still in progress) they are masked. The `ts_end` target
  is post_change, or transition_date between the transition and post_change. While
  the change is in progress it is the next of those dates, after the stack, so it
  maps to the last timestep. The `ts_start` target is each point's own pre_change.
  Negative points still train the change-category heads as "none". The per-point
  dates come from the label_ts_start, label_ts_end and label_transition_* layers,
  which the training configs pass through to `StackSampler` (as optional inputs, so
  older datasets without the transition layers still load); without label_ts_start
  or label_ts_end, the window's sidecar pre_change or post_change is used. Which
  images are sampled still depends only on the window's reference point (the first
  fully dated positive point).
- Augmentation: random horizontal/vertical flips.
- crop_size: 64, batch_size: 4, 100 epochs max.

The other `config_*.yaml` files in `data/olmoearth_lcc/lcc_model/` are variants that
were tried along the way; `config_bp_abs_qdrop_bpcat_1pass.yaml` (and its `_predict`
counterpart) is the one used by the prediction pipeline.

`config_rslearn_bpcat_notemporal.yaml` expresses the model with generic rslearn
components (`BreakpointScan`, `TokensToChannels`) instead of
`rslp.olmoearth_lcc.lcc_model.model.ChangeModel`, so the checkpoint can be run by
systems that depend on rslearn alone. Its `_predict` counterpart contains no `rslp`
imports at all.

In this config, `ts_start` / `ts_end` use rslearn's `PerPixelTimestepHead` and
`PerPixelTimestepTask`. Their labels come from the `label_ts_start` / `label_ts_end`
layers (per-point dates) as day targets, instead of the timestep indices that
`StackSampler` computes, which this config runs with `emit_timestep_targets: false`. The head
maps each labeled date to an index of the sampled 22-image stack: `ts_start` uses
`mode: BEFORE` (the latest image on or before pre_change) and `ts_end` uses
`mode: AFTER` (the earliest image on or after post_change, or the transition_date
as described above), compared at day granularity. `ts_end` sets
`fallback_to_closest_timestep_if_no_match`, so a date after the stack maps to the
last timestep; for `ts_start`, pixels where no image qualifies are excluded from
the loss. Training is
then a 22-way classification over the input timestep index, but at prediction the
`_predict` config writes each pixel's
predicted change start/end as a date, in uint16 days since 1970-01-01 (65535 =
nodata), to the `output_ts_start` / `output_ts_end` layers of
`config_predict_rslearn.json`, instead of the timestep index. The date is the midpoint
of the predicted input image's time range. For a mosaic, that is the time range of its
first (least cloudy) item, not of the 7-day or 90-day period.

#### OlmoEarth Studio (olmoearth_run)

The Studio configs for the `rslearn_bpcat_notemporal` model live in olmoearth_projects
under `olmoearth_run_data/olmoearth_lcc/`. `model.yaml` there is derived from
`config_rslearn_bpcat_notemporal_predict.yaml` and `dataset.json` from
`config_predict_rslearn.json`; keep the model blocks in sync when the model changes.
Those Studio configs have not yet been updated for the 22-image stack (6 frequent
images from a 90-day block, quarterly ending at T-90d) and must be synced before
deploying a model trained on it.
Studio currently supports a single output per model, so only the `post_change` head
(argmax class index) is written there; the other heads will be exposed once
multi-output support is available. Where the Studio configs include `ts_start` /
`ts_end`, they still write timestep indices. Switching them to the date head and task
needs olmoearth_run changes: it writes all fields to one band set with a single dtype
(currently uint8, so it would need to become uint16), and it requires all fields to
share one field type and one nodata value, while the other fields are segmentation
classes with nodata 0 and the dates use nodata 65535. See the README in that
directory for the time range semantics (`oe_start_time` is the reference date) and how
to run it.

---

### Prediction

For prediction, a separate dataset config handles imagery via rslearn's
standard prepare/materialize pipeline (using OlmoEarth Datasets). The user
creates windows with a single point-in-time timestamp T, set to the
current/reference time you want to evaluate. The frequent stack is the
least-cloudy mosaic of each of the 6 most recent 7-day periods with imagery in the
90 days before T, and the quarterly mosaics come from the ~5 years before that
90-day block.

#### 1. Create windows

```bash
PREDICT_DS=/path/to/predict_dataset/
mkdir -p "$PREDICT_DS"
cp data/olmoearth_lcc/lcc_model/config_predict.json "$PREDICT_DS/config.json"

# Create windows over an AOI. The time range is a single point in time: T, the
# current/reference time. The frequent stack covers [T-90d, T].
# Example: to evaluate current/reference time 2025-06-01, use T=2025-06-01.
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
- `sentinel2_quarterly`: time_offset=-1890d, duration=1800d → searches
  [T-1890d, T-90d] for quarterly mosaics (ending where the frequent block
  starts); prediction uses the last 16.
- `sentinel2_frequent_0`: time_offset=-90d, duration=90d, period_duration=7d,
  sorted by cloud cover ascending, max_matches=6, min_matches=6 → searches
  [T-90d, T] week by week from T backwards, skipping weeks with no imagery, and
  returns the least-cloudy mosaic of each of the 6 most recent weeks with
  imagery. Windows with fewer than 6 such weeks get no frequent layer and
  therefore no prediction.

`config_predict_rslearn.json` uses the same frequent layer. Prediction windows
(2048 px) combine multiple scenes within a week (e.g. across tile or swath edges)
more often than the 128 px training windows do.

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
    --config data/olmoearth_lcc/lcc_model/config_bp_abs_qdrop_bpcat_1pass_predict.yaml \
    --data.init_args.path="$PREDICT_DS"
```

The `PredictStackBuilder` transform takes the quarterly + frequent inputs and
builds the 22-image stack without needing annotations or multiple frequent options.

#### Output

Per-window output at `<window>/layers/output_change/<bandset>/geotiff.tif` (62 bands):
- `binary`: 3-channel probability map (nodata / no_change / change)
- `src`: 14-channel source category probabilities
- `dst`: 14-channel destination category probabilities
- `ts_pre_days` / `ts_post_days`: predicted pre/post change dates as integer days
  since the timestamp epoch
- `pre_change` / `post_change`: 8/21-channel change-category probabilities (class
  layout [nodata, none, <options...>])

The band layout is defined in `lcc_model/output_bands.py`.

---

### Scaled Prediction (global tiled inference)

To apply the LCC model at scale (e.g. globally), the world is divided into
16384x16384 UTM tiles (10 m/pixel, 64 windows of 2048x2048 each; about 24,800 tiles
globally). Each tile, for a single reference timestamp, becomes one task file in a
job directory on WEKA, which workers on Beaker claim and predict one tile at a time
(see "Job directory" below). Imagery comes from the OlmoEarth Datasets source.

Tiling (`lcc_model/tiling.py`): each UTM zone number (1-60) is processed once, in its
northern CRS (EPSG:326NN), with a continuous northing axis that goes negative south
of the equator, so one zone covers 80S to 84N. Southern-hemisphere tiles therefore
have EPSG:326NN names, with negative northings (positive pixel rows, since the y
resolution is negative). Tiles and 2048x2048 windows are only kept if they intersect
their zone's canonical 6-degree longitude wedge, so each location is predicted in
(nearly) one zone. Windows are also skipped if no point on a 512-pixel lattice (25
points including corners and edges) is land according to `global_land_mask`. That
includes all-water windows within Sentinel-2's coastal coverage, which would
otherwise be predicted. The mask is about 1 km resolution, so islands smaller than
that may be missed.

Per-tile pipeline (`lcc_model/predict_pipeline.py`): create 2048x2048 windows and
materialize their Sentinel-2 imagery (`prepare_tile`), then run the model
(`predict_tile`). The model's outputs go to `LCCTileWriter`
(`lcc_model/prediction_writer.py`), which replaces the predict config's
`RslearnWriter` through a `--trainer.callbacks` override. It merges each window's
crops in memory and, in background threads, writes the window into the GeoZarr
store (see below), so no per-window GeoTIFF is written or read back. The pipeline
then writes the tile's pyramid levels if a pyramid store is given, and last a
`<CRS>_<x>_<y>_meta.json` (e.g. `EPSG:32610_49152_-540672_meta.json`, with x/y the
pixel coordinates of the tile's top-left corner) recording the input scenes of each
window and the per-stage timings.

The worker (`lcc_model/worker.py`) claims the next tile and prepares it in a
background process (spawned, so it doesn't inherit CUDA state) while the current
tile is predicted on the GPU, so only a worker's first tile waits for
materialization. Each tile gets its own scratch directory, so the scratch disk holds
two tiles of imagery at once. The `materialize_wait` timing in `_meta.json` is how
long prediction waited for the tile's preparation, and `write_wait` how long it
waited for the writer threads after the last batch.

The prediction pipeline accepts any tile size that is a multiple of 2048; only the
job writer fixes the tile size to 16384 (`tiling.TILE_SIZE`). Output filenames
include the tile bounds, so don't resume an output directory from a run with a
different tile size.

#### 1. Write tiles for a reference timestamp to a job directory

```bash
python -m rslp.main olmoearth_lcc write_jobs \
    --timestamp 2025-06-01T00:00:00+00:00 \
    --out_path /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_outputs_20260529/ \
    --job_dir /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_jobs_20250601/ \
    --zarr_store_path /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_20250601.zarr
```

`--job_dir` is the job directory (see "Job directory" below). It must be a path on
the WEKA mount that the workers also see (not a `gs://` or other URL).

`--zarr_store_path` is required: the GeoZarr store to write predictions into (see
below). It is created if missing.

Optional flags:
- `--zarr_detail full`: create the store in full mode, with the per-head
  probabilities (meant for small runs; default `compact`).
- `--zarr_pyramid_path /path/to/lcc_20250601_pyramid.zarr`: also write each tile's
  overviews into a separate pyramid store (see below), created if missing.
- `--epsg_code 32651`: restrict to one UTM zone (326NN and 327NN select the same
  zone, which covers both hemispheres).
- `--wgs84_bounds '[-122.5, 47.0, -121.5, 48.0]'`: restrict to tiles intersecting a
  WGS84 bounding box.
- `--count 100`: randomly sample at most this many tiles (for testing).
- `--fast_inference false`: run the model eagerly. By default the OlmoEarth encoder
  is compiled with `torch.compile` and uses cuDNN attention (see
  `lcc_model/fast_inference.py`): same speed as eager at batch size 8 but about
  1.7 GB instead of 16.5 GB of GPU memory. That requires torch 2.8 or newer and has
  only been tested on H100. The setting is recorded in each tile's `_meta.json`.

These settings are stored in the job directory's `config.json`, which the workers
read. Tiles whose `_meta.json` already exists in `out_path`, or that already have a
task file, are skipped, so re-running resumes or extends the job (with the same
settings).

#### 2. Launch workers

The model's imagery source needs the OlmoEarth Datasets credentials, passed via
`--extra_env_secrets` (the `LCC_DATASETS_API_TOKEN` secret must exist in the
`ai2/earth-systems` Beaker workspace). Start one worker first to warm caches, then
scale up.

```bash
python -m rslp.main olmoearth_lcc launch_workers \
    --job_dir /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_jobs_20250601/ \
    --image_name favyen/rslp_image \
    --num_workers 50 --gpus 1 --shared_memory 256GiB \
    --cluster '["ai2/jupiter", "ai2/ceres"]' \
    --weka_mounts '[{"bucket_name":"dfive-default","mount_path":"/weka/dfive-default"}]' \
    --extra_env_vars '{"OEDATASETS_API_URL":"https://datasets.olmoearth.allenai.org","RSLP_PREFIX":"/weka/dfive-default/rslearn-eai"}' \
    --extra_env_secrets '{"DATASETS_API_TOKEN":"LCC_DATASETS_API_TOKEN"}'
```

Each worker runs the `worker` workflow on the job directory: it predicts tiles until
none is claimable, then exits.

With fast inference (the default), compiled kernels are cached in inductor's default
directory on the worker's local disk (`/tmp/torchinductor_<user>`) and reused across
tiles. Don't point `TORCHINDUCTOR_CACHE_DIR` at a directory shared with non-cuDNN
runs (e.g. on Weka): the inductor FX cache doesn't key on the SDPA backend.

#### 3. Check progress

```bash
python -m rslp.main olmoearth_lcc job_status \
    --job_dir /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_jobs_20250601/
```

It logs how many tiles are done (have a `_meta.json`) and, of the remaining tasks,
how many are claimed, stale, failed or unclaimed.

#### Job directory

Instead of a Beaker queue, tasks live in a job directory on WEKA
(`lcc_model/job_dir.py`, whose docstring has the details), accessed with plain POSIX
calls:

```
<job_dir>/
  config.json                    # shared worker settings (from write_jobs)
  tasks/<tile>.json              # one task per tile
  claims/<tile>.claim.<gen>      # versioned claims, refreshed by the holder
  failures/<tile>.<uuid>         # one marker (with the traceback) per failed attempt
```

`<tile>` is the stem of the tile's `_meta.json` (`<CRS>_<x>_<y>`).

- **Claiming:** a worker picks a random task and exclusively creates
  `claims/<tile>.claim.<g+1>`, where `g` is the newest existing generation; only one
  worker can win, even across nodes.
- **Stale claims:** each worker refreshes the mtime of the claims it holds every
  `refresh_seconds`. A claim not refreshed for `stale_seconds` can be taken over
  by creating the next generation. On SIGTERM (e.g. preemption) a worker releases
  its claims right away.
- **Completion:** after writing a tile's `_meta.json` (still the completion marker),
  the worker deletes the task file and then the tile's claims. Tiles that already
  have a `_meta.json` are skipped and cleaned up.
- **Failures:** a failed tile gets a failure marker and its claim is deleted, so
  another worker can retry it; a worker doesn't retry tiles that failed in it.
  Tiles with `max_failures` markers are skipped. A worker exits after
  `max_consecutive_errors` failures in a row.

Since workers exit once nothing is claimable, tiles held by workers that died (e.g.
preempted without SIGTERM) become claimable only after `stale_seconds`; launch a
few more workers later to pick them up. To retry tiles that hit `max_failures`,
delete their markers in `failures/`.

#### GeoZarr output

A store normally holds the predictions for one reference timestamp, named in its
path (e.g. a new store per month for monthly prediction); the random-tile job
writers instead put tiles with different timestamps into one store. It is a Zarr v3
store with one group per UTM zone,
`utm01` to `utm60`, each in EPSG:326NN at 10 m/pixel using the grid from
`tiling.get_zone_grid`. Group attributes follow the
[spatial](https://github.com/zarr-conventions/spatial) and
[proj](https://github.com/zarr-conventions/proj) conventions (`proj:code`,
`spatial:transform`, `spatial:shape`, `spatial:bbox`). Each group also has x/y
coordinate arrays (pixel centers). Each zone group holds two arrays:

- `change`: uint8, dimensions `(change_band, y, x)`:
  1. `binary_change`: binary change probability (0-255).
  2. `pre_lc`: pre-change land cover class (0 = nodata, otherwise an index into
     `lc_class_names`).
  3. `post_lc`: post-change land cover class (same encoding).
  4. `pre_change_category`: 0 = no prediction, 1 = none, otherwise an index into
     `pre_change_category_names`.
  5. `post_change_category`: same, indexing `post_change_category_names`.
  6. `post_change_score`: probability (0-255) of the post change category.
- `change_dates`: uint16, dimensions `(change_dates_band, y, x)`:
  1. `change_start`: predicted change start, days since 1970-01-01 (0 = nodata).
  2. `change_end`: predicted change end, same encoding.

Pixels with no prediction are 0 in every band. Where both change categories are
none and `binary_change < 128`, `pre_lc`, `post_lc`, `post_change_score` and both
dates are also 0, and `binary_change < 16` is stored as 0 everywhere (these low
probabilities were most of the compressed size). The class-name lists, detail mode
and model config are stored in the root attributes.

The store's detail mode (`detail` root attribute) is set at creation. Compact stores
(the default) hold only the two arrays above. Full stores (`--zarr_detail full`,
meant for small runs) add one uint8 array per classification head with the softmax
probabilities scaled 0-255, without the nodata class and without zeroing. Each has
its own band dimension, with the class names as band coordinates:

- `src_probs` / `dst_probs` (`src_probs_band` / `dst_probs_band`): the 13 land
  cover classes.
- `pre_change_probs` (`pre_change_probs_band`): "none" plus the 6 pre-change
  categories.
- `post_change_probs` (`post_change_probs_band`): "none" plus the 19 post-change
  categories.

`write_jobs` fails if an existing store has a different detail mode.

The shards are 2048x2048, one per prediction window, so concurrent workers never
write the same object. Inner chunks are 512x512 and all arrays use zstd level 9.

```python
import xarray as xr

ds = xr.open_zarr("/path/to/lcc_20250601.zarr", group="utm10", consolidated=False)
ds["change"].sel(change_band="post_change_category")
```

xarray decodes `change_dates` to datetimes (nodata becomes NaT) from its `units`
attribute; pass `decode_times=False` to get the raw day values.

#### GeoZarr pyramid

With `--zarr_pyramid_path`, each tile's overviews are written into a separate
GeoZarr store (see `lcc_model/geozarr_pyramid.py`), so the 10 m store keeps its
layout; a web client reads level 0 from the 10 m store and the overviews from the
pyramid store, using the same zone group names.

Each zone group `utmNN` of the pyramid store has child groups `1` to `5` (20, 40, 80,
160 and 320 m/pixel), each with the `change` and `change_dates` arrays (same bands
and encodings as level 0, for both compact and full main stores), band coordinates
and x/y. The zone group has a
[multiscales](https://github.com/zarr-conventions/multiscales) attribute listing the
levels with their `spatial:transform` / `spatial:shape`; level 1 is derived from
level 0 in the main store, which the root `source_store` and zone group
`source_group` attributes record. `write_jobs` fails if an existing pyramid store
records a different `source_store`. Pyramids need tiles aligned to 16384, so the
random-tile job writer can't write one.

Each level is a 2x reduction of the previous one: every output pixel copies all
bands from one pixel of its 2x2 block, picking the one that maximizes
`valid + 2 * binary_change + 512 * pre_or_post_change` (pre_or_post_change: a pre or
post change category other than none; valid: any prediction). The bands stay
consistent with each other and small changes stay visible at coarse levels. Each
tile's job writes its own overviews, so there is no global pass and resuming still
works.

```python
ds = xr.open_zarr("/path/to/lcc_20250601_pyramid.zarr", group="utm10/3", consolidated=False)
```

#### Random 2048x2048 land tiles (diagnostic sampling)

To get diverse predictions from random locations worldwide (useful for spotting
model mispredictions and iterating on annotations), use `write_jobs_random_2048`.
It samples random lat/lon points, snaps them to 2048-pixel-aligned tiles in their
UTM zone's northern projection (EPSG:326NN, like the GeoZarr store), and keeps tiles
with land (the same check the prediction pipeline applies to its windows).

```bash
python -m rslp.main olmoearth_lcc write_jobs_random_2048 \
    --start_time 2025-01-01T00:00:00+00:00 \
    --end_time 2025-06-01T00:00:00+00:00 \
    --out_path /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_model_outputs_20260529/ \
    --job_dir /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_random_jobs_20260529/ \
    --zarr_store_path /weka/dfive-default/rslearn-eai/datasets/change_finder/lcc_random_20260529.zarr \
    --count 500
```

Each tile gets a randomly chosen reference timestamp between `start_time` and
`end_time` (recorded in its `_meta.json`), so the GeoZarr store (created if
missing) holds tiles with different timestamps.
Tiles are deduplicated by location within the run. Sampled tiles that already have
a `_meta.json` in `out_path` or a task file in the job directory are not queued
again (so a rerun may queue fewer than `--count` new tiles).

Arguments:
- `--start_time` / `--end_time`: range from which to uniformly sample each tile's
  reference timestamp.
- `--zarr_store_path`: the GeoZarr store (required); `--zarr_detail full` for
  the per-head probabilities.
- `--job_dir`: the job directory to write the tasks to.
- `--region china` / `--region africa`: sample within a rough polygon of mainland
  China or sub-Saharan Africa instead of worldwide (`global`, the default).
- `--count 500`: number of land tiles to write.
- `--fast_inference false`: as for `write_jobs`.

Launch workers on the job directory the same way as for scaled prediction above.
