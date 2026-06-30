# DRC Ituri Worldcover Inference

Investigate potential deforestation linked to the April/May 2026 Ebola outbreak in
Ituri Province, northeastern DRC. Uses the `worldcover_4ts` model (4 Sentinel-2
timesteps) to run land cover segmentation over Ituri in January 2026 and May 2026.

Affected health zones: Bunia, Mongbwalu, Rwampara.

## Geographic bounds (Ituri Province)

WGS84 bounding box: `27.5,1.0,31.5,3.5` (west, south, east, north)

## Window count estimate

At 10m resolution with `--grid_size 2048`:
- ~22 cols × 14 rows = **~300 windows per time period**
- Two groups (jan2026 + may2026) = **~600 total windows**

## Setup

### 1. Create the inference dataset

Copy `config.json` to the dataset root and create windows:

```bash
DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/2026_05_19_drc_ituri_worldcover
mkdir -p $DATASET_PATH
cp one_off_projects/2026_05_19_drc_worldcover/config.json $DATASET_PATH/

# December 2025 — 4 weekly mosaics within Dec 1–31
rslearn dataset add_windows --root $DATASET_PATH --group dec2025 \
    --src_crs EPSG:4326 --utm --resolution 10 \
    --box=27.5,1.0,31.5,3.5 \
    --grid_size 2048 \
    --start 2025-12-01T00:00:00+00:00 --end 2026-01-01T00:00:00+00:00
# January 2026 — 4 weekly mosaics within Jan 1–31
rslearn dataset add_windows --root $DATASET_PATH --group jan2026 \
    --src_crs EPSG:4326 --utm --resolution 10 \
    --box=27.5,1.0,31.5,3.5 \
    --grid_size 2048 \
    --start 2026-01-01T00:00:00+00:00 --end 2026-02-01T00:00:00+00:00
# February 2026 — 4 weekly mosaics within Feb 1–28
rslearn dataset add_windows --root $DATASET_PATH --group feb2026 \
    --src_crs EPSG:4326 --utm --resolution 10 \
    --box=27.5,1.0,31.5,3.5 \
    --grid_size 2048 \
    --start 2026-02-01T00:00:00+00:00 --end 2026-03-01T00:00:00+00:00

# May 2026 — 4 most recent weekly mosaics within Apr 19 – May 19
rslearn dataset add_windows --root $DATASET_PATH --group may2026 \
    --src_crs EPSG:4326 --utm --resolution 10 \
    --box=27.5,1.0,31.5,3.5 \
    --grid_size 2048 \
    --start 2026-04-19T00:00:00+00:00 --end 2026-05-19T00:00:00+00:00
```

### 2. Prepare and materialize

Run once for the whole dataset before inference:

```bash
rslearn dataset prepare --root $DATASET_PATH --workers 32
rslearn dataset materialize --root $DATASET_PATH --workers 32
```

### 3. Train the model

Training uses the worldcover finetuning dataset at
`/weka/dfive-default/rslearn-eai/datasets/worldcover` (set in the yaml).

```bash
rslearn model fit --config one_off_projects/2026_05_19_drc_worldcover/worldcover_4ts.yaml
```

The model uses patch size 2 so it can find pretty fine-grained changes. Due to the small
patch size, it also uses input size 16x16.

### 4. Run inference

`worldcover_4ts.yaml` is configured for inference on this dataset. Sliding-window
inference uses `crop_size: 16` (matching training) over 2048×2048 windows. Override
the dataset path and group:

```bash
# December 2025
rslearn model predict --config one_off_projects/2026_05_19_drc_worldcover/worldcover_4ts.yaml \
    --data.init_args.path=$DATASET_PATH \
    --data.init_args.predict_config.groups='["dec2025"]'
# January 2026
rslearn model predict --config one_off_projects/2026_05_19_drc_worldcover/worldcover_4ts.yaml \
    --data.init_args.path=$DATASET_PATH \
    --data.init_args.predict_config.groups='["jan2026"]'
# February 2026
rslearn model predict --config one_off_projects/2026_05_19_drc_worldcover/worldcover_4ts.yaml \
    --data.init_args.path=$DATASET_PATH \
    --data.init_args.predict_config.groups='["feb2026"]'

# May 2026
rslearn model predict --config one_off_projects/2026_05_19_drc_worldcover/worldcover_4ts.yaml \
    --data.init_args.path=$DATASET_PATH \
    --data.init_args.predict_config.groups='["may2026"]'
```

Predictions are written to the `output` layer under each window via `RslearnWriter`.
With `management_dir` set, the best checkpoint is loaded automatically.

The model outputs 12-band float32 softmax probabilities (`output_probs: true`),
one band per worldcover class.

### 5. Detect land cover changes

Compare the pre-period (dec2025, jan2026, feb2026) and post-period (may2026)
probability outputs to find locations where tree/forest cover was confidently
lost. For each pixel the **minimum** tree probability across all pre-period
months must exceed `--pre_threshold` (0.75), and the May probability must fall
below `--post_threshold` (0.25). The pre-period confident mask is eroded by
`--erode_pixels` (default 1) to exclude boundary artifacts at forest edges
(it will apply this many iterations of binary erosion on the mask, so e.g.
single pixel lines will be eroded entirely).

Windows are matched across groups by their spatial bounds prefix.

```bash
python one_off_projects/2026_05_19_drc_worldcover/detect_changes.py \
    --ds_path $DATASET_PATH \
    --out_path deforestation_events.geojson
```

Output is a GeoJSON `FeatureCollection` of Point features with properties:
- `pre_category` / `post_category` — dominant class name at the centroid
  (from argmax of the mean pre-period / post-period probability map)
- `timestamp` — approximate date of the change (default `2026-04-01`)
- `num_pixels` — number of pixels in the connected component

Override pre-period groups with `--pre_groups dec2025,jan2026,feb2026`.

To detect loss of a different class, pass `--pre_class_index` (default 8 =
tree). Class indices: 0=bare, 1=burnt, 2=crops, 3=fallow, 4=grassland,
5=lichen\_moss, 6=shrub, 7=snow\_ice, 8=tree, 9=urban, 10=water, 11=wetland.

Note: it may be useful to remove forest loss events after the initial GeoJSON is
generated, most of these seem to be false positives.

## Dataset config notes

- **Sentinel-2**: 12 bands, `max_matches: 4` with `period_duration: "7d"` — splits
  each window's time range into weekly sub-periods and keeps the 4 most recent,
  preferring lowest cloud cover within each period.
- **Output layer**: float32 raster with 12 bands (per-class softmax probabilities).

## Model config notes

- `predict_config`: `load_all_crops: true`, `crop_size: 16`, `overlap_pixels: 4`,
  `skip_targets: true`
- `output_probs: true` on `SegmentationTask` — writes softmax probabilities
  instead of argmax class indices
- Transforms inherited from `default_config`: `RandomTimeDropping` (min_keep=4) +
  `OlmoEarthNormalize`
