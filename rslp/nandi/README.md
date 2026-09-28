# Nandi County Crop Type Classification

## v2: polygon windows (2026-09-21)

The v1 pipeline exploded 819 ground truth polygons into ~19K 10 m points and created
one 32x32 window per point, with only the centre pixel labelled. That meant ~23K
forward passes per epoch to supervise ~23K pixels, ~20 near-duplicate windows per
polygon, and a model that learned the answer always sits at the window centre.

v2 keeps the polygons intact. Windows are laid on a fixed 16x16 grid over the label
footprint, and every polygon intersecting a window is burned into its label raster:

| | v1 (point) | v2 (polygon) |
|---|---|---|
| windows | 23,047 | 2,180 |
| labelled px / window | 1 | ~20 |
| labelled px / epoch | 23,047 | 43,707 |
| label density | 0.025% | 7.8% |
| classes | 10 | 11 (adds Shrubland) |

Pixels not covered by a polygon **centre** are IGNORE (255) and masked out of the loss
and metrics. That deliberately discards the boundary ring `all_touched=True` would
have claimed -- 11,381 px, 33% of everything it would label, all of them mixed pixels
straddling a field edge.

### Why Shrubland

ESA WorldCover over Nandi County is 44.0% Tree cover, **27.6% Shrubland**, 15.0%
Cropland, 13.2% Grassland, 0.47% Built-up. Shrubland had no class in the v1 taxonomy,
so every shrubland pixel at inference was forced into Grassland, Trees or a crop class.
v2 adds it from WorldCover blobs, alongside the Water and Built-up classes v1 already
sampled as points.

WorldCover Tree cover (10), Cropland (40) and Grassland (30) are deliberately **not**
used: Coffee and Tea both map to Tree cover or Cropland there, so taking labels from
those codes would teach the model the exact Trees/Coffee confusion we want to remove.

### Class table

`rslp/nandi/classes.py` is the single source of truth. The index in `CLASS_NAMES` is
the class ID in `label_raster` and the model's output channel, and the same list is
passed to `SegmentationTask(class_names=...)` so per-class metrics and the confusion
matrix label themselves.

### Step 1. Build the unified label GeoJSON

```bash
python -m rslp.nandi.prepare_label_polygons \
  --gt_shapefile /weka/dfive-default/yawenz/datasets/CGIAR/NandiGroundTruth/NandiGroundTruth.shp \
  --studio_geojson /weka/dfive-default/yawenz/datasets/CGIAR/20250910_annotations.geojson \
  --worldcover_tif /weka/dfive-default/yawenz/datasets/CGIAR/WorldCover/NandiCounty_worldcover.tif \
  --out_path /weka/dfive-default/yawenz/datasets/CGIAR/nandi_labels_v2.geojson
```

Pass `--studio_geojson` once per Studio export to add more annotations; any GeoJSON
with a `category` property (or a Studio `metadata_values` `tag_name`) works.

`--max_pixels_per_class_source` (default 6000) caps each (class, source) group by
cutting oversized polygons into 160 m tiles and sampling them. Without it the 14 Studio
forest polygons alone contribute ~73K px against ~3K for each surveyed crop class.

### Step 2. Create windows and rasterize labels

```bash
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20260921

python -m rslp.nandi.create_polygon_windows \
  --labels_path /weka/dfive-default/yawenz/datasets/CGIAR/nandi_labels_v2.geojson \
  --ds_path $DATASET_PATH --group polygon_grid_16 --window_size 16

python -m rslp.nandi.rasterize_polygon_labels \
  --ds_path $DATASET_PATH --group polygon_grid_16 --workers 64
```

Splits are assigned by hashing a 2 km spatial block, not a polygon ID. A grid window
can straddle a polygon, so a polygon-level hash would leak that polygon across splits;
blocks much larger than a window keep every window touching a polygon on one side.

To leave room for random cropping during fine-tuning, use `--window_size 32
--grid_size 16`. Every 16x16 crop of a 32x32 window then overlaps the labelled centre
cell by at least 8x8.

### Step 3. Materialize

```bash
rslearn dataset prepare --root $DATASET_PATH --group polygon_grid_16 --workers 64 --retry-max-attempts 8
rslearn dataset materialize --root $DATASET_PATH --group polygon_grid_16 --workers 64 --retry-max-attempts 8
```

### Step 4. Train

Configs live in `data/helios/v3_nandi_polygon/`. `common.yaml` holds the data module
and task; compose it with one model config.

Linear probe (frozen OlmoEarth v1.2 Base, `EmbeddingCache`, 1x1 Conv head). The encoder
runs once per window in epoch 1 and every later epoch is a matmul over ~1.7 GB of
cached features, so this is the cheap first signal:

```bash
python -m rslp.main common beaker_train --image_name <image> --cluster+=ai2/titan-cirrascale \
  --config_paths+=data/helios/v3_nandi_polygon/common.yaml \
  --config_paths+=data/helios/v3_nandi_polygon/linear_probe_s2.yaml
```

Full fine-tune, using `LayerDecayAdamW` (per-layer LR decay) rather than the old
freeze-then-unfreeze recipe:

```bash
python -m rslp.main common beaker_train --image_name <image> --cluster+=ai2/titan-cirrascale \
  --config_paths+=data/helios/v3_nandi_polygon/common.yaml \
  --config_paths+=data/helios/v3_nandi_polygon/finetune_s2.yaml
```

Both log to W&B project `20260921_nandi_v2`. Checkpoint selection is on
`val_segment/AverageF1` (macro), not accuracy: train pixel counts span 307
(Vegetables) to 7295 (Trees), so a micro average just tracks the big classes.

### Step 5. Predict

Create large county windows on the same UTM grid. 1024 is a multiple of the 16 px
training grid, so inference crops line up with training windows:

```bash
rslearn dataset add_windows --root $DATASET_PATH --group nandi_county \
  --utm --resolution 10 --grid_size 1024 \
  --fname /weka/dfive-default/yawenz/datasets/CGIAR/Nandi_County/nandi_county.shp \
  --start 2023-03-01T00:00:00+00:00 --end 2023-03-31T00:00:00+00:00
rslearn dataset prepare --root $DATASET_PATH --group nandi_county --workers 64
rslearn dataset materialize --root $DATASET_PATH --group nandi_county --workers 64
```

```bash
python -m rslp.main common beaker_train --image_name <image> --cluster+=ai2/saturn-cirrascale \
  --mode predict --gpus 4 \
  --config_paths+=data/helios/v3_nandi_polygon/common.yaml \
  --config_paths+=data/helios/v3_nandi_polygon/finetune_s2.yaml \
  --config_paths+=data/helios/v3_nandi_polygon/predict_s2.yaml
```

`predict_s2.yaml` uses `load_all_crops` with `crop_size: 16` and `overlap_pixels: 4`,
and `RslearnWriter`'s `RasterMerger` trims the overlap from each crop. That replaces
the v1 approach of tiling into 128x128 windows and stitching afterwards with
`scripts/merge_geotiff.py`; `scripts/cleanup_geotiff.py` is still useful for removing
small islands.

Note that `predict_s2.yaml` replaces `trainer.callbacks` rather than appending to it,
which is intended -- checkpoint loading in predict mode comes from `management_dir`
plus `load_checkpoint_mode`, not from the checkpoint callback.

### Known limitations

- **Rare-class val/test is thin.** Vegetables has 396 px in total (65 val, 24 test) and
  Water 1,618 px clustered in a handful of blobs (14 val, 22 test). Per-class metrics
  for these two will be very noisy; treat them as directional only. More Studio
  polygons for Vegetables would be the highest-value annotation to collect next.
- **Inference cost.** At `patch_size: 1` the county is ~37M px, which is ~256K crops of
  16x16 with overlap. This is inherent to per-pixel patching; dropping to `patch_size: 2`
  at inference would quarter it, at some accuracy cost.
- **WorldCover is 2021, the labels are 2023.** WorldCover blobs are eroded, size
  filtered, and dropped within 100 m of any surveyed polygon
  (`--worldcover_exclusion_m`), so the weak labels never sit next to ground truth. They
  are still weak labels.

---

## v1: point windows (legacy)


We start with 819 ground-truth polygons, converted into ~19K points at 10 m resolution. The original dataset doesn't have Water and Built-up. To support LULC mapping, we added 1K sampled points each from ESA WorldCover.

---

## Step 1. Create Windows

**Ground-truth:**
```bash
python rslp/nandi/create_windows_for_groundtruth.py --csv_path=/weka/dfive-default/yawenz/datasets/CGIAR/NandiGroundTruthPoints.csv --ds_path=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250815 --window_size=32
```

**WorldCover:**
```bash
python rslp/nandi/create_windows_for_worldcover.py --csv_path=/weka/dfive-default/yawenz/datasets/CGIAR/NandiWorldCoverPoints_sampled.csv --ds_path=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250815 --window_size=32
```

---

## Step 2. Prepare / Materialize Windows

**Ground-truth:**
```bash
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250815
export DATASET_GROUP=groundtruth_polygon_split_window_32
rslearn dataset prepare --root $DATASET_PATH --group $DATASET_GROUP --workers 64 --retry-max-attempts 8
rslearn dataset materialize --root $DATASET_PATH --group $DATASET_GROUP --workers 64 --retry-max-attempts 8
```

**WorldCover:**
```bash

export DATASET_GROUP=worldcover_window_32
rslearn dataset prepare --root $DATASET_PATH --group $DATASET_GROUP --workers 64 --retry-max-attempts 8
rslearn dataset materialize --root $DATASET_PATH --group $DATASET_GROUP --workers 64 --retry-max-attempts 8
```

---

## Step 3. Finetune Helios

**Sentinel-2 only (12 months), ws=4, ps=1**
```bash
python -m rslp.main olmoearth_pretrain launch_finetune --image_name favyen/rslphelios10 --config_paths+=data/helios/v2_nandi_crop_type/finetune_s2_20250815.yaml --cluster+=ai2/titan-cirrascale --project_name 2025_08_15_nandi_crop_type --run_name nandi_crop_type_segment_helios_base_S2_ts_ws4_ps1_bs8
```

---

## Step 4. Make Predictions

**Create 128×128 windows for the county in 2023:**
```bash
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250616
rslearn dataset add_windows --root $DATASET_PATH --group nandi_county --utm --resolution 10 --grid_size 128 --src_crs EPSG:4326 --box=34.6999,-0.114,35.4549,0.5672 --start 2023-03-01T00:00:00+00:00 --end 2023-03-31T00:00:00+00:00 --name nandi
```

**Run prediction:**
```bash
python -m rslp.main olmoearth_pretrain launch_finetune --image_name favyen/rslphelios10 --config_paths+=data/helios/v2_nandi_crop_type/finetune_s2_20250815.yaml --cluster+=ai2/saturn-cirrascale --mode predict --gpus 4 --run_name nandi_crop_type_segment_helios_base_S2_S1_ts_ws4_ps1_bs8_add_annotations_2 --project_name 2025_08_15_nandi_crop_type
```

---

## Changelog

### 2025-07-29

As noted in `data/helios/v2_nandi_crop_type/README.md`, for inference we switched to a segmentation task, converting vector labels into raster labels (only the center pixel is valid).
```bash
python rslp/nandi/create_label_raster.py --ds_path=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250815
```

### 2025-08-15

Performance on key crops like Coffee was still low. To improve this, we used all pixels per polygon (instead of just 10), which added ~1K more Coffee samples and improved Coffee precision. The sampled dataset is `20250625` (8K windows) and the full dataset is `20250815` (21K windows). We also kept minor classes (Vegetables and Legumes) for complete category coverage at inference.

---

### 2025-09-10

To address misclassification of Trees → Coffee, we added extra Tree polygons via Studio (mainly from South Nandi Forests: https://earth-system-studio-dev.allen.ai/tasks/73daa5e4-08b4-4500-aac0-b9e43a9dea8a/5589b78d-cbeb-4be1-a264-7ed69a89d003#9.6/0.2315/35.0802) and converted them to 10 m points. We sampled 2K more Tree points, created windows, and used them for another finetuning round. The config `finetune_s2_20250815.yaml` already includes this group.

Create windows for extra annotations
```bash
python rslp/nandi/create_windows_for_additional_annotations.py --csv_path=/weka/dfive-default/yawenz/datasets/CGIAR/20250910_10m_pixels.csv --ds_path=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250815 --group_name 20250912_annotations --window_size=32
```

Although S1 + S2 gives slightly higher accuracy than S2 only, the S1 images introduce noticeable artifacts during inference. Therefore, we use the S2-only model for final predictions. The final predictions need to be further merged into a single geotiff via `rslp/nandi/scripts/merge_geotiff.py` and cleaned up via `rslp/nandi/scripts/cleanup_geotiff.py`. The cleanup tool can be used to smooth edges and remove small islands in the final maps.

---

### 2025-09-12

Added AEF embeddings into `20250625` dataset.

---

### 2025-09-16

Worked on olmoearth_run inference, see `olmoearth_run_data/nandi` for more details.
