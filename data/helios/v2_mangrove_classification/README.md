# Mangrove Classification (v2)

## Overview

4-class segmentation task (0=invalid, 1=Mangrove, 2=Water, 3=Other) using OlmoEarth V1 Base
encoder fine-tuned on Sentinel-2 imagery. Each sample is a single 20x20m point (window_size=2
at 10m resolution) with 12 monthly S2 composites over a year.

Source data: [Global Mangrove Watch v4](https://zenodo.org/records/17394267).
Remote deployment config: https://github.com/allenai/olmoearth_projects/tree/main/olmoearth_run_data/mangrove

## Config Files

| File | Purpose |
|------|---------|
| `config.json` | Dataset layer definition (copy to dataset root as `config.json`) |
| `finetune_olmoearth_s2.yaml` | Training config using `rslearn.models.olmoearth_pretrain.model.OlmoEarth` |
| `finetune_s2_20250915.yaml` | Training config using internal `rslp.helios.model.Helios` |
| `finetune_s2.yaml` | Older template-based config (3-class classification, vector labels) |

### Key differences: `finetune_olmoearth_s2.yaml` vs `finetune_s2_20250915.yaml`

| | `finetune_olmoearth_s2.yaml` | `finetune_s2_20250915.yaml` |
|---|---|---|
| Encoder | `rslearn.models.olmoearth_pretrain.model.OlmoEarth` with `model_id` | `rslp.helios.model.Helios` with hardcoded checkpoint |
| Decoder | `rslearn.models.pooling_decoder.SegmentationPoolingDecoder` | `rslp.nandi.train.SegmentationPoolingDecoder` |
| Normalization | `OlmoEarthNormalize` (auto-loads defaults) | `HeliosNormalize` (needs `/opt/helios/data/norm_configs/computed.json`) |

Both produce the same model architecture. Use `finetune_olmoearth_s2.yaml` for new runs.

## Original Data Locations (Weka)

### Classification

| Data | Path |
|------|------|
| Ground truth (~5M points) | `/weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_v4_points.csv` |
| Reference (expert-validated) | `/weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_reference_points.csv` |
| Existing 100K sample dataset | `/weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250626` |

### Loss Detection

| Data | Path |
|------|------|
| True positives | `/weka/dfive-default/yawenz/datasets/mangrove/csv/gmw_2021_202504_true_positives.csv` |
| False positives | `/weka/dfive-default/yawenz/datasets/mangrove/csv/gmw_2021_202504_false_positives.csv` |
| Existing dataset | `/weka/dfive-default/rslearn-eai/datasets/mangrove/loss_detection/20250626` |

### Public download (eval)

```
https://storage.googleapis.com/ai2-olmoearth-projects-public-data/evals/partner_tasks/mangrove.tar
```

### CSV format

The `gmw_v4_points.csv` has columns: `x` (longitude), `y` (latitude), `fid` (sample ID),
`ref_cls` (1=Mangrove, 2=Water, 3=Other). The create_windows script samples 100K from the
full 5M and splits train/val by SHA-256 hash of the window name.

## Full Pipeline: Sample and Train a New Model

### 1. Set up dataset directory

```bash
export NEW_DS=/weka/dfive-default/rslearn-eai/datasets/mangrove/classification/$(date +%Y%m%d)
mkdir -p $NEW_DS
cp data/helios/v2_mangrove_classification/config.json $NEW_DS/config.json
```

### 2. Create windows (sample 100K points)

```bash
python rslp/mangrove/create_windows_for_classification.py \
  --csv_path /weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_v4_points.csv \
  --ds_path $NEW_DS \
  --group_name sample_100K \
  --window_size 32
```

### 3. Create label rasters

Converts vector labels to center-pixel rasters (class IDs 0-3).

```bash
python rslp/mangrove/create_label_raster.py --ds_path $NEW_DS
```

### 4. Prepare and materialize (fetch Sentinel-2 imagery)

```bash
rslearn dataset prepare \
  --root $NEW_DS \
  --group sample_100K \
  --workers 64 \
  --no-use-initial-job \
  --retry-max-attempts 8 \
  --retry-backoff-seconds 60

python rslp/scripts/launch_beaker_data_materialization.py \
  --project mangrove_classification \
  --ds_path $NEW_DS \
  --group sample_100K \
  --image_name favyen/rslp \
  --clusters ai2/ceres-cirrascale \
  --num_jobs 10
```

### 5. Update dataset path in training config

Edit `finetune_olmoearth_s2.yaml` and set the `path` field to your new dataset:

```yaml
data:
  init_args:
    path: /weka/dfive-default/rslearn-eai/datasets/mangrove/classification/YYYYMMDD
```

### 6. Train

```bash
python -m rslp.main olmoearth_pretrain launch_finetune \
  --config_paths+=data/helios/v2_mangrove_classification/finetune_olmoearth_s2.yaml \
  --cluster+=ai2/saturn-cirrascale \
  --image_name favyen/rslphelios3 \
  --project_name mangrove_classification \
  --run_name mangrove_olmoearth_base_S2_$(date +%Y%m%d)
```

Or run locally:

```bash
python -m rslp.main olmoearth_pretrain launch_finetune \
  --config_paths+=data/helios/v2_mangrove_classification/finetune_olmoearth_s2.yaml \
  --local \
  --project_name mangrove_classification \
  --run_name mangrove_olmoearth_base_S2_local
```

### 7. Analyze original data (optional)

To explore the source CSV before sampling:

```python
import pandas as pd

df = pd.read_csv("/weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_v4_points.csv")
df.rename(columns={"x": "longitude", "y": "latitude"}, inplace=True)

print(f"Total points: {len(df)}")
print(f"Class distribution:\n{df['ref_cls'].value_counts()}")
# ref_cls: 1=Mangrove, 2=Water, 3=Other

# Geographic spread
print(f"Lat range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
print(f"Lon range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")
```

## Model Architecture

```
OlmoEarth V1 Base (768-dim, frozen until epoch 20)
  └── SegmentationPoolingDecoder (768 → 4 classes)
       └── SegmentationHead → per-pixel probabilities
```

- Encoder: OlmoEarth V1 Base (patch_size=2), frozen for first 20 epochs then unfrozen at 10x lower LR
- Decoder: SegmentationPoolingDecoder pools globally then copies to all pixels (works for tiny windows)
- LR: 1e-4 with ReduceLROnPlateau (factor=0.2, patience=2, cooldown=10)
- Batch size: 32, 100 epochs, 10K training samples per epoch
- Normalization: OlmoEarthNormalize (mean ± 2*std per S2 band)
- Metrics: per-class precision/recall for Mangrove (1), Water (2), Other (3), plus micro accuracy

## Changelog

### 20250915
Created label rasters, updated to segmentation head in config `finetune_s2_20250915.yaml`.

### 20260407
Added `config.json` (dataset config using MOSAIC mode, replaces old 12-layer config).
Added `finetune_olmoearth_s2.yaml` (uses canonical rslearn OlmoEarth classes instead of
internal rslp.helios wrappers, no deprecated APIs).
