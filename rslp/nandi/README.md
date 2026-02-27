# Nandi County Crop Type Classification

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
