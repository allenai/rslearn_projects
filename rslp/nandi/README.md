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
python -m rslp.main helios launch_finetune --image_name favyen/rslphelios10 --config_paths+=data/helios/v2_nandi_crop_type/finetune_s2_20250815.yaml --cluster+=ai2/titan-cirrascale --rslp_project 2025_08_15_nandi_crop_type --experiment_id nandi_crop_type_segment_helios_base_S2_ts_ws4_ps1_bs8
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
python -m rslp.main helios launch_finetune --image_name favyen/rslphelios10 --config_paths+=data/helios/v2_nandi_crop_type/finetune_s1_s2_20250815.yaml --cluster+=ai2/saturn-cirrascale --mode predict --gpus 4 --experiment_id nandi_crop_type_segment_helios_base_S2_S1_ts_ws4_ps1_bs8_add_annotations_2 --rslp_project 2025_08_15_nandi_crop_type
```

---

## Changelog

### 2025-08-15

As noted in `data/helios/v2_nandi_crop_type/README.md`, for inference purpose, we changed to segmentation task, where the vector labels need to be converted to raster labels (only the center point is valid).
```bash
python raslp

- Added back **Vegetables** and **Legumes**.  
- Use all points within polygons (19K total).  
- Category stats:

| Category   | Count |
|------------|------:|
| Sugarcane  |  3870 |
| Maize      |  3126 |
| Tea        |  2906 |
| Grassland  |  2724 |
| Trees      |  2586 |
| Coffee     |  2533 |
| Legumes    |   977 |
| Vegetables |   325 |

- New finetune run:  
```bash
python -m rslp.main helios launch_finetune   --image_name favyen/rslphelios10   --config_paths+=data/helios/v2_nandi_crop_type/finetune_s2_20250815.yaml   --cluster+=ai2/titan-cirrascale   --rslp_project 2025_08_15_nandi_crop_type   --experiment_id nandi_crop_type_segment_helios_base_S2_ts_ws4_ps2_bs8
```

---

### 2025-09-10
- Added **extra annotations** (Trees).  
- Script:
```bash
python rslp/crop/kenya_nandi/create_windows_for_additional_annotations.py   --csv_path=/weka/dfive-default/yawenz/datasets/CGIAR/20250910_10m_pixels.csv   --ds_path=/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250815   --group_name 20250912_annotations   --window_size=32
```

---

### 2025-09-12
- Use **20K Tree points** instead of 2K.  
- Try adding Sentinel-1.  
- Compare with **AEF embeddings** on Nandi task.

---

### 2025-09-16
- Copied weights & re-wrote checkpoint keys:
```bash
gsutil cp gs://rslearn-eai/projects/2025_08_15_nandi_crop_type/.../last.ckpt /weka/dfive-default/yawenz/test/2025_09_16_nandi_checkpoints
python /weka/dfive-default/yawenz/test/20250905_rewrite_checkpoint.py
```
