# Mozambique LULC and Crop Type Classification

This project has two main tasks:
	1.	Land Use/Land Cover (LULC) and cropland classification
	2.	Crop type classification

The annotations come from field surveys across three provinces in Mozambique: Gaza, Zambezia, and Manica.

For LULC classification, the train/test splits are:
- Gaza: 2,262 / 970
- Manica: 1,917 / 822
- Zambezia: 1,225 / 525

## LULC Classification

#### 2025-11-05

Updates so that it works with all the changes. Also now a forward pass does a patch not a pixel; this makes inference far faster.

```
python -m rslp.main olmoearth_pretrain launch_finetune --image_name gabrielt/rslpomp_20251027b --config_paths+=data/helios/v2_mozambique_lulc/finetune_s2_20251024.yml --cluster+=ai2/saturn --rslp_project 2025_09_18_mozambique_lulc --experiment_id mozambique_lulc_helios_base_S2_ts_ws4_ps1_gaza_20251105_saturn_b
```

#### 2025-10-23

Update S1 and S2 training scripts to run with all the updates. This also requires running `python -m rslp.main olmoearth_pretrain` instead of `python -m rslp.main helios`:

```
python -m rslp.main olmoearth_pretrain launch_finetune --image_name favyen/favyen/rslpomp20251022a --config_paths+=data/helios/v2_mozambique_lulc/finetune_s2.yaml --cluster+=ai2/neptune --rslp_project 2025_09_18_mozambique_lulc --experiment_id mozambique_lulc_helios_base_S2_ts_ws4_ps1_gaza_20251023
```

Also - the geometry for Gaza province was enormous (hundreds of thousands of points). I have drawn a cruder polygon around the province for the prediction request geometry to try and keep things manageable.

#### Original commands
```
python /weka/dfive-default/yawenz/rslearn_projects/rslp/crop/mozambique/create_windows_for_lulc.py --gpkg_dir /weka/dfive-default/yawenz/datasets/mozambique/train_test_samples --ds_path /weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc --window_size 32

export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/crop/mozambique_lulc
rslearn dataset prepare --root $DATASET_PATH --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
python -m rslp.main common launch_data_materialization_jobs --image favyen/rslp_image --ds_path $DATASET_PATH --clusters+=ai2/neptune-cirrascale --num_jobs 5

python -m rslp.main helios launch_finetune --image_name favyen/rslphelios10 --config_paths+=data/helios/v2_mozambique_lulc/finetune_s1_s2.yaml --cluster+=ai2/neptune --rslp_project 2025_09_18_mozambique_lulc --experiment_id mozambique_lulc_helios_base_S1_S2_ts_ws4_ps1_gaza
```
