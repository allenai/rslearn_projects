# Global Mangrove Watch (GMW)

For this project, there're two tasks: 1. Mangrove classification, i.e. predict each point as `Mangrove`, `Water`, or `Other`. An initial mask is applied to ensure training uses only points within mangrove regions, and all the training samples are from 2020, 2. Mangrove loss detection, GMW already uses an NDVI-based scoring system to generate loss alerts (from 2021 to 2025), but it still produces some false alarms.

## Task 1: Mangrove Classification

- Mangrove ground-truth data (about 5M): `/weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_v4_points.csv`
- Mangrove reference data: `/weka/dfive-default/yawenz/datasets/Mangrove/csv/gmw_reference_points.csv`

We haven't used the reference data yet, but they're separate points validated by experts and also hold out for validation.

### 1. Create Windows

We sampled 100K points from the original 5M points. Run the following command to create windows:
```
python rslp/mangrove/create_windows_for_classification.py --csv_path /weka/dfive-default/yawenz/datasets/mangrove/csv/gmw_v4_points.csv --ds_path /weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250626 --group_name sample_100K --window_size 32
```

### 2. Prepare/Materialize Windows

Run the following commands to prepare and materialize windows:
```
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/mangrove/classification/20250626
export DATASET_GROUP=sample_100K
rslearn dataset prepare --root DATASET_PATH --group DATASET_GROUP --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
python rslp/scripts/launch_beaker_data_materialization.py --project mangrove_classification --ds_path DATASET_PATH --group DATASET_GROUP --image_name favyen/rslp --clusters ai2/ceres-cirrascale --num_jobs 10
```

### 3. Finetune Helios

Each point represents an 20x20m area, so by default, we set window_size = 2. Run the following command to finetune Helios for mangrove classification.

```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 2 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_mangrove_classification/finetune_s2.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_mangrove_classification_helios_base_S2_ts_ws2_ps2
```

By changing the `num_samples` in the `train_config`, we can train Helios with different number of samples but evaluate on the same validation set.

## Task 2: Mangrove Loss Detection

From 2021 to April 2025, there were 575K true alerts and 94K false alerts in total. Each alert is a point representing 20x20m area on the ground (i.e., a Sentinel2 pixel where the 10m bands were resampled to the 20m bands). The baseline method is based on a scoring system that applies thresholding to the NDVI time series, as described in [paper](https://doi.org/10.3390/rs15082050).

As a starting point, we can randomly select equal number of true and false alerts to form a training set, and then finetune Helios model to classify if the input time-series (Sentinel1 + Sentinel2) indicating a Mangrove loss or not.

| Year | True Positives | False Positives | Precision  |
|------|----------------|-----------------|------------|
| 2025 | 34,026         | 224             | 99.34%     |
| 2024 | 186,881        | 7,169           | 96.31%     |
| 2023 | 165,912        | 67,522          | 71.08%     |
| 2022 | 106,648        | 3,574           | 96.76%     |
| 2021 | 82,151         | 15,840          | 83.83%     |

We split train/val by years (188K samples in total), using 2021-2023 to train, and 2024-2025 to validate.

### 1. Create Windows

Run the following command to create windows:
```
python /weka/dfive-default/yawenz/rslearn_projects/rslp/mangrove/create_windows_for_loss_detection.py --true_positives_csv_path /weka/dfive-default/yawenz/datasets/mangrove/csv/gmw_2021_202504_true_positives.csv --false_positives_csv_path /weka/dfive-default/yawenz/datasets/mangrove/csv/gmw_2021_202504_false_positives.csv --ds_path /weka/dfive-default/rslearn-eai/datasets/mangrove/loss_detection/20250626 --window_size 32
```

### 2. Prepare/Materialize Windows

Run the following commands to prepare and materialize windows:
```
export DATASET_PATH=/weka/dfive-default/rslearn-eai/datasets/mangrove/loss_detection/20250626
export DATASET_GROUP=sample_188K_temporal_split
rslearn dataset prepare --root DATASET_PATH --group DATASET_GROUP --workers 64 --no-use-initial-job --retry-max-attempts 8 --retry-backoff-seconds 60
python rslp/scripts/launch_beaker_data_materialization.py --project mangrove_loss_detection --ds_path DATASET_PATH --group DATASET_GROUP --image_name favyen/rslp --clusters ai2/titan-cirrascale --num_jobs 15
```

### 3. Finetune Helios

Run the following commands to finetune Helios for alert classification:
```
python -m rslp.main helios launch_finetune --helios_checkpoint_path /weka/dfive-default/helios/checkpoints/favyen/v0.2_base_latent_mim_128_alldata_random_fixed_modality_0.5/step320000 --patch_size 2 --encoder_embedding_size 768 --image_name favyen/rslphelios3 --config_paths+=data/helios/v2_mangrove_loss/finetune_s2.yaml --cluster+=ai2/saturn-cirrascale --rslp_project 2025_06_26_helios_finetuning --experiment_id v2_mangrove_loss_helios_base_S2_ts_ws2_ps2
```
